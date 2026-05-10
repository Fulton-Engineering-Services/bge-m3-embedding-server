# 3. Bin-Packing

The cost model from §2 is a function. The bin-packer is its consumer: it transforms an arbitrary stream of texts of varying lengths into a sequence of right-sized `session.run()` calls, each predicted to fit within the per-worker workspace budget.

The key observation is that the cost of a chunk depends on the *maximum* sequence length in the chunk: every text in the chunk is padded up to the longest one, so a chunk of 16 texts containing one 8192-token text costs the same as 16 texts that are all 8192 tokens long. A single outlier inflates the bill for the whole chunk. The mitigation is a sort-then-greedy strategy: sort texts by length so that similar-length texts end up neighbours, then pack greedily, re-checking the fit predicate against the chunk-local maximum after each candidate addition.

## Setting up the cost model

The fitted $(a, b)$ pair is wrapped in a `CostModel` along with `max_workspace_bytes`, the per-worker workspace ceiling derived from container memory:

```46:59:src/binpack.rs
impl CostModel {
    /// Conservative static defaults calibrated so a `(16, 512)` chunk lands at
    /// ~140 MB workspace — matching the old static budget at the previous default
    /// `BGE_M3_ONNX_BATCH_SIZE = 16`, `MAX_SEQ_LENGTH = 512`.
    ///
    /// These are used when the probe cannot run (no ORT, no model, macOS without
    /// cgroup support) or when `BGE_M3_DISABLE_AUTO_BUDGET` is set.
    ///
    /// Formula check: 16 KiB/token × 16 × 512 + 8 B/token² × 16 × 512²
    ///   = 16384 × 8192 + 8 × 16 × 262144
    ///   = 134 217 728 + 33 554 432
    ///   = 167 772 160 ≈ 160 MB per chunk (workers run sequentially inside one worker).
    pub const CONSERVATIVE_A: f64 = 16_384.0; // 16 KiB per token-position
    pub const CONSERVATIVE_B: f64 = 8.0; // 8 bytes per token-position^2
```

Two predicates drive the packer:

```92:102:src/binpack.rs
    pub fn chunk_cost(&self, count: usize, max_seq: usize) -> u128 {
        let n = count as u128 * max_seq as u128;
        let linear = (self.a * n as f64) as u128;
        let quad = (self.b * n as f64 * max_seq as f64) as u128;
        linear.saturating_add(quad)
    }

    /// Returns `true` if the chunk fits within the workspace budget.
    pub fn fits(&self, count: usize, max_seq: usize) -> bool {
        self.chunk_cost(count, max_seq) <= self.max_workspace_bytes as u128
    }
```

`chunk_cost` is the closed-form prediction. `fits` is the predicate the packer evaluates before adding any text to a chunk. Both use `u128` to avoid overflow at extreme $(B, S)$ and a `saturating_add` to keep arithmetic well-defined even if a misconfigured override drives the prediction past $2^{128}/2$.

## The packing strategy

The bin-packer (`bin_pack` in the same file) proceeds as follows:

1. Tokenise every text once (a separate concern; see §7).
2. Sort the resulting list by tokenised length, ascending.
3. Walk through the sorted list, accumulating into a "current chunk." For each text:
   - Tentatively expand the chunk by adding this text. The chunk's new maximum sequence length is $\max(S_{\text{cur}},\; S_{\text{new}})$.
   - Evaluate `cost_model.fits(chunk_size + 1, new_max_seq)`.
   - If true, commit the text to the chunk; otherwise emit the current chunk to ORT and start a fresh chunk containing only this text.
4. Emit any final non-empty chunk.

Because attention is quadratic in `max_seq`, packing texts of similar length together is more valuable than count-based batching: the chunk-local maximum only inflates when a long text joins, which is exactly when adding more texts becomes expensive.

The cost model must therefore be accurate in both regimes. Under-estimating $b$ causes long-text OOMs, while over-estimating $a$ wastes throughput on short-text batches. That asymmetry — slow versus crash — motivates the asymmetric clamps of §8.

## A worked example

Suppose the probe has fitted $a = 18432$ (B/token), $b = 6.2$ (B/token²), and the workspace budget is `max_workspace_bytes = 2 GiB`. The number of texts of $S = 4096$ that fit in one chunk is the largest integer $B$ satisfying `chunk_cost`$(B, 4096) \leq 2 \cdot 1024^3$:

```
2 GiB = 2_147_483_648 B

chunk_cost(B, 4096) = 18432 × B × 4096   +   6.2 × B × 4096²
                    = 75_497_472 × B     +   104_018_739 × B
                    = 179_516_211 × B

B ≤ 2_147_483_648 / 179_516_211
B ≤ 11.96
B_max = 11 texts per chunk at S = 4096
```

Repeating at $S = 8192$:

```
chunk_cost(B, 8192) = 18432 × B × 8192   +   6.2 × B × 8192²
                    = 150_994_944 × B    +   416_087_476 × B
                    = 567_082_420 × B

B ≤ 2_147_483_648 / 567_082_420
B ≤ 3.79
B_max = 3 texts per chunk at S = 8192
```

Doubling $S$ reduces the per-chunk capacity by $3\times$ ($11 \to 3$), not $2\times$. The quadratic term is responsible: at $S = 4096$ the linear and quadratic terms are roughly equal in cost, while at $S = 8192$ the quadratic term dominates by ${\sim}2.7\times$ and per-chunk capacity falls accordingly.

At the small end, $S = 256$:

```
chunk_cost(B, 256)  = 18432 × B × 256    +   6.2 × B × 256²
                    = 4_718_592 × B      +     406_323 × B
                    = 5_124_915 × B

B ≤ 2_147_483_648 / 5_124_915
B ≤ 419
B_max = 419 texts per chunk at S = 256
```

The capacity at $S = 256$ is $35\times$ that at $S = 4096$, even though $S$ changed by a factor of 16. The bin-packer's task is to find this $B$-versus-$S$ surface at every length and pack to the boundary.

## Why sorting matters

Without sorting — packing in arrival order — a chunk could form as `[short, short, short, LONG]`. The chunk's max-seq is then `LONG`, and the full $B \cdot S^2$ cost is paid for all four texts: a $4\times$ waste relative to splitting into `[LONG]` and `[short, short, short]`.

In the worst case (a stream alternating short and long texts), unsorted packing approaches the cost of running every text at the maximum length. Sorted packing keeps each chunk's max-seq close to its members' actual lengths, recovering near-optimal throughput. The sort itself is $O(N \log N)$ and runs once per request batch — negligible relative to the inference cost.

## Edge cases

Several easy-to-miss cases are handled explicitly:

- **A single text that does not fit by itself.** Even a chunk of size 1 can exceed the budget if the text is long enough and the budget small enough. This is rejected upstream by the `validate` step that runs before bin-packing; the request fails with HTTP 413 rather than being silently truncated.
- **Empty input.** The bin-packer returns an empty plan, the worker pool returns an empty result vector, and the handler returns HTTP 200 with an empty `data` array.
- **All texts of identical length.** The sort is a no-op and packing reduces to "fit as many as the budget allows at this length" — the easy case the cost model nails.
- **Budget set below `chunk_cost(1, 1)`.** A pathological configuration; the bin-packer would refuse every chunk. The conservative-defaults calibration of §8 ensures the budget is always large enough for at least a single short text.

## Why this matters for the probe

The bin-packer is the consumer of the probe's output. Every decision the probe makes — which shapes to measure, how to fit the model, how to clamp the coefficients — exists to give `chunk_cost` and `fits` accurate numbers. If the probe over-estimates $b$, the packer leaves throughput on the table; if it under-estimates $b$, the packer overshoots the budget and the worker is OOM-killed. The asymmetry of those failure modes drives the asymmetric clamping of §8 and the non-negotiable conditioning fix of §5.

## Code reference

The full bin-packer is short enough to read end-to-end:

```60:130:src/binpack.rs
impl CostModel {
    /// Conservative cost model used when the probe cannot run.
    pub fn conservative(max_workspace_bytes: usize) -> Self {
        Self {
            a: Self::CONSERVATIVE_A,
            b: Self::CONSERVATIVE_B,
            max_workspace_bytes,
        }
    }

    /// Cost in bytes for a single ORT call with `count` texts padded to `max_seq`.
    pub fn chunk_cost(&self, count: usize, max_seq: usize) -> u128 {
        let n = count as u128 * max_seq as u128;
        let linear = (self.a * n as f64) as u128;
        let quad = (self.b * n as f64 * max_seq as f64) as u128;
        linear.saturating_add(quad)
    }

    /// Returns `true` if the chunk fits within the workspace budget.
    pub fn fits(&self, count: usize, max_seq: usize) -> bool {
        self.chunk_cost(count, max_seq) <= self.max_workspace_bytes as u128
    }
}
```

## Interactive exploration

The companion notebook for this section runs interactively in the browser via JupyterLite (no install required):

**[▶ Open Workspace Budget Calculator](https://fulton-engineering-services.github.io/bge-m3-embedding-server/lab/index.html?path=02_workspace_budget_calculator.ipynb)**

The notebook computes per-worker workspace and worst-case peak memory from operator-tunable parameters: worker count, per-worker model RSS, container memory, and safety factor. A traffic-light indicator flags configurations that approach the cgroup ceiling and a stacked bar chart decomposes the worst-case allocation.

To run locally instead:

```bash
cd tools/visuals
uv sync --group notebooks
uv run jupyter notebook notebooks/02_workspace_budget_calculator.ipynb
```

---

← [Previous: Cost decomposition](02-cost-decomposition.md) | [↑ Series overview](../startup-probe.md) | [Next: OLS fitting →](04-ols-fitting.md)
