# 3. Bin-Packing — How the Cost Model Gets Used

> The cost model from the previous page is just a function. This page shows how the bin-packer calls that function to turn an arbitrary stream of texts into a sequence of right-sized ONNX calls — and why packing similar-length texts together matters far more than packing similar *counts*.

## Intuition

Imagine you have a hundred texts of varying lengths and a single ONNX session that has a strict memory budget. You can't just feed all hundred at once — you'd OOM. You can't feed them one at a time either — that wastes the GPU/CPU's parallelism. You need to split them into chunks where each chunk's predicted workspace fits the budget.

The naive answer is "always pack 16 at a time." That works if every text is the same length. But our cost is `a·B·S + b·B·S²`, where `S` is the **maximum sequence length in the chunk** — every text gets padded up to the longest one. So a chunk of 16 texts with one 8192-token text inside it costs the same as 16 texts that are *all* 8192 tokens. One outlier blows up the bill for everyone.

The fix is to **sort by length first**, then pack greedily. Texts of similar length end up neighbors, the chunk-local max grows slowly, and the cost stays predictable. Adding a longer text to a chunk only makes the workspace explode if the chunk *already* has many texts in it — which is exactly when the bin-packer should split and start a fresh chunk.

This is the classic insight of bin-packing for variable-cost items: **sort first, then greedy**. Our twist is that the per-item cost depends on the chunk's max — so the predicate "does it still fit?" has to be re-checked with the *new* max every time we try to add a text.

## Setting up the cost model

The fitted `(a, b)` are wrapped in a `CostModel` along with `max_workspace_bytes` (the per-worker workspace ceiling derived from container memory):

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

Two predicates drive the bin-packer:

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

`chunk_cost` is the closed-form prediction. `fits` is the predicate the packer asks before adding any text to a chunk. Both use `u128` to avoid overflow at extreme `(B, S)` and a `saturating_add` to keep arithmetic well-defined even if a misconfigured override drives the prediction past `2^128 / 2`.

## The packing strategy

The bin-packer (`bin_pack` in the same file) does the following:

1. Tokenize every text once (a separate concern — see [Measurement](07-measurement.md)).
2. Sort the resulting list by tokenized length, ascending.
3. Walk through the sorted list, accumulating into a "current chunk." For each text:
   - Tentatively expand the chunk by adding this text. The chunk's new max-seq is `max(current_max, this_text_len)`.
   - Ask `cost_model.fits(chunk_size + 1, new_max_seq)`.
   - If yes: commit the text to the chunk.
   - If no: emit the current chunk to ORT, start a fresh chunk containing only this text.
4. Emit any final non-empty chunk.

Because attention is quadratic in `max_seq`, packing texts of similar length together is worth far more than naive count-based batching: the chunk-local max only inflates when a *long* text joins, which is exactly when adding more texts becomes expensive.

This means the cost model has to be accurate in both regimes: under-estimating `b` causes long-text OOMs, while over-estimating `a` wastes throughput on short-text batches. That asymmetry — slow vs crash — motivates the asymmetric clamps in [Clamps & fallback](08-clamps-fallback.md).

## A worked example

Suppose the probe has fitted `a = 18432` (B/token), `b = 6.2` (B/token²), and the workspace budget is `max_workspace_bytes = 2 GiB`. How many texts of `S = 4096` can fit in one chunk?

Solve `chunk_cost(B, 4096) ≤ 2 × 1024³`:

```
2 GiB = 2_147_483_648 B

chunk_cost(B, 4096) = 18432 × B × 4096   +   6.2 × B × 4096²
                    = 75_497_472 × B     +   104_018_739 × B
                    = 179_516_211 × B

B ≤ 2_147_483_648 / 179_516_211
B ≤ 11.96
B_max = 11 texts per chunk at S = 4096
```

Now do the same at `S = 8192`:

```
chunk_cost(B, 8192) = 18432 × B × 8192   +   6.2 × B × 8192²
                    = 150_994_944 × B    +   416_087_476 × B
                    = 567_082_420 × B

B ≤ 2_147_483_648 / 567_082_420
B ≤ 3.79
B_max = 3 texts per chunk at S = 8192
```

Doubling `S` reduces the per-chunk capacity by **3×** (11 → 3), not 2×. That's the quadratic term doing its work: at `S = 4096` the linear and quadratic terms are roughly equal in cost; at `S = 8192` the quadratic term dominates by ~2.7× and the per-chunk capacity drops accordingly.

What about the small end? At `S = 256`:

```
chunk_cost(B, 256)  = 18432 × B × 256    +   6.2 × B × 256²
                    = 4_718_592 × B      +     406_323 × B
                    = 5_124_915 × B

B ≤ 2_147_483_648 / 5_124_915
B ≤ 419
B_max = 419 texts per chunk at S = 256
```

That's 35× more than at `S = 4096`, even though the count of texts only changed by 16×. The bin-packer's whole job is to find this **B-vs-S surface** at every length and pack to the boundary.

## Why sort first

Suppose we *didn't* sort — just packed in arrival order. A chunk could look like `[short, short, short, LONG]`. The chunk's max-seq is now `LONG`, but we're paying the full `B·S²` cost for all four texts. This is a 4× waste relative to the same texts split into `[LONG]` and `[short, short, short]`.

In the worst case (a stream alternating short and long), unsorted packing approaches the cost of running every text at the maximum length. Sorted packing keeps each chunk's max-seq close to its members' actual lengths, recovering near-optimal throughput.

The sort itself is `O(N log N)` and runs once per request batch. It's free relative to the inference cost.

## Edge cases

The bin-packer handles a few easy-to-miss situations:

- **A single text that doesn't fit by itself** — even a chunk of size 1 can exceed the budget if the text is long enough and the budget is small enough. This is rejected upstream by the `validate` step that runs before bin-packing; the request fails with HTTP 413 rather than being silently truncated.
- **Empty input** — bin-packer returns an empty plan, the worker pool returns an empty result vector, the handler returns HTTP 200 with an empty `data` array.
- **All texts identical length** — the sort is a no-op, packing reduces to "fit as many as the budget allows at this length." This is the easy case the cost model nails.
- **Budget set below `chunk_cost(1, 1)`** — pathological configuration; the bin-packer would refuse every chunk. The conservative-defaults calibration in [Clamps & fallback](08-clamps-fallback.md) ensures the budget is always at least large enough for a single short text.

## Why this matters for the probe

The bin-packer is the **consumer** of the probe's output. Every decision the probe makes — which shapes to measure, how to fit the model, how to clamp the coefficients — is ultimately in service of giving `chunk_cost` and `fits` accurate numbers. If the probe over-estimates `b`, the packer leaves throughput on the table. If it under-estimates `b`, the packer overshoots the budget and the worker OOM-kills. The asymmetry of those failure modes is why the probe's clamping is asymmetric (page 8) and why the conditioning fix on page 5 is non-negotiable.

## Code reference

The full bin-packer is small enough to read end-to-end:

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

## What's next

We've now seen *what* the cost model is and *how* the bin-packer uses it. The next page goes to the heart of the matter: how the probe takes a handful of `(B, S, RSS_delta)` measurements and solves for the two coefficients `(a, b)`.

---

← [Previous: Cost decomposition](02-cost-decomposition.md) | [↑ Series overview](../startup-probe.md) | [Next: OLS fitting →](04-ols-fitting.md)
