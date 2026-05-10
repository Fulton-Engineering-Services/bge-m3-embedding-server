# Startup Workspace Probe — Theory and Implementation

**Audience:** operators tuning deployments, contributors editing `src/probe.rs` or `src/binpack.rs`, and reviewers auditing the auto-budget logic.

This document is the canonical primer for the **memory probe and quadratic cost-model fitter**. It explains the math (transformer workspace decomposition, ordinary least squares, normalized normal equations), the engineering choices (probe shape selection, RSS measurement, persistent caching, lock-free coefficient handoff), and the safety properties (clamping, capability check, graceful fallback).

If you only want to run the server, the [README](../README.md) and [architecture overview](architecture.md) are enough. This document explains *why* the probe exists and *how* it makes its decisions.

---

## Table of Contents

1. [Why a Probe at All? — The Workspace-Cost Problem](#1-why-a-probe-at-all--the-workspace-cost-problem)
2. [Where the Quadratic Comes From — Transformer Workspace Decomposition](#2-where-the-quadratic-comes-from--transformer-workspace-decomposition)
3. [The Cost Model and How the Bin-Packer Uses It](#3-the-cost-model-and-how-the-bin-packer-uses-it)
4. [Fitting Coefficients — Ordinary Least Squares Without Intercept](#4-fitting-coefficients--ordinary-least-squares-without-intercept)
5. [The Conditioning Problem at `MAX_SEQ_LENGTH=8192`](#5-the-conditioning-problem-at-max_seq_length8192)
6. [Column Normalization — A Jacobi Preconditioner for OLS](#6-column-normalization--a-jacobi-preconditioner-for-ols)
7. [Probe Shape Selection — Information Geometry for Two Coefficients](#7-probe-shape-selection--information-geometry-for-two-coefficients)
8. [Measurement Pipeline — Synthesizing Texts and Reading RSS](#8-measurement-pipeline--synthesizing-texts-and-reading-rss)
9. [Sanity Bounds and Capability Check — Why We Clamp](#9-sanity-bounds-and-capability-check--why-we-clamp)
10. [Persistent Coefficient Cache — Fingerprinting and Atomic Writes](#10-persistent-coefficient-cache--fingerprinting-and-atomic-writes)
11. [Background Execution and Lock-Free Handoff](#11-background-execution-and-lock-free-handoff)
12. [End-to-End Example — From Probe Run to `/health`](#12-end-to-end-example--from-probe-run-to-health)
13. [Failure Modes and Conservative Defaults](#13-failure-modes-and-conservative-defaults)
14. [Operator Quick Reference](#14-operator-quick-reference)
15. [References](#15-references)

---

## 1. Why a Probe at All? — The Workspace-Cost Problem

### 1.1 What "workspace" means here

For one ONNX `session.run()` call, three categories of memory exist:

| Category | When allocated | Lifetime | Scales with |
|----------|----------------|----------|-------------|
| **Model weights** | Once at session creation | Session lifetime | Model size only |
| **Activations / workspace** | Per `run()` call | Single call | Batch × sequence × layer count |
| **OS / runtime overhead** | Process boot | Process lifetime | Roughly constant |

The first and third are essentially fixed once the process is up. The middle one — the *transient* workspace allocated and freed by every `session.run()` call — is the one that can blow up unpredictably. This is what the probe measures.

The bin-packer's job is to make sure this transient workspace never exceeds a per-worker budget. To do that, the bin-packer needs a *prediction function*: given a hypothetical chunk of `count` texts padded to `max_seq` tokens, how much workspace will the next `session.run()` use?

### 1.2 Why static knobs don't work at long context

The previous design (`BGE_M3_ONNX_BATCH_SIZE`) used a single integer: "never call `run()` with more than this many texts." That works when the sequence length is fixed and small (the `max_length=512` era), because workspace is approximately linear in batch size.

At `MAX_SEQ_LENGTH=8192`, this approximation breaks. The dominant cost is the attention score tensor, whose size is `O(batch × seq²)`. Holding `batch=8` constant, going from `seq=512` to `seq=8192` increases the attention workspace by `(8192/512)² = 256×`. A static batch ceiling can't see that — it would either be pessimistic at short lengths (wasting throughput) or fatal at long lengths (OOM kill). The fix is to model both regimes explicitly and pick a *workspace ceiling* that the bin-packer enforces dynamically.

### 1.3 Why measure instead of compute

We could in principle compute the workspace from the ONNX graph plus the ORT execution plan: count attention layers, multiply by `[B, H, S, S] × dtype_size`, add FFN intermediates, etc. We don't, for three reasons:

1. **ORT's arena allocator and graph optimizations change the numbers.** Constant folding, layer fusion, in-place ops, and EP-specific subgraph rewrites all change peak workspace from what the static graph would suggest. The arena's reuse policy further means peak ≠ sum-of-tensors.
2. **The model variant matters.** Fp32, fp16-with-Cast-nodes, and int8-with-DequantizeLinear all have different intermediate footprints even at the same `(batch, seq)` shape. A model-aware static analysis would mean re-deriving constants every time the variant set changes.
3. **The hardware matters.** Page granularity, NUMA placement, and even kernel version affect RSS accounting. The probe measures the *actual* RSS delta on *this* host running *this* model, which subsumes all the above.

A measured cost model is also self-documenting: the fitted `(a, b)` pair and the measurement source appear in `/health`, so operators see exactly what budget the server is using.

---

## 2. Where the Quadratic Comes From — Transformer Workspace Decomposition

BGE-M3 is a 24-layer XLM-RoBERTa-style transformer with 16 attention heads, hidden dim `D = 1024`, and FFN intermediate dim `D_ff = 4096`. For every `session.run()` call with batch `B` and padded sequence length `S`, ORT has to materialize, per layer:

| Tensor | Shape | Size (fp32 bytes) |
|--------|-------|-------------------|
| Q / K / V projections | `[B, S, D]` × 3 | `3 · B · S · D · 4` |
| Attention scores | `[B, H, S, S]` | `B · H · S² · 4` |
| Attention output | `[B, S, D]` | `B · S · D · 4` |
| FFN intermediate | `[B, S, D_ff]` | `B · S · D_ff · 4` |
| FFN output | `[B, S, D]` | `B · S · D · 4` |

Stripping the constants and grouping by how the size grows with `(B, S)`:

```text
linear-in-S terms:    B · S · (3D + D + D_ff + D) · 4 = B · S · k₁
quadratic-in-S term:  B · S² · (H · 4)              = B · S² · k₂
```

Summed across 24 layers and ignoring sub-leading constants:

$$
W(B, S) \;\approx\; a \cdot (B \cdot S) \;+\; b \cdot (B \cdot S^2)
$$

This is the cost model in [`src/binpack.rs`](../src/binpack.rs):

```17:30:src/binpack.rs
/// where `a` (bytes/token-position) captures the FFN / projection contribution
/// and `b` (bytes/token-position^2) captures the attention contribution.
///
/// At sequence length 512 attention is small relative to FFN, so a linear
/// approximation works. At 8192, `b * N^2` dominates by ~16×, so using only
/// `a` would under-budget by that same factor.
///
/// Coefficients are derived at startup by [`crate::probe`] or set
/// conservatively from compile-time defaults when measurement is unavailable.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct CostModel {
    /// Bytes per token-position (linear term: FFN intermediates, projections).
    pub a: f64,
    /// Bytes per token-position-squared (quadratic term: attention scores).
    pub b: f64,
```

### 2.1 The crossover point

The two terms are equal when `S = a / b`. With typical fitted values `a ≈ 18 KiB/token` and `b ≈ 6 B/token²`, the crossover sits around `S ≈ 3000`. Below that the FFN/projection term dominates; above it the attention term takes over. This is exactly the regime where a single linear knob fails.

| `S` | linear term `a · B · S` | quadratic term `b · B · S²` | ratio |
|-----|-------------------------|------------------------------|-------|
| 512 | 9.4 MB · B | 1.5 MB · B | 0.16× |
| 2048 | 38 MB · B | 25 MB · B | 0.66× |
| 4096 | 75 MB · B | 100 MB · B | 1.3× |
| 8192 | 150 MB · B | 400 MB · B | 2.7× |

At `S = 8192`, the quadratic term is roughly 16× larger than at `S = 2048`. Any planner that ignores it under-budgets by exactly that factor.

### 2.2 What we are *not* modeling

The two-coefficient model deliberately omits:

- **Constant per-call overhead** (ORT setup, arena initialization). This is a fixed offset; we absorb it into the linear term by not adding an intercept.
- **Sub-leading polynomial terms** (e.g., `S³` from any cubic fusion patterns). At BGE-M3's scale these are negligible; including them would require more probe shapes for marginal gain.
- **Concurrency effects.** The cost model is per-`session.run()`; per-worker concurrency is already serialized inside one ORT session. Multi-worker effects come from the global memory budget, not the cost model itself.

The model is "wrong but useful" in the George Box sense: it captures the two regimes that actually drive OOMs, with one parameter per regime.

---

## 3. The Cost Model and How the Bin-Packer Uses It

The fitted `(a, b)` are wrapped in a `CostModel` along with `max_workspace_bytes` (the per-worker workspace ceiling derived from container memory):

```32:60:src/binpack.rs
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

```78:88:src/binpack.rs
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

The bin-packer (`bin_pack` in the same file) sorts texts by sequence length, then greedily packs them into chunks while `fits(chunk_size, current_max_seq)` holds. Because attention is quadratic in `max_seq`, packing texts of similar length together is worth far more than naive count-based batching: the chunk-local max only inflates when a *long* text joins, which is exactly when adding more texts becomes expensive.

This means the cost model has to be accurate in both regimes: under-estimating `b` causes long-text OOMs, while over-estimating `a` wastes throughput on short-text batches. That asymmetry — slow vs crash — motivates the asymmetric clamps in §9.

---

## 4. Fitting Coefficients — Ordinary Least Squares Without Intercept

We have `n` measurements `{(B_i, S_i, y_i)}` where `y_i` is the observed RSS delta. We want `(a, b)` minimizing the sum of squared residuals:

$$
\min_{a, b} \;\sum_{i=1}^{n} \Bigl( y_i \;-\; a \cdot (B_i \cdot S_i) \;-\; b \cdot (B_i \cdot S_i^2) \Bigr)^2
$$

This is ordinary least squares (OLS) with two columns and no intercept. Using the standard linear-algebra form, define:

- design matrix `X ∈ ℝ^(n×2)` with columns `x¹_i = B_i · S_i` and `x²_i = B_i · S_i²`,
- response vector `y ∈ ℝⁿ` with entries `y_i`,
- parameter vector `θ = (a, b)ᵀ`.

The least-squares solution satisfies the **normal equations**:

$$
X^\top X \, \theta \;=\; X^\top y
$$

For our 2-column case `XᵀX` is a 2×2 matrix:

$$
G \;=\; X^\top X \;=\; \begin{pmatrix} \sum (x^1_i)^2 & \sum x^1_i x^2_i \\ \sum x^1_i x^2_i & \sum (x^2_i)^2 \end{pmatrix}, \quad
X^\top y \;=\; \begin{pmatrix} \sum x^1_i y_i \\ \sum x^2_i y_i \end{pmatrix}
$$

Cramer's rule gives the closed-form solution:

$$
a \;=\; \frac{G_{22}\,(X^\top y)_1 \;-\; G_{12}\,(X^\top y)_2}{\det G}, \qquad
b \;=\; \frac{G_{11}\,(X^\top y)_2 \;-\; G_{12}\,(X^\top y)_1}{\det G}
$$

with `det G = G₁₁ G₂₂ - G₁₂²`.

### 4.1 Why no intercept

Workspace at `B = 0` is identically zero — there's no `session.run()` to allocate for. Adding a free intercept would let the fit absorb the (already small) ORT-arena setup cost into a constant term, which only matters for very small chunks where it doesn't hurt anything. Omitting it keeps the model two-parameter and the small-batch regime correctly underestimated by exactly the constant we don't care about.

### 4.2 Why two parameters and no more

A 2-parameter fit needs at least 2 data points. With more, OLS minimizes the sum-of-squares — extra points serve as noise rejection rather than as additional degrees of freedom. The 7 probe shapes (§7) give us 5 degrees of freedom for residual estimation, which is sufficient to detect when a fit is suspect (large residuals → fall back to defaults, see §13).

Going to three parameters (e.g., adding `c · S³`) would force more shapes, more probe time, and would mostly fit measurement noise. The two-regime decomposition in §2 already captures the dominant terms.

---

## 5. The Conditioning Problem at `MAX_SEQ_LENGTH=8192`

The naïve solver above works perfectly when the design matrix columns are of similar magnitude. They are not.

For our highest probe shape `(B=1, S=8192)`:

- column 1: `B · S = 8 192`
- column 2: `B · S² = 67 108 864`

The columns differ by **roughly 8000×** in scale at the upper end. This causes two real numerical problems even though the data is mathematically full-rank:

### 5.1 The Gram matrix is dominated by one entry

The diagonal entries scale like `Σ(x^k_i)²`. With `x²_i ≈ 8000 · x¹_i`, we get `G_22 ≈ 64 000 000 · G_11`. The determinant `det G = G_11 G_22 - G_12²` is then small *relative to* `G_22²`, which is what numerical-stability checks compare it against.

### 5.2 Singularity threshold becomes scale-dependent

A common defensive check is `|det G| ≥ ε · max(diag)²`. With unscaled columns, `max(diag)² = G_22²` is enormous and the threshold rejects perfectly-good full-rank fits. Lowering `ε` masks the scale problem but also masks actual ill-conditioning when probe shapes happen to align (e.g., all data at the same `B/S` ratio).

The condition number of the unnormalized Gram matrix grows roughly as `(S_max / S_min)⁴`. At `S_max = 8192`, `S_min = 64`, that's `128⁴ ≈ 2.7 × 10⁸`. Solving a 2×2 system at this condition number in `f64` is numerically fine (we keep ~7 significant digits), but stability *checks* that compare `det G` to `max(diag)²` see only the dominant entry.

Empirically, before the fix, the 16-shape sweep would silently fall back to conservative defaults despite valid data — a test in `probe.rs` (`fit_cost_model_production_scale_16_shapes_with_max_seq_8192`) reproduces this and then verifies the normalized version succeeds.

---

## 6. Column Normalization — A Jacobi Preconditioner for OLS

The fix is a textbook diagonal preconditioner. Define `n_max = max_i x¹_i` and `m_max = max_i x²_i`. Substitute scaled columns:

$$
\xi^1_i \;=\; \frac{x^1_i}{n_{\max}}, \qquad \xi^2_i \;=\; \frac{x^2_i}{m_{\max}}, \qquad \xi^k_i \in [0, 1]
$$

This is just the substitution `x = D ξ` where `D = diag(n_max, m_max)`. The new model is:

$$
y_i \;\approx\; \alpha \cdot \xi^1_i \;+\; \beta \cdot \xi^2_i \quad\text{with}\quad \alpha = a \cdot n_{\max},\;\; \beta = b \cdot m_{\max}
$$

Solve OLS in `(α, β)` space — both columns are now in `[0, 1]`, so the Gram-matrix entries are all `O(n)`-scale and the determinant test compares like-to-like:

```363:392:src/probe.rs
    // Build normalized Gram matrix: n1 = x1/x1_max, n2 = x2/x2_max ∈ [0,1].
    // Variable names use single-letter prefixes to avoid clippy::similar_names
    // on the longer accumulator names (g11, g12, g22, gy1, gy2).
    let mut g11 = 0.0_f64; // sum(n1²)
    let mut g12 = 0.0_f64; // sum(n1*n2)
    let mut g22 = 0.0_f64; // sum(n2²)
    let mut gy1 = 0.0_f64; // sum(n1*y)
    let mut gy2 = 0.0_f64; // sum(n2*y)

    for dp in data {
        let n1 = (dp.batch * dp.seq) as f64 / x1_max;
        let n2 = (dp.batch * dp.seq * dp.seq) as f64 / x2_max;
        let y = dp.rss_delta as f64;

        g11 += n1 * n1;
        g12 += n1 * n2;
        g22 += n2 * n2;
        gy1 += n1 * y;
        gy2 += n2 * y;
    }

    // 2×2 determinant in normalized space.
    // With n1, n2 ∈ [0,1], max_diag ≤ N and det is directly comparable.
    let det = g11 * g22 - g12 * g12;
    let max_diag_sq = g11.max(g22).powi(2);
    if det.abs() < 1e-6 * max_diag_sq {
        // Nearly singular — likely all data points at the same shape or
        // concentrated along one direction in design space.
        return None;
    }
```

After solving for `(α, β)`, unscale to recover `(a, b)`:

$$
a \;=\; \alpha / n_{\max}, \qquad b \;=\; \beta / m_{\max}
$$

This transformation is mathematically equivalent to the original OLS — both produce the same residual-minimizing `(a, b)` when the problem is well-posed. What changes is which inputs the *solver* sees: in `[0, 1]` space the Gram matrix's spectrum is condition-number-bounded by the geometry of the probe shape distribution, not by the absolute magnitudes.

This is exactly a Jacobi (diagonal) preconditioner — the simplest case of preconditioning where the diagonal of the design's column scales is folded out before solving. For this 2-coefficient problem it is sufficient; for larger systems one would generally use SVD-based pseudoinverse, but at `n=7, p=2` the closed-form Cramer's solution is overwhelmingly the right tool.

---

## 7. Probe Shape Selection — Information Geometry for Two Coefficients

A 2-coefficient OLS needs at least 2 distinct points; more points reduce noise but only if they carry *new information*. The probe sweeps **6 fixed shapes plus the configured `max_seq`**:

```49:57:src/probe.rs
const PROBE_SHAPES: &[Shape] = &[
    (1, 64),   // linear anchor
    (4, 64),   // pairs with (1,256) for direct b isolation
    (1, 256),  // linear anchor
    (1, 1024), // mid-range
    (1, 2048), // mid-range, improves stability condition
    (1, 4096), // quadratic anchor
               // (1, max_seq) is added dynamically based on configured max.
];
```

Each shape is chosen for a reason:

| Shape | Role | Purpose in the fit |
|-------|------|---------------------|
| `(1, 64)` | Linear anchor | Pure low-`S` regime; quadratic term is `b · 4096` ≈ 25 KB — negligible. Nails down `a`. |
| `(1, 256)` | Linear anchor | Same regime, longer arm. Confirms `a` and starts probing `b`. |
| `(4, 64)` | **`b`-isolator** | Has the same `B·S = 256` as `(1, 256)`, but `B·S² = 16384` vs `65536`. The two shapes share the linear column but differ on the quadratic column by 4×. The difference of their RSS deltas is almost purely a measurement of `b`. |
| `(1, 1024)` | Mid-range | Bridges linear and quadratic regimes. Improves leverage on `(a, b)` jointly. |
| `(1, 2048)` | Mid-range | Improves the stability condition of the (normalized) Gram matrix — adds spread along the diagonal of the design space. |
| `(1, 4096)` | Quadratic anchor | Quadratic term ~50% of total cost. Strong leverage on `b`. |
| `(1, max_seq)` | Quadratic anchor + capability check | Dominant `b` measurement at the upper end. Doubles as a fail-fast check that the model can actually run at `max_seq` (see §9). |

### 7.1 Why these and not others

A previous draft used 16 shapes including `(8, 64)`, `(8, 256)`, `(8, 1024)`, `(16, 64)`, `(16, 256)`, `(16, 512)`. Those were removed because:

1. **Probe time.** Each `session.run()` at large batch is expensive — especially on the slowest target architectures. Sweeping 16 shapes pushed total probe time on aarch64 MLAS into the tens of minutes.
2. **No information gain.** Once you have `(1, S)` for several `S` plus *one* `(B>1, S)` shape that breaks the pure `B=1` line, additional `(B>1, S)` shapes mostly add noise. The OLS fit weights them all equally; a single noisy 16-batch measurement can drag the coefficients more than its information contribution justifies.
3. **Conditioning.** All `(B>1, large S)` shapes lie roughly along the same direction in design-matrix space (`B·S ≈ B·S²/S` — collinear with the `(1, S)` line at fixed `S`). They don't add a new dimension, only repetition.

The chosen 7 give us:
- two clean low-`S` linear anchors at different effective `S`,
- one shape that *breaks* the `B=1` line in a controlled way (`(4, 64)` paired with `(1, 256)`),
- four `(1, S)` shapes spanning `64–8192` for quadratic leverage.

Geometrically: the shapes form a roughly L-shaped distribution in `(N, M) = (B·S, B·S²)` space — points along the `B=1` arc plus one off-arc point. That's the minimum-information geometry for separating linear from quadratic.

### 7.2 Skipping shapes that won't fit

Each shape is checked against a *conservative* model before dispatch:

```231:243:src/probe.rs
    for (batch, seq) in &shapes {
        let batch = *batch;
        let seq = *seq;

        // Skip shapes estimated to exceed the rss_ceiling by more than
        // conservative cost model says (avoids OOM mid-probe).
        if !conservative.fits(batch, seq) {
            info!(
                batch,
                seq, "Probe: skipping shape (estimated to exceed rss_ceiling)"
            );
            continue;
        }
```

This protects the probe itself from the OOM it's trying to predict. On a small container (e.g., 4 GB available with 1 worker), the conservative model will pre-rule out shapes like `(1, 8192)` and the fit will still produce reasonable coefficients from the surviving shapes.

---

## 8. Measurement Pipeline — Synthesizing Texts and Reading RSS

### 8.1 Synthesizing inputs

The probe needs `B` texts each tokenizing to approximately `S` tokens. The probe doesn't have a tokenizer handy (it lives in the worker), so it approximates: at ~4 chars/token for natural English, a `S`-token input is ~`4S` characters. The probe synthesizes batches by repeating curated corpus snippets and trimming:

```465:481:src/probe.rs
fn synthesize_texts(corpus: &[String], batch: usize, target_seq: usize) -> Vec<String> {
    let target_chars = target_seq.saturating_mul(4).max(16);
    (0..batch)
        .map(|i| {
            let base = &corpus[i % corpus.len()];
            // Repeat the base text until we have enough characters.
            let repeated = base.repeat((target_chars / base.len().max(1)).max(2) + 1);
            // Trim to target_chars bytes (not chars, but close enough for probing).
            let trimmed = if repeated.len() > target_chars {
                &repeated[..target_chars]
            } else {
                &repeated
            };
            trimmed.to_string()
        })
        .collect()
}
```

The corpus is the same fixture used by the benchmarks (`benches/fixtures/corpus.json`) — real production-shaped strings drawn from three databases. This matters: synthetic strings of a single repeated character can tokenize very differently from natural language (run-length compression, fewer subword splits), and we want the workspace measurement to reflect realistic ORT execution paths.

The resulting tokenized lengths are *approximate*. The tokenizer truncates to the configured `max_seq_length` upper bound, so the probe shape `(B, S)` becomes "B texts, each padded to at most S tokens" — which is exactly what the bin-packer needs to predict.

### 8.2 Measuring RSS

The probe wraps each `session.run()` with two RSS reads:

```582:606:src/embedder.rs
pub(crate) fn probe_run_dense(
    session: &Session,
    tokenizer: &Tokenizer,
    texts: Vec<String>,
    max_seq: usize,
) -> Result<ProbeResult> {
    let rss_before = sysinfo::read_process_rss_bytes().unwrap_or(0);
    // ... build_chunk_arrays, session.run() ...
    let rss_after = sysinfo::read_process_rss_bytes().unwrap_or(rss_before);
    Ok(ProbeResult {
        rss_before,
        rss_after,
    })
}
```

`read_process_rss_bytes()` parses field 1 of `/proc/self/statm` (RSS in pages) and multiplies by the page size (4096 on Linux/x86_64 and Linux/aarch64). The delta `rss_after - rss_before` is the RSS growth attributable to the call.

This is *not* the same as the peak workspace allocated during the call. RSS is a high-water mark *as observed at the moment of reading* — it captures pages still resident immediately after the call returns. ORT's arena allocator typically retains its high-water buffer rather than releasing pages back to the kernel, so the RSS delta is a good proxy for peak workspace **on the first allocation at a given shape** and a near-zero floor on repeat calls (the arena is already big enough).

The probe uses each shape exactly once for this reason: the first call at a shape grows the arena to that shape's peak, which the RSS delta captures. Subsequent calls at the same shape would show ~zero delta and be useless data points.

### 8.3 The Linux-only constraint

Both `cgroup` memory detection and `/proc/self/statm` are Linux-specific. On macOS:

- `detect_available_memory()` falls back to `sysctl hw.memsize` for total host RAM.
- `read_process_rss_bytes()` returns `None`, because reading task RSS requires `task_info` FFI, which conflicts with `unsafe_code = "forbid"`.

The probe logs a warning and uses conservative defaults on macOS. The native macOS deployment (LaunchAgent build) instead uses the CoreML-tuned plist defaults — see [deployment.md](deployment.md).

---

## 9. Sanity Bounds and Capability Check — Why We Clamp

Two safety layers wrap the OLS output.

### 9.1 Coefficient clamping

Even with a well-conditioned fit, measurement noise can produce values outside the physically reasonable range. We clamp:

```404:425:src/probe.rs
    // Reject negative coefficients — physically impossible.
    if a_raw < 0.0 || b_raw < 0.0 {
        return None;
    }

    // Clamp to sane operational ranges.
    // a: [4 KiB, 256 KiB] per token-position
    let a = a_raw.clamp(4_096.0, 262_144.0);
    // b: [0.01, 50_000] bytes per token-position^2
    let b = b_raw.clamp(0.01, 50_000.0);
```

| Coefficient | Lower bound | Upper bound | Rationale |
|-------------|-------------|-------------|-----------|
| `a` | 4 KiB / token | 256 KiB / token | Below 4 KiB is implausibly small — suggests measurement saturated to zero. Above 256 KiB would cause vacuous packing budgets at any realistic batch size. |
| `b` | 0.01 / token² | 50 000 / token² | Below ~0.01 makes the quadratic term negligible at `S=8192`; above 50 000 would force every long text into a single-element chunk. |

A clamp is a *graceful degradation*: a fit that lands outside these bounds still produces a valid cost model, just one that is less aggressive than the noise might have suggested. Without clamping, an outlier RSS reading could push `b` to e.g. `1e9` and effectively disable batching for any sequence longer than a few tokens.

### 9.2 Capability check at `(1, max_seq)`

The dynamically-added probe shape doubles as a fail-fast check:

```265:282:src/probe.rs
            Err(e) => {
                if seq == max_seq {
                    // The max_seq capability check failed — fail fast.
                    tracing::error!(
                        error = %e,
                        seq = max_seq,
                        model_hint = "Set BGE_M3_MODEL=fp32 or lower BGE_M3_MAX_SEQ_LENGTH",
                        "Probe: model failed at configured max_seq_length — \
                         variant may not support this sequence length"
                    );
                    // Propagate as warning; caller converts to startup failure.
                    warn!("Falling back to conservative cost model after capability check failure");
                    return (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B);
                }
                warn!(batch, seq, error = %e, "Probe shape failed; skipping");
            }
```

Some Xenova ONNX exports were generated with `max_position_embeddings=512`, which means an `(1, 8192)` `session.run()` will error before producing output. The probe runs that shape at startup; if it fails, the operator sees an actionable log message naming the variant and the env var to change. This is significantly better than discovering the limit on a production request, which would surface as a generic ORT error.

When this happens, the probe falls back to conservative defaults (with `ProbeStatus::Failed` reported in `/health`) rather than aborting the process. The server still serves requests at sequence lengths the model *can* handle.

---

## 10. Persistent Coefficient Cache — Fingerprinting and Atomic Writes

The probe takes ~120 s on a Fargate amd64 task at default settings, longer on slower architectures. That's a sunk cost on every cold start unless we cache the result.

### 10.1 The fingerprint

The cache file at `{BGE_M3_CACHE_DIR}/probe-coefficients.json` carries enough metadata to know when the cached `(a, b)` are still valid:

```72:82:src/probe.rs
#[derive(serde::Serialize, serde::Deserialize)]
struct ProbeCache {
    schema_version: u32,
    server_version: String,
    model: String,
    max_seq: usize,
    arch: String,
    fitted_at_unix: u64,
    a: f64,
    b: f64,
}
```

The cache key is the tuple `(schema_version, server_version, model, max_seq, arch)`. Any difference invalidates the cache entry. This is conservative — for example, a patch bump that doesn't touch ORT or the tokenizer technically doesn't need a re-probe, but invalidating broadly is safer than maintaining a hand-curated "compatible versions" list.

The fingerprint deliberately *excludes*:
- `BGE_M3_WORKERS`, `BGE_M3_MAX_BATCH`: don't affect per-call workspace.
- `BGE_M3_MEMORY_SAFETY_FACTOR`, `BGE_M3_AVAILABLE_MEMORY_BYTES`: affect `max_workspace_bytes` (computed from current memory + safety factor on each start) but not `(a, b)`.
- `BGE_M3_IDLE_TIMEOUT_SECS`: lifecycle policy only, no effect on workspace.

So changing memory or worker settings never invalidates the cache: only the probe-relevant tuple does.

### 10.2 Atomic writes

Cache writes use a temp-file-plus-rename to avoid partial-write corruption:

```168:189:src/probe.rs
    let final_path = cache_dir.join("probe-coefficients.json");
    let tmp_path = cache_dir.join("probe-coefficients.json.tmp");

    if let Err(e) = std::fs::write(&tmp_path, &json) {
        warn!(error = %e, path = %tmp_path.display(), "Failed to write probe cache temp file");
        return;
    }

    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        warn!(error = %e, "Failed to atomically rename probe cache file");
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }

    info!(
        path = %final_path.display(),
        a,
        b,
        "Probe coefficients cached to EFS"
    );
}
```

`rename(2)` on POSIX file systems is atomic — readers see either the old file or the new file, never a partial write. On the production EFS volume this means a server reading the cache during a probe-cache update will never see a half-written file.

A cache-write failure is *non-fatal*: the warning is logged but startup continues. The fitted coefficients are used in this run; the next cold start will re-probe.

### 10.3 The cache lifecycle

```
                ┌──────────────────────┐
                │ try_load_probe_cache │
                └─────────┬────────────┘
                          │
                  fingerprint match?
                  ┌───────┴────────┐
                 yes              no
                  │                │
    ┌─────────────▼──┐   ┌─────────▼─────────┐
    │ apply (a,b)    │   │ run probe sweep   │
    │ skip probe     │   │ fit coefficients  │
    │ status: cache_ │   │ save_probe_cache  │
    │ hit            │   │ status: complete  │
    └────────────────┘   └───────────────────┘
```

Setting `BGE_M3_DISABLE_PROBE_CACHE=1` forces a fresh probe even when a valid cache exists. Use this when validating a new deployment, after manual edits to the cache file, or when debugging probe behavior.

---

## 11. Background Execution and Lock-Free Handoff

The probe takes long enough (~120 s on a cache miss) that blocking startup on it would stall liveness probes and delay rolling-update completion. The implementation runs the probe in a Tokio task and updates the cost model atomically:

```161:194:src/main.rs
        } else {
            // Cache miss — launch background probe.
            state
                .probe_status
                .store(ProbeStatus::Running as u8, Ordering::Release);
            let state_bg = Arc::clone(&state);
            let model_variant_bg = model_variant_str.clone();
            tokio::spawn(async move {
                let (a, b) =
                    probe::run_probe(&state_bg.pool, cfg_max_seq, per_worker_workspace).await;
                let cm = CostModel {
                    a,
                    b,
                    max_workspace_bytes: per_worker_workspace,
                };
                info!(
                    a = cm.a,
                    b = cm.b,
                    max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
                    "Background probe complete — updating cost model"
                );
                state_bg.cost_model.store(Arc::new(cm));
                // Distinguish real fit from conservative fallback.
                let status = if (a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON
                    && (b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON
                {
                    ProbeStatus::Failed
                } else {
                    probe::save_probe_cache(&cache_dir, &model_variant_bg, cfg_max_seq, a, b);
                    ProbeStatus::Complete
                };
                state_bg.probe_status.store(status as u8, Ordering::Release);
                info!(probe_status = status.as_str(), "Probe status updated");
            });
        }
```

Two synchronization primitives make this safe:

### 11.1 `Arc<ArcSwap<CostModel>>` for the coefficients

`ArcSwap` is a wait-free pointer swap: every worker holds a clone of the same `Arc<ArcSwap<CostModel>>`, calls `.load()` to get a snapshot pointer when it needs to bin-pack, and the probe task calls `.store(Arc::new(cm))` once when it's done. Readers never block the writer; writers never block readers. The new cost model becomes visible on each worker's next `load()` — no restart, no message, no lock.

This means three things during a cold start:

1. **Workers start with conservative defaults.** The bin-packer over-budgets, packing slightly fewer texts per chunk than necessary. This is safe and slightly slower than optimal.
2. **The server flips `ready=true` *before* the probe finishes.** Liveness checks pass; production traffic can flow immediately at conservative pack ratios.
3. **The probe task finishes asynchronously.** The next `chunk_cost()` call sees the fitted coefficients via the next `ArcSwap` load.

The transition is lock-free and observation-consistent: any single worker either uses old-everywhere or new-everywhere within one bin-pack call.

### 11.2 `AtomicU8` for `probe_status`

The companion field tracks the probe lifecycle:

```12:50:src/state.rs
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    /// Probe not run — a cost-model override (`BGE_M3_DISABLE_AUTO_BUDGET`,
    /// `BGE_M3_TOKEN_BUDGET`, or explicit A/B env vars) was in effect.
    Disabled = 0,
    /// Probe is running in the background; workers are using conservative defaults.
    Running = 1,
    /// Probe completed successfully; fitted `(a, b)` are now active.
    Complete = 2,
    /// Probe failed or produced invalid coefficients; conservative defaults remain.
    Failed = 3,
    /// Probe was skipped — valid coefficients loaded from the EFS cache file.
    CacheHit = 4,
}
```

`probe_status` is exposed in `/health` so operators can distinguish "we just started up" from "the probe failed and we're stuck on conservative defaults":

| Status | Meaning |
|--------|---------|
| `disabled` | A cost-model override env var bypasses the probe entirely. |
| `running` | Probe in flight; workers using conservative defaults. |
| `complete` | Fitted `(a, b)` are active and have been written to the cache. |
| `failed` | Probe ran but the fit was invalid (singular system, capability-check failure, or all-conservative coefficients). Conservative defaults remain in effect. |
| `cache_hit` | Probe skipped because a fingerprint-matching cache file existed. |

---

## 12. End-to-End Example — From Probe Run to `/health`

A representative cold-start trace at default settings on a 28 GB Fargate task:

```
[INFO] Starting bge-m3-embedding-server bind=0.0.0.0:8081 workers=7 max_seq=8192 model=Fp16
[INFO] Phase 1/4 git: cloning model files                                  ┐
[INFO] Phase 4/4 saveAll: tokenizer + dense + sparse loaded                │ leader-first
[INFO] Leader worker ready, model cache warm (1/7)                         │ §10 cold-start.md
[INFO] Workers 1..7 loaded from warm cache                                 ┘
[INFO] Memory detected available_bytes=28991029248 source=cgroup_v2
[INFO] Estimated model RSS per worker model_rss_per_worker_mb=1100
[INFO] Probe cache fingerprint mismatch; will re-probe                     ┐
[INFO] Starting memory probe max_seq=8192 rss_ceiling_mb=2500              │ probe sweep
[INFO] Probe shape measured batch=1 seq=64 rss_delta_mb=2                  │ §8
[INFO] Probe shape measured batch=4 seq=64 rss_delta_mb=8                  │
[INFO] Probe shape measured batch=1 seq=256 rss_delta_mb=6                 │
[INFO] Probe shape measured batch=1 seq=1024 rss_delta_mb=27               │
[INFO] Probe shape measured batch=1 seq=2048 rss_delta_mb=68               │
[INFO] Probe shape measured batch=1 seq=4096 rss_delta_mb=210              │
[INFO] Probe shape measured batch=1 seq=8192 rss_delta_mb=720              ┘
[INFO] Probe: fitted cost model a=18432 b=6.2 data_points=7                ┐ OLS, §6
[INFO] Background probe complete — updating cost model                     │
[INFO] Probe coefficients cached to EFS                                    │ §10.2
[INFO] Probe status updated probe_status=complete                          ┘
[INFO] Models ready — accepting requests
```

After this, `GET /health` returns:

```json
{
  "status": "ok",
  "workers": { "live": 7, "total": 7 },
  "max_seq_length": 8192,
  "tuning": {
    "a_bytes_per_token": 18432.0,
    "b_bytes_per_token_sq": 6.2,
    "max_workspace_bytes": 2500000000,
    "probe_status": "complete",
    "memory_source": "cgroup_v2",
    "available_bytes": 28991029248,
    "model_rss_bytes_per_worker": 1100000000
  }
}
```

On the *next* cold start (cache hit), the probe sweep is skipped:

```
[INFO] Probe cache hit — skipping startup probe a=18432 b=6.2 fitted_at_unix=1746...
[INFO] Cost model loaded from EFS cache
[INFO] Models ready — accepting requests
```

…and `/health` reports `probe_status: "cache_hit"`.

---

## 13. Failure Modes and Conservative Defaults

The probe is designed so every failure path leads to a *functional* server with conservative budgets. The compile-time defaults are calibrated to match the legacy `BGE_M3_ONNX_BATCH_SIZE=16, MAX_SEQ_LENGTH=512` behaviour:

```44:52:src/binpack.rs
    pub const CONSERVATIVE_A: f64 = 16_384.0; // 16 KiB per token-position
    pub const CONSERVATIVE_B: f64 = 8.0; // 8 bytes per token-position^2

    /// Default maximum workspace per worker when memory cannot be detected.
    ///
    /// 2 GiB is conservatively safe for the Fargate 28 GiB task with 7 workers
    /// (`28 GB * 0.7 safety / 7 workers ≈ 2.8 GB`); we round down for headroom.
    pub const DEFAULT_MAX_WORKSPACE: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
```

Every failure path leads to one of these cells:

| Failure | What the probe does | What `probe_status` becomes | Server behaviour |
|---------|---------------------|------------------------------|-------------------|
| RSS reads return 0 (non-Linux) | Fit fails with empty data | `failed` | Conservative defaults; server functional |
| One probe shape errors mid-sweep | Skip that data point, continue | `complete` (if others succeed) | Fit may be slightly less accurate |
| `(1, max_seq)` errors (capability check) | Fail fast with hint log | `failed` | Conservative defaults; user reads log |
| OLS Gram is singular | `fit_cost_model` returns `None` | `failed` | Conservative defaults |
| Negative coefficient | `fit_cost_model` returns `None` | `failed` | Conservative defaults |
| Coefficient outside clamp | Clamp; log warning if difference >1% | `complete` | Fitted-but-clamped coefficients used |
| Cache file unreadable / truncated | Treat as miss; re-probe | normal lifecycle | Probe runs as if no cache |
| Cache write fails | Log warning, keep fitted coefficients | `complete` | Next cold start re-probes |

The asymmetry of conservatism is intentional: bin-packing under-counting is a slow service, bin-packing over-counting is no service. We over-count when in doubt.

---

## 14. Operator Quick Reference

### 14.1 Diagnosing probe state

`curl http://host:8081/health | jq '.tuning'` shows:
- `probe_status` — pinpoints which path was taken (cache_hit / complete / failed / running / disabled).
- `(a_bytes_per_token, b_bytes_per_token_sq)` — the live coefficients. Compare against typical values (~18 KiB / ~6 for fp16 amd64).
- `max_workspace_bytes` — the per-worker budget the bin-packer uses.

### 14.2 Forcing a fresh probe

```bash
BGE_M3_DISABLE_PROBE_CACHE=1 ./bge-m3-embedding-server
```

Bypasses the cache without affecting other behavior. Use when validating a new model variant or new container size.

### 14.3 Skipping the probe entirely

```bash
BGE_M3_DISABLE_AUTO_BUDGET=1 ./bge-m3-embedding-server
```

Server boots with conservative defaults immediately. Use for fast dev-loop iteration on macOS or when running smoke tests where probe time matters more than packing optimality.

### 14.4 Pinning explicit coefficients

```bash
BGE_M3_COST_MODEL_A=20000 BGE_M3_COST_MODEL_B=5.0 \
  BGE_M3_AVAILABLE_MEMORY_BYTES=10737418240 \
  ./bge-m3-embedding-server
```

All three must be set together — partial overrides are intentionally rejected (see `Config::from_env`). Use when reproducing a production incident locally with the same coefficients.

### 14.5 Legacy translation

`BGE_M3_ONNX_BATCH_SIZE` is deprecated; setting it logs a `WARN` and translates internally to `BGE_M3_TOKEN_BUDGET` (a workspace ceiling). Migrate to:

- `BGE_M3_TOKEN_BUDGET` for "give me roughly the same packing as before",
- the auto-budget (default) for "give me the best packing my container can support."

---

## 15. References

The probe combines standard numerical-linear-algebra techniques with transformer-architecture-specific cost reasoning. Useful background reading for contributors:

- **Ordinary least squares and the normal equations.** Trefethen & Bau, *Numerical Linear Algebra*, Lectures 11 (least squares) and 18 (conditioning of least squares).
- **Diagonal preconditioning / Jacobi scaling.** Saad, *Iterative Methods for Sparse Linear Systems*, ch. 10 — covers preconditioners; the simplest case is diagonal scaling, which is exactly what we do for the 2-column normal equations.
- **Condition number of a Gram matrix.** Golub & Van Loan, *Matrix Computations*, §5.3 (LS conditioning) and §3.5.4 (column scaling).
- **Transformer attention complexity.** Vaswani et al., *Attention Is All You Need* (NeurIPS 2017) — original derivation of the `O(B·S²)` attention term and `O(B·S·D)` projection terms.
- **ONNX Runtime arena allocator.** [`onnxruntime` docs on arena-based allocators](https://onnxruntime.ai/docs/api/c/struct_ort_arena_cfg.html) — explains why RSS deltas at first-touch overstate steady-state workspace and why we sample only once per shape.
- **`ArcSwap` and lock-free pointer swaps.** [`arc-swap` crate documentation](https://docs.rs/arc-swap/latest/arc_swap/) — the wait-free read-many, write-rarely primitive used for the cost-model handoff.
- **POSIX atomic rename semantics.** `rename(2)` on Linux man page — the basis for the cache file's atomic-write strategy.

For BGE-M3-specific details (model variants, hybrid retrieval, fp16/int8 trade-offs), see [bge-m3-model.md](bge-m3-model.md) and [model-variants.md](model-variants.md).
