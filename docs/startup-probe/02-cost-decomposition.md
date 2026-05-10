# 2. Cost Decomposition

The form of the cost model — $W = a \cdot B \cdot S + b \cdot B \cdot S^2$ — is not chosen empirically. It follows from the per-layer tensor shapes inside a transformer of BGE-M3's class. This page derives the form, examines its behaviour as $B$ and $S$ vary, and identifies the regime in which each coefficient dominates.

## The two-term structure

A transformer is a stack of identical layers. Each layer maps a $[B, S, D]$ tensor of token embeddings (batch $B$, sequence length $S$, hidden dimension $D$) to another $[B, S, D]$ tensor for the next layer. Producing the output requires several intermediate tensors. Most of them are shaped $[B, S, \cdot]$ and grow linearly with the total number of tokens $B \cdot S$. One — the attention-score matrix — is shaped $[B, H, S, S]$ and carries a factor of $S^2$.

The two coefficients $a$ and $b$ are the per-token-position and per-token-position-squared workspace contributions on the active hardware running the active model. The probe measures both; the bin-packer respects both.

<div align="center">

<img src="../figures/startup-probe/fig01_cost_decomposition.png" width="900" alt="Figure 1 — Workspace cost decomposition: orange linear term, teal quadratic term, dashed total, with crossover at S ≈ 2,973 marked.">

</div>

Figure 1 plots the two cost components against sequence length $S$ at fixed $B = 1$ (linear axes on the left, log–log on the right). The orange curve is $a \cdot S$, the teal curve is $b \cdot S^2$, and the dashed black curve is their sum $W(1, S)$. The vertical gold line marks the crossover $S^* = a/b \approx 2{,}973$ where the two terms are equal.

Below $S^*$ the orange curve is taller and most workspace is feed-forward. Above $S^*$ the teal curve takes over and grows quadratically. By the time $S = 8192$ is reached, the quadratic term dominates the total by a factor of $2.7\times$ and the linear term has become a rounding error. A planner that assumes linear growth will pack too many texts at long $S$ and OOM-kill the worker; a planner that assumes quadratic growth will refuse to batch short texts and waste throughput. The two-coefficient model captures both regimes with the minimum number of parameters.

## The 3-D view

<div align="center">

<img src="../figures/startup-probe/fig02_workspace_surface.png" width="720" alt="Figure 2 — Surface of W(B, S) with the per-worker workspace budget shown as a horizontal plane intersecting the surface; the floor projection traces the feasible (B, S) region as a contour line.">

</div>

Figure 2 plots the same cost function as a surface over the $(B, S)$ plane, with workspace $W$ on the vertical axis. The translucent horizontal plane is the per-worker budget `max_workspace_bytes`. Where the surface crosses the plane, the corresponding chunk does not fit; the floor projection traces the `fits()` boundary as a contour.

The surface is a flat ramp at small $S$ (linear regime) and curls into a parabolic bowl as $S$ grows (quadratic regime). The budget plane is most easily crossed by sliding outward along the $S$ axis, which is why the bin-packer sorts texts by length and packs similar-length texts together (§3).

### Animated version

<div align="center">

<img src="../figures/startup-probe/animated/fig02_workspace_surface_animated.gif" width="840" alt="Figure 2a — Animated 360-degree rotation of the W(B, S) surface around its vertical axis, showing the budget plane intersection from every angle.">

</div>

Figure 2a rotates the camera azimuth around the vertical workspace axis. The intersection contour preserves its shape under rotation, but the geometry of the over-budget region — the part of the surface above the plane — is most easily seen when the camera is roughly perpendicular to the long axis of the bowl.

## Derivation from the per-layer tensor sizes

BGE-M3 is a 24-layer XLM-RoBERTa-style transformer with 16 attention heads, hidden dimension $D = 1024$, and FFN intermediate dimension $D_{\text{ff}} = 4096$. For one `session.run()` call at batch $B$ and padded sequence length $S$, ORT must materialise per layer:

| Tensor | Shape | Size (fp32 bytes) |
|--------|-------|-------------------|
| Q / K / V projections | $[B, S, D] \times 3$ | $3 \cdot B \cdot S \cdot D \cdot 4$ |
| Attention scores | $[B, H, S, S]$ | $B \cdot H \cdot S^2 \cdot 4$ |
| Attention output | $[B, S, D]$ | $B \cdot S \cdot D \cdot 4$ |
| FFN intermediate | $[B, S, D_{\text{ff}}]$ | $B \cdot S \cdot D_{\text{ff}} \cdot 4$ |
| FFN output | $[B, S, D]$ | $B \cdot S \cdot D \cdot 4$ |

Stripping constants and grouping by how the size grows with $(B, S)$:

```text
linear-in-S terms:    B · S · (3D + D + D_ff + D) · 4 = B · S · k₁
quadratic-in-S term:  B · S² · (H · 4)               = B · S² · k₂
```

Summing across 24 layers and absorbing sub-leading constants:

$$
W(B, S) \;\approx\; a \cdot (B \cdot S) \;+\; b \cdot (B \cdot S^2)
$$

This is the cost model in [`src/binpack.rs`](../../src/binpack.rs):

```26:44:src/binpack.rs
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
pub struct CostModel {
    /// Bytes per token-position (linear term: FFN intermediates, projections).
    pub a: f64,
    /// Bytes per token-position-squared (quadratic term: attention scores).
    pub b: f64,
```

## The crossover point

The two terms are equal at $S = a / b$. With typical fitted values $a \approx 18\,\text{KiB/token}$ and $b \approx 6\,\text{B/token}^2$, the crossover lies near $S \approx 3000$. Below that, the FFN/projection term dominates; above it, the attention term takes over. This is precisely the regime in which a single linear knob fails:

| $S$ | Linear $a \cdot B \cdot S$ | Quadratic $b \cdot B \cdot S^2$ | Ratio |
|-----|----------------------------|----------------------------------|-------|
| 512  | $9.4\,\text{MB} \cdot B$  | $1.5\,\text{MB} \cdot B$  | $0.16\times$ |
| 2048 | $38\,\text{MB} \cdot B$   | $25\,\text{MB} \cdot B$   | $0.66\times$ |
| 4096 | $75\,\text{MB} \cdot B$   | $100\,\text{MB} \cdot B$  | $1.3\times$  |
| 8192 | $150\,\text{MB} \cdot B$  | $400\,\text{MB} \cdot B$  | $2.7\times$  |

At $S = 8192$ the quadratic term is roughly $16\times$ larger than at $S = 2048$. Any planner that ignores it under-budgets by exactly that factor.

### Worked example: the cliff at $S = 8192$

Consider a chunk of $B = 4$ texts, all padded to $S = 8192$ tokens, with the fitted $a = 18432$, $b = 6.2$:

```
linear  = 18432 × 4 × 8192      ≈ 604 MB
quad    =   6.2 × 4 × 8192²     ≈ 1665 MB
total   ≈ 2.27 GB
```

The chunk consumes ${\sim}2.3\,\text{GiB}$ of workspace. With a per-worker budget of $2\,\text{GiB}$, the chunk does not fit despite holding only four texts. Under a static `BGE_M3_ONNX_BATCH_SIZE = 16` budget calibrated for $S = 512$ (where 16 texts at 512 tokens uses ${\sim}140\,\text{MB}$), this chunk would be admitted with a $16\times$ overshoot — the OOM scenario.

The same $B = 4$ at $S = 1024$:

```
linear  = 18432 × 4 × 1024      ≈ 75 MB
quad    =   6.2 × 4 × 1024²     ≈ 26 MB
total   ≈ 101 MB
```

Same number of texts, eight times shorter, $23\times$ cheaper. The bin-packer can safely pack $20\times$ more texts at $S = 1024$ than at $S = 8192$, but only when it knows the quadratic term is there.

## What the model deliberately omits

The two-coefficient model excludes:

- **Constant per-call overhead** (ORT setup, arena initialisation). This is a fixed offset; it is absorbed into the linear term by the no-intercept design (§4).
- **Sub-leading polynomial terms** (e.g., $S^3$ from any cubic fusion patterns). At BGE-M3's scale these are negligible; including them would require more probe shapes for marginal gain.
- **Concurrency effects.** The cost model is per-`session.run()`; concurrency inside one ORT session is already serialised. Multi-worker effects are accounted for by the global memory budget, not the cost model itself.
- **Page-level RSS jitter.** The probe measures process RSS at $4\,\text{KiB}$ granularity on Linux. A $4\,\text{KiB}$ measurement floor on a $2\,\text{GiB}$ chunk is a $0.0002\%$ noise floor — comfortably below what the OLS fit cares about.

The model is "wrong but useful" in the George Box sense: it captures the two regimes that drive OOMs, with one parameter per regime.

## Why two parameters

A two-parameter fit requires at least two data points; with more, OLS minimises the sum of squared residuals and additional points serve as noise rejection. The seven probe shapes provide five degrees of freedom for residual estimation, sufficient to detect when a fit is suspect (large residuals trigger the fall-back path of §8).

A three-parameter extension (such as adding a $c \cdot S^3$ term) would force more probe shapes, lengthen the sweep, and would mostly fit measurement noise. The two-regime decomposition above already captures the dominant terms — and two terms is the minimum number required to model both the FFN-dominated low-$S$ regime and the attention-dominated high-$S$ regime separately.

## Interactive exploration

The companion notebook for this section runs interactively in the browser via JupyterLite (no install required):

**[▶ Open Cost Decomposition Explorer](https://fulton-engineering-services.github.io/bge-m3-embedding-server/lab/index.html?path=01_cost_decomposition_explorer.ipynb)**

The notebook provides slider controls for $a$ (linear coefficient) and $b$ (quadratic coefficient). Moving the sliders shifts the crossover point $S^* = a/b$ in real time and updates both the linear-axes and log-log panels. Preset buttons load the fitted values $(a = 18{,}432,\; b = 6.2)$ and the conservative defaults $(a = 16{,}384,\; b = 8)$.

To run locally instead:

```bash
cd tools/visuals
uv sync --group notebooks
uv run jupyter notebook notebooks/01_cost_decomposition_explorer.ipynb
```

---

← [Previous: Overview](01-overview.md) | [↑ Series overview](../startup-probe.md) | [Next: Bin-packing →](03-bin-packing.md)
