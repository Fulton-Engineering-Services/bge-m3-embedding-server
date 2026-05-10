# 4. OLS Fitting

After the probe sweeps its seven $(B, S)$ shapes, it has seven measurements $(x^1_i, x^2_i, y_i)$ where $x^1_i = B_i \cdot S_i$, $x^2_i = B_i \cdot S_i^2$, and $y_i$ is the observed RSS delta. The fitter must extract the two coefficients $(a, b)$ from those measurements. This page derives the closed-form ordinary least-squares (OLS) solution and justifies the two design choices that look unusual on first reading: no intercept, and exactly two parameters.

## OLS as a geometric problem

Given $n$ measurements $\{(B_i, S_i, y_i)\}$ and the two-coefficient model $y \approx a \cdot x^1 + b \cdot x^2$, the ordinary least-squares estimator $(\hat a, \hat b)$ minimises the sum of squared residuals:

$$
\mathcal{L}(a, b) \;=\; \sum_{i=1}^{n} \bigl( y_i \;-\; a \cdot x^1_i \;-\; b \cdot x^2_i \bigr)^2
$$

For two parameters and one linear equation per measurement, the solution is closed-form: solve a $2 \times 2$ system once and the result is a pair $(\hat a, \hat b)$. There is no iterative optimiser, no learning rate, no convergence test — only one matrix inversion and a few multiplications.

![Figure 3 — OLS best-fit plane through 7 probe measurements with residuals shown as vertical green segments.](../figures/startup-probe/fig03_ols_geometry.png)

Figure 3 visualises the fit as a 3-D problem. The seven probe measurements appear as coloured points in $(x^1, x^2, y)$ space. The shaded plane is the OLS best fit $y = a \cdot x^1 + b \cdot x^2$; it passes through the origin because the model has no intercept (justified below). The vertical green segments are the residuals $y_i - \hat y_i$ that OLS drives to a minimum-sum-of-squares solution. A perfect fit would place every point on the plane; real data carries small residuals due to RSS measurement noise, page granularity, and the model's inherent approximation error.

The plane is tilted because the two coefficients have very different magnitudes: $a$ is in the tens of thousands (bytes per token) while $b$ is single digits (bytes per token²). This tilt has consequences for the conditioning of the solver, addressed in §5.

### Animated version

![Figure 3a — Animation: the seven scatter points fade in sequentially, then the best-fit plane materialises through them, and finally the camera orbits to show the plane from multiple angles.](../figures/startup-probe/animated/fig03_ols_geometry_animated.gif)

Figure 3a constructs the fit incrementally: the seven points appear one at a time, the OLS plane materialises after the last point, and the camera then orbits to confirm that the surface is genuinely two-dimensional — flat, with no warping or curvature. The orbit reveals residuals that were hidden in the static image.

## Derivation of the closed form

In matrix notation, define:

- the design matrix $X \in \mathbb{R}^{n \times 2}$ with columns $x^1_i = B_i \cdot S_i$ and $x^2_i = B_i \cdot S_i^2$,
- the response vector $y \in \mathbb{R}^n$ with entries $y_i$,
- the parameter vector $\theta = (a, b)^\top$.

The least-squares solution satisfies the normal equations:

$$
X^\top X \, \theta \;=\; X^\top y
$$

For two columns $X^\top X$ is a $2 \times 2$ matrix:

$$
G \;=\; X^\top X \;=\; \begin{pmatrix} \sum (x^1_i)^2 & \sum x^1_i x^2_i \\ \sum x^1_i x^2_i & \sum (x^2_i)^2 \end{pmatrix}, \quad
X^\top y \;=\; \begin{pmatrix} \sum x^1_i y_i \\ \sum x^2_i y_i \end{pmatrix}
$$

The matrix $G$ is the *Gram matrix*; it is symmetric and positive semi-definite for any design $X$, and positive definite (hence invertible) iff $X$ has full column rank.

Cramer's rule gives the closed form:

$$
a \;=\; \frac{G_{22}\,(X^\top y)_1 \;-\; G_{12}\,(X^\top y)_2}{\det G}, \qquad
b \;=\; \frac{G_{11}\,(X^\top y)_2 \;-\; G_{12}\,(X^\top y)_1}{\det G}
$$

with $\det G = G_{11} G_{22} - G_{12}^2$. Two sums to compute the right-hand side, three sums for the Gram matrix, one division per coefficient: a few dozen lines of straight-line arithmetic, no iteration, no convergence test.

## Why no intercept

Workspace at $B = 0$ is identically zero — there is no `session.run()` to allocate for. The cost model is physically anchored at the origin: an empty batch costs nothing. Adding a free intercept $c$ to the model,

$$
y \;\approx\; c \;+\; a \cdot x^1 \;+\; b \cdot x^2
$$

would let the fit absorb the (already small) ORT-arena setup cost into a constant term that only matters at very small chunks where the bin-packer's behaviour is unaffected. Omitting the intercept keeps the model two-parameter and lets the small-batch regime be correctly under-estimated by exactly the constant the bin-packer does not need.

A second consideration is statistical. With an intercept, three parameters are fit from the same seven shapes. The residual degrees of freedom drop from $n - p = 5$ to $n - p = 4$, a $20\%$ reduction in the noise-rejection budget — for a parameter that captures something already modelled better elsewhere (the per-worker arena baseline, measured directly in §7).

Geometry, physics, and statistics all agree: anchor the plane at the origin.

## Why exactly two parameters

A two-parameter fit needs at least two non-degenerate data points; with more, OLS minimises the sum of squared residuals and the additional points serve as noise rejection. Seven probe shapes provide five degrees of freedom for residual estimation, sufficient to detect when a fit is suspect (large residuals trigger the fall-back path of §8).

Extending to three parameters (e.g., adding $c \cdot S^3$) would force more probe shapes — fitting $p$ parameters requires at least $p$ non-degenerate data points — would lengthen the sweep, and would mostly fit measurement noise. The two-regime decomposition of §2 already captures the dominant terms: linear (FFN/projection) and quadratic (attention).

If a future model architecture introduces a meaningfully cubic term — some long-context attention variants do — this is where to add it. Until then, two is the right number.

## Where the seven measurements come from

The probe sweeps six fixed shapes plus a dynamic $(1, \text{max\_seq})$ shape:

```
(1,   64)    linear anchor
(4,   64)    pairs with (1, 256) for direct b isolation
(1,  256)    linear anchor
(1, 1024)    mid-range
(1, 2048)    mid-range, anchors quadratic stability
(1, 4096)    quadratic anchor
(1, max_seq) dynamic — sized to the configured upper bound
```

Each shape contributes a distinct piece of information to the fit. §6 develops the information geometry — why these particular seven, and what would go wrong with sixteen or with an all-$B=1$ set.

## Code reference

The fitter lives in `src/probe/fit.rs`. The interesting twist — column normalisation to fix conditioning — is the subject of §5, but the core OLS solve has the structure:

```60:80:src/probe/fit.rs
    // (Pseudo-code for the unscaled solve, before the Jacobi normalization
    // covered on the next page. See src/probe/fit.rs for the actual implementation,
    // which solves in normalized [0,1] space and unscales at the end.)
    //
    // let mut g11 = 0.0_f64;
    // let mut g12 = 0.0_f64;
    // let mut g22 = 0.0_f64;
    // let mut gy1 = 0.0_f64;
    // let mut gy2 = 0.0_f64;
    //
    // for dp in data {
    //     let x1 = (dp.batch * dp.seq) as f64;
    //     let x2 = (dp.batch * dp.seq * dp.seq) as f64;
    //     let y  = dp.rss_delta as f64;
    //     g11 += x1 * x1;
    //     g12 += x1 * x2;
    //     g22 += x2 * x2;
    //     gy1 += x1 * y;
    //     gy2 += x2 * y;
    // }
    //
    // let det = g11 * g22 - g12 * g12;
    // let a = (g22 * gy1 - g12 * gy2) / det;
    // let b = (g11 * gy2 - g12 * gy1) / det;
```

In production, the naïve solve carries a numerical pitfall at long sequence lengths. §5 is dedicated to that pitfall and its fix.

## Interpreting residuals

After fitting $(a, b)$, every measurement has a residual giving the gap between the model's prediction and the observed RSS:

$$
r_i = y_i - (a \cdot x^1_i + b \cdot x^2_i)
$$

Large residuals are diagnostic:

- A single large residual at one shape: that probe call hit unusual noise (page-cache settling, ORT arena jitter). OLS averages it out.
- Large residuals across all shapes: the model is systematically wrong — perhaps a third regime exists, perhaps RSS is not tracking workspace, perhaps the model variant has unusual memory behaviour. This is when the fall-back path of §8 takes over and the probe reverts to conservative defaults.

The fit-quality figure in §7 shows residuals plotted against the fitted curve.

## Onward

At the highest probe shape $(B = 1, S = 8192)$, the two design columns differ in magnitude by roughly $8000\times$: $x^1 = 8192$ while $x^2 = 67{,}108{,}864$. A direct solve is dominated by the larger column and fails the standard $\det G \geq \varepsilon \cdot \max(\mathrm{diag})^2$ stability check. §5 derives the one-line preconditioner that fixes this.

---

← [Previous: Bin-packing](03-bin-packing.md) | [↑ Series overview](../startup-probe.md) | [Next: Conditioning →](05-conditioning.md)
