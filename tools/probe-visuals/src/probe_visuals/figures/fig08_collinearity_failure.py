# Copyright (c) 2026 J. Patrick Fulton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
fig08_collinearity_failure.py
§7 — Why Not Just Sweep One Direction

Two-panel comparison.
Panel A: chosen shape set — good Gram determinant, compact loss ellipse.
Panel B: hypothetical "all B=1, linear spacing" set — near-zero Gram det,
         loss valley stretches to infinity in one direction.

For each set we overlay the loss landscape L(a_hat, b_hat) in normalised
parameter space to visualise the ellipse / valley.
"""

import matplotlib.pyplot as plt
import numpy as np

from probe_visuals.common import A_FIT, B_FIT, MEASURED_RSS_MB, PROBE_SHAPES, SIZE_2D, save


def gram_det_normalised(shapes: list[tuple[int, int]]) -> float:
    """Gram det of normalised design matrix (gives numerical stability)."""
    x1 = np.array([b * s for b, s in shapes], dtype=float)
    x2 = np.array([b * s**2 for b, s in shapes], dtype=float)
    n_max = x1.max()
    m_max = x2.max()
    xi1 = x1 / n_max
    xi2 = x2 / m_max
    X = np.column_stack([xi1, xi2])
    G = X.T @ X
    return float(np.linalg.det(G))


def loss_grid_norm(
    shapes: list[tuple[int, int]],
    a_range: np.ndarray,
    b_range: np.ndarray,
    y_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loss on a 2-D (normalised alpha, beta) grid."""
    x1 = np.array([b * s for b, s in shapes], dtype=float)
    x2 = np.array([b * s**2 for b, s in shapes], dtype=float)
    n_max = x1.max() if x1.max() > 0 else 1
    m_max = x2.max() if x2.max() > 0 else 1
    xi1 = x1 / n_max
    xi2 = x2 / m_max
    A, B = np.meshgrid(a_range, b_range)
    # L(α,β) = Σ(y_i - α·ξ1_i - β·ξ2_i)²
    L = sum((y_vals[i] - A * xi1[i] - B * xi2[i]) ** 2 for i in range(len(shapes)))
    return A, B, L


def opt_alpha_beta(
    shapes: list[tuple[int, int]], y_vals: np.ndarray
) -> tuple[float, float]:
    x1 = np.array([b * s for b, s in shapes], dtype=float)
    x2 = np.array([b * s**2 for b, s in shapes], dtype=float)
    n_max = x1.max() if x1.max() > 0 else 1
    m_max = x2.max() if x2.max() > 0 else 1
    xi1 = x1 / n_max
    xi2 = x2 / m_max
    X = np.column_stack([xi1, xi2])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_vals, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def main() -> None:
    # ── Shape sets ────────────────────────────────────────────────────────────
    chosen_shapes = PROBE_SHAPES
    collinear_shapes = [(1, s) for s in [64, 512, 1024, 2048, 4096, 6144, 8192]]

    # y values (bytes)
    y_chosen = np.array([MEASURED_RSS_MB[sh] * 1024**2 for sh in chosen_shapes], dtype=float)
    y_collin = np.array(
        [A_FIT * s + B_FIT * s**2 for (_, s) in collinear_shapes], dtype=float
    )

    det_chosen = gram_det_normalised(chosen_shapes)
    det_collin = gram_det_normalised(collinear_shapes)

    a_c, b_c = opt_alpha_beta(chosen_shapes, y_chosen)
    a_k, b_k = opt_alpha_beta(collinear_shapes, y_collin)

    grid_pts = 300
    A_c_range = np.linspace(a_c * 0.3, a_c * 1.7, grid_pts)
    B_c_range = np.linspace(b_c * 0.3, b_c * 1.7, grid_pts)
    Ac, Bc, Lc = loss_grid_norm(chosen_shapes, A_c_range, B_c_range, y_chosen)

    A_k_range = np.linspace(a_k * 0.3, a_k * 1.7, grid_pts)
    B_k_range = np.linspace(b_k * 0.3, b_k * 1.7, grid_pts)
    Ak, Bk, Lk = loss_grid_norm(collinear_shapes, A_k_range, B_k_range, y_collin)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=SIZE_2D)
    fig.suptitle(
        "Collinearity Failure: Loss Landscape Comparison\n"
        "Chosen probe shapes vs. all B=1 shapes",
        fontsize=12,
        fontweight="bold",
    )

    plot_params = [
        (axes[0], Ac, Bc, Lc, det_chosen, "Chosen shapes\n(well-conditioned)", a_c, b_c),
        (
            axes[1],
            Ak,
            Bk,
            Lk,
            det_collin,
            "All B=1 shapes\n(near-collinear)",
            a_k,
            b_k,
        ),
    ]
    for ax, Ag, Bg, Lg, det, title_str, opt_a, opt_b in plot_params:
        Lg_log = np.log10(Lg + 1)
        ax.contourf(Ag, Bg, Lg_log, levels=25, cmap="RdYlGn_r")
        ax.contour(Ag, Bg, Lg_log, levels=25, colors="k", linewidths=0.3, alpha=0.35)
        ax.plot(opt_a, opt_b, "w*", ms=12, zorder=5, label="OLS optimum")
        ax.set_xlabel(r"$\alpha$  (normalised $a$)")
        ax.set_ylabel(r"$\beta$  (normalised $b$)")
        ax.set_title(title_str)
        ax.legend(fontsize=8)
        ax.text(
            0.03,
            0.03,
            f"Gram det ≈ {det:.2e}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            color="navy" if det > 1e-3 else "darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    axes[1].text(
        0.03,
        0.92,
        "Valley → b undetermined\nalong x1/x2 direction",
        transform=axes[1].transAxes,
        fontsize=8,
        va="top",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.85),
    )

    fig.tight_layout()
    path = save(fig, "fig08_collinearity_failure")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
