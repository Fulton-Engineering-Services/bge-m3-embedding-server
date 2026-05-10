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
fig04_loss_landscape_conditioning.py
§5 — The Conditioning Problem

Two side-by-side contour plots of the OLS loss L(a, b) in (a, b) parameter space.

Left panel (raw):
  x1 = B·S,  x2 = B·S²  — columns differ by ~8000×.
  Contours are very elongated along the b-axis.
  A gradient-descent step from an off-optimum start overshoots wildly.

Right panel (normalized):
  ξ1 = x1/n_max,  ξ2 = x2/m_max  — both in [0,1].
  Contours are near-circular.  Same step lands near the optimum.

L(θ) = Σ (y_i - x1_i·a - x2_i·b)²  in raw coords
     = Σ (y_i - ξ1_i·(a·n_max) - ξ2_i·(b·m_max))²  in normalised coords
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from visuals.common import (
    A_FIT,
    B_FIT,
    C_BLUE,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    SIZE_2D,
    save,
)


def main() -> None:
    # ── Build design matrix ────────────────────────────────────────────────────
    x1_data = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_data = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    y_data = np.array([MEASURED_RSS_MB[shape] * 1024**2 for shape in PROBE_SHAPES], dtype=float)

    n_max = x1_data.max()  # = 1*8192 = 8192
    m_max = x2_data.max()  # = 1 * 8192² = 67,108,864

    xi1 = x1_data / n_max  # normalised features in [0,1]
    xi2 = x2_data / m_max

    # Raw optimum: coefficients in raw space
    a_opt = A_FIT
    b_opt = B_FIT

    # Normalised optimum: coefficients in normalised space
    # y_i = (a·n_max)·ξ1_i + (b·m_max)·ξ2_i  → α = a·n_max, β = b·m_max
    alpha_opt = A_FIT * n_max
    beta_opt = B_FIT * m_max

    # ── Loss function helpers ─────────────────────────────────────────────────
    def loss_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L(a, b) in raw parameter space."""
        residuals = (
            y_data[:, None, None]
            - a[None, :, :] * x1_data[:, None, None]
            - b[None, :, :] * x2_data[:, None, None]
        )
        return (residuals**2).sum(axis=0)

    def loss_norm(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """L(α, β) in normalised space (α = a·n_max, β = b·m_max)."""
        residuals = (
            y_data[:, None, None]
            - alpha[None, :, :] * xi1[:, None, None]
            - beta[None, :, :] * xi2[:, None, None]
        )
        return (residuals**2).sum(axis=0)

    # ── Grid extents ──────────────────────────────────────────────────────────
    a_range = np.linspace(a_opt * 0.2, a_opt * 1.8, 400)
    b_range = np.linspace(b_opt * 0.2, b_opt * 1.8, 400)
    A_grid, B_grid = np.meshgrid(a_range, b_range)
    L_raw = loss_raw(A_grid, B_grid)

    # Normalised grid
    alpha_range = np.linspace(alpha_opt * 0.2, alpha_opt * 1.8, 400)
    beta_range = np.linspace(beta_opt * 0.2, beta_opt * 1.8, 400)
    AL_grid, BL_grid = np.meshgrid(alpha_range, beta_range)
    L_norm = loss_norm(AL_grid, BL_grid)

    # ── Gradient step from a starting point ──────────────────────────────────
    start_a_raw = a_opt * 1.5
    start_b_raw = b_opt * 0.3

    def grad_raw(a: float, b: float) -> tuple[float, float]:
        r = y_data - a * x1_data - b * x2_data
        da = -2 * (r * x1_data).sum()
        db = -2 * (r * x2_data).sum()
        return da, db

    da, db = grad_raw(start_a_raw, start_b_raw)
    lr = 5e-21
    end_a_raw = start_a_raw - lr * da
    end_b_raw = start_b_raw - lr * db

    start_alpha = alpha_opt * 1.5
    start_beta = beta_opt * 0.3

    def grad_norm(alpha: float, beta: float) -> tuple[float, float]:
        r = y_data - alpha * xi1 - beta * xi2
        da = -2 * (r * xi1).sum()
        db = -2 * (r * xi2).sum()
        return da, db

    da_n, db_n = grad_norm(start_alpha, start_beta)
    lr_n = 5e-21
    end_alpha = start_alpha - lr_n * da_n
    end_beta = start_beta - lr_n * db_n

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE_2D)
    fig.suptitle(
        "OLS Loss Landscape  $L(\\theta) = \\sum_i (y_i - \\theta^\\top x_i)^2$\n"
        "Conditioning: raw columns vs. Jacobi-normalised columns",
        fontsize=12,
        fontweight="bold",
    )

    log_levels = 30

    # Left: raw
    L_raw_log = np.log10(L_raw + 1)
    ax1.contourf(A_grid, B_grid, L_raw_log, levels=log_levels, cmap="RdYlGn_r")
    ax1.contour(A_grid, B_grid, L_raw_log, levels=log_levels, colors="k", linewidths=0.3, alpha=0.4)
    ax1.plot(a_opt, b_opt, "w*", ms=12, zorder=5, label="Optimum")
    ax1.annotate(
        "",
        xy=(end_a_raw, end_b_raw),
        xytext=(start_a_raw, start_b_raw),
        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.0),
    )
    ax1.plot(start_a_raw, start_b_raw, "o", color=C_BLUE, ms=8, label="Start")
    ax1.plot(end_a_raw, end_b_raw, "s", color=C_RED, ms=8, label="After 1 step")
    ax1.set_xlabel("$a$  (bytes / token-position)")
    ax1.set_ylabel("$b$  (bytes / token-position²)")
    ax1.set_title("Raw columns\n(ellipses very elongated — ill-conditioned)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax1.text(
        0.03,
        0.04,
        "Step overshoots b-direction",
        transform=ax1.transAxes,
        fontsize=8,
        color="navy",
        va="bottom",
    )

    # Right: normalised
    L_norm_log = np.log10(L_norm + 1)
    ax2.contourf(AL_grid, BL_grid, L_norm_log, levels=log_levels, cmap="RdYlGn_r")
    ax2.contour(
        AL_grid,
        BL_grid,
        L_norm_log,
        levels=log_levels,
        colors="k",
        linewidths=0.3,
        alpha=0.4,
    )
    ax2.plot(alpha_opt, beta_opt, "w*", ms=12, zorder=5, label="Optimum")
    ax2.annotate(
        "",
        xy=(end_alpha, end_beta),
        xytext=(start_alpha, start_beta),
        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.0),
    )
    ax2.plot(start_alpha, start_beta, "o", color=C_BLUE, ms=8, label="Start")
    ax2.plot(end_alpha, end_beta, "s", color=C_RED, ms=8, label="After 1 step")
    ax2.set_xlabel(r"$\alpha = a \cdot n_{\max}$  (normalised)")
    ax2.set_ylabel(r"$\beta = b \cdot m_{\max}$  (normalised)")
    ax2.set_title("Jacobi-normalised columns\n(near-circular — well-conditioned)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
    ax2.text(
        0.03,
        0.04,
        "Same step lands near optimum",
        transform=ax2.transAxes,
        fontsize=8,
        color="darkgreen",
        va="bottom",
    )

    fig.tight_layout()
    path = save(fig, "fig04_loss_landscape_conditioning")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
