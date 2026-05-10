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
§5 — The Conditioning Problem (static)

The OLS loss is quadratic, so its level sets are ellipses with semi-axes
``1/√λᵢ`` along the eigenvectors of the Hessian.  The *visual aspect ratio*
of the ellipse is therefore exactly ``√κ`` — the square root of the
condition number.

This figure shows the level-set ellipse of the same OLS problem under two
parameter coordinate systems:

  Left   — raw columns        (a, b)        — Hessian H = Xᵀ X
  Right  — Jacobi-normalised  (α, β)        — Hessian H̃ = X̃ᵀ X̃, X̃ = X·D⁻¹

Both panels are drawn in eigenvector-aligned coordinates with
``set_aspect("equal")`` and the same axis range, so the visible eccentricity
of each ellipse is *exactly* ``√κ`` — no coordinate-system trickery.  The
shape difference IS the conditioning problem.

Annotated quantities:
  * κ  — condition number (eigenvalue ratio of the Hessian)
  * √κ — visible axis ratio of the ellipse
  * col-max ratio max|x₁| / max|x₂| — the source of the asymmetry,
    explicitly equalised by Jacobi normalisation
"""

import matplotlib.pyplot as plt
import numpy as np

from visuals.common import (
    C_BLUE,
    C_GREEN,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    SIZE_2D,
    save,
)


def _build_design_matrix() -> tuple[np.ndarray, np.ndarray]:
    x1 = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2 = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    y = np.array([MEASURED_RSS_MB[s] * 1024**2 for s in PROBE_SHAPES], dtype=float)
    return np.column_stack([x1, x2]), y


def _kappa(design_matrix: np.ndarray) -> float:
    """Condition number of the Hessian Xᵀ X."""
    eigvals = np.linalg.eigvalsh(design_matrix.T @ design_matrix)
    return float(eigvals[-1] / max(eigvals[0], 1e-300))


def _format_kappa(kappa: float) -> str:
    if kappa < 1e3:
        return f"{kappa:.2f}"
    exp = int(np.floor(np.log10(kappa)))
    mant = kappa / 10**exp
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def _fill_log_loss_field(ax: plt.Axes, kappa: float, n_grid: int = 600) -> None:
    """Render log-scaled quadratic-form loss field (u² + κ·v²) on the panel."""
    u = np.linspace(-1.2, 1.2, n_grid)
    v = np.linspace(-1.2, 1.2, n_grid)
    u_grid, v_grid = np.meshgrid(u, v)
    quad = u_grid**2 + kappa * v_grid**2
    log_quad = np.log10(quad + 1.0)
    ax.contourf(u_grid, v_grid, log_quad, levels=18, cmap="viridis")
    ax.contour(
        u_grid,
        v_grid,
        log_quad,
        levels=18,
        colors="white",
        linewidths=0.4,
        alpha=0.30,
    )


def _draw_level_set(ax: plt.Axes, kappa: float, level: float = 1.0) -> None:
    """Plot the ellipse u² + κ·v² = level (drawn explicitly for emphasis)."""
    theta = np.linspace(0, 2 * np.pi, 720)
    a = np.sqrt(level)
    b = np.sqrt(level / kappa)
    ax.fill(
        a * np.cos(theta),
        b * np.sin(theta),
        color=C_RED,
        alpha=0.18,
        zorder=4,
    )
    ax.plot(
        a * np.cos(theta),
        b * np.sin(theta),
        color=C_RED,
        lw=2.0,
        zorder=5,
        label=f"$L = {level:g}$ level set",
    )


def _draw_principal_axes(ax: plt.Axes, kappa: float) -> None:
    """Draw the two principal-axis arrows of the ellipse from the optimum."""
    long_len = 1.0
    short_len = 1.0 / np.sqrt(kappa)
    ax.annotate(
        "",
        xy=(long_len, 0),
        xytext=(-long_len, 0),
        arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.6, alpha=0.85),
        zorder=6,
    )
    ax.text(
        0.95,
        -0.06,
        "$v_{\\min}$ (long)",
        color=C_BLUE,
        fontsize=8,
        va="top",
        ha="right",
        zorder=11,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    if short_len > 0.04:
        ax.annotate(
            "",
            xy=(0, short_len),
            xytext=(0, -short_len),
            arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.6, alpha=0.85),
            zorder=6,
        )
        ax.text(
            0.06,
            short_len + 0.04,
            "$v_{\\max}$ (short)",
            color=C_GREEN,
            fontsize=8,
            va="bottom",
            ha="left",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="none"),
        )
    else:
        ax.text(
            0,
            0.30,
            (
                f"short axis = $1/\\sqrt{{\\kappa}} \\approx {short_len:.1e}$\n"
                "(below pixel resolution — ellipse collapses to a line)"
            ),
            color="#7a0000",
            fontsize=8.5,
            va="bottom",
            ha="center",
            zorder=11,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.92,
                edgecolor="#7a0000",
            ),
        )


def _annotate_panel(ax: plt.Axes, kappa: float, col_ratio: float, caption: str) -> None:
    """Place the κ / √κ / column-norm annotation block on the panel."""
    block = (
        f"$\\kappa = {_format_kappa(kappa)}$\n"
        f"$\\sqrt{{\\kappa}} \\approx {_format_kappa(np.sqrt(kappa))}$\n"
        f"col-max ratio $= {_format_kappa(col_ratio)}$"
    )
    ax.text(
        0.04,
        0.96,
        block,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.92,
            edgecolor="lightgray",
        ),
    )
    ax.text(
        0.5,
        -0.18,
        caption,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="center",
        style="italic",
        color="#444444",
    )


def _plot_panel(
    ax: plt.Axes,
    kappa: float,
    col_ratio: float,
    title_main: str,
    caption: str,
) -> None:
    """Render one conditioning panel in eigenvector-aligned coords."""
    _fill_log_loss_field(ax, kappa)
    _draw_level_set(ax, kappa, level=1.0)
    _draw_principal_axes(ax, kappa)

    ax.plot(
        0,
        0,
        marker="*",
        color="white",
        markersize=18,
        markeredgecolor="black",
        markeredgewidth=0.9,
        zorder=10,
        label="OLS optimum $\\theta^*$",
    )
    _annotate_panel(ax, kappa, col_ratio, caption)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("displacement along $v_{\\min}$  (long-axis direction)")
    ax.set_ylabel("displacement along $v_{\\max}$  (short-axis direction)")
    ax.set_title(title_main, fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)


def main() -> None:
    x_raw, _ = _build_design_matrix()
    n_max = float(x_raw[:, 0].max())
    m_max = float(x_raw[:, 1].max())
    x_norm = x_raw / np.array([n_max, m_max])

    kappa_raw = _kappa(x_raw)
    kappa_norm = _kappa(x_norm)
    col_ratio_raw = m_max / n_max
    col_ratio_norm = 1.0  # by construction Jacobi makes column maxima equal

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=SIZE_2D)
    fig.suptitle(
        "The Conditioning Problem and its Jacobi-normalisation Fix\n"
        "OLS loss-level ellipses, eigenvector-aligned view  "
        "(visible aspect ratio $=\\sqrt{\\kappa}$)",
        fontsize=11.5,
        fontweight="bold",
    )

    _plot_panel(
        ax_raw,
        kappa_raw,
        col_ratio_raw,
        title_main="Raw columns:  $x_1 = B \\cdot S, \\; x_2 = B \\cdot S^2$",
        caption=(
            "Short axis $\\sim\\!26{,}000\\times$ smaller than long — the "
            "ellipse degenerates to a line.\n"
            "No fixed learning rate works for both directions; OLS solve is "
            "numerically degenerate."
        ),
    )

    _plot_panel(
        ax_norm,
        kappa_norm,
        col_ratio_norm,
        title_main="Jacobi-normalised:  $\\xi_i = x_i / \\max|x_i|$",
        caption=(
            "Long-to-short ratio $\\sim\\!7\\!:\\!1$ — workable ellipse.\n"
            "A single learning rate handles both $\\alpha$ and $\\beta$; the OLS "
            "solve is well-posed."
        ),
    )

    fig.tight_layout()
    path = save(fig, "fig04_loss_landscape_conditioning")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
