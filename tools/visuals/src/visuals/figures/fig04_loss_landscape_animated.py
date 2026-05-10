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
fig04_loss_landscape_animated.py
§5 — The Conditioning Problem (animated)

Continuous morph from raw to Jacobi-normalised parameter coordinates.

A column-scaling matrix ``D(t) = diag(n_max^t, m_max^t)`` interpolates between
``D(0) = I`` (raw columns) and ``D(1) = diag(n_max, m_max)`` (full Jacobi
normalisation).  At each ``t`` the design matrix is ``X(t) = X_raw · D(t)⁻¹``
and the Hessian is ``H(t) = X(t)ᵀ X(t)``.  The condition number ``κ(t)``
decreases monotonically from ~6.7×10⁸ at t=0 to ~49 at t=1.

Visualisation is in eigenvector-aligned coordinates with
``set_aspect("equal")``: the OLS loss-level ellipse has aspect ratio exactly
``√κ(t)``.  As ``t`` morphs from 0 → 1 the ellipse expands from a degenerate
line into a workable ~7:1 ellipse.

Frames: 150 at 15 fps ≈ 10 s.
  * 20 frames hold at t=0 (raw)
  * 100 frames smoothstep morph t: 0 → 1
  *  30 frames hold at t=1 (Jacobi-normalised)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from visuals.common import (
    C_BLUE,
    C_GREEN,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    save_animation,
)

_N_HOLD_START = 20
_N_MORPH = 100
_N_HOLD_END = 30
_N_FRAMES = _N_HOLD_START + _N_MORPH + _N_HOLD_END  # 150
_FPS = 15


def _build_design_matrix() -> np.ndarray:
    x1 = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2 = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    # We don't need y for this animation — only the Hessian shape.
    _ = MEASURED_RSS_MB  # imported for namespace consistency / future use
    return np.column_stack([x1, x2])


def _kappa_at_t(x_raw: np.ndarray, n_max: float, m_max: float, t: float) -> float:
    """Condition number of H(t) = X(t)ᵀ X(t) where X(t) = X_raw / D(t)."""
    d = np.array([n_max**t, m_max**t])
    x_t = x_raw / d
    eigvals = np.linalg.eigvalsh(x_t.T @ x_t)
    return float(eigvals[-1] / max(eigvals[0], 1e-300))


def _smoothstep(x: float) -> float:
    """Cubic smoothstep easing on [0, 1]."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _t_at_frame(frame: int) -> float:
    if frame < _N_HOLD_START:
        return 0.0
    if frame >= _N_HOLD_START + _N_MORPH:
        return 1.0
    return _smoothstep((frame - _N_HOLD_START) / _N_MORPH)


def _format_kappa(kappa: float) -> str:
    if kappa < 1e3:
        return f"{kappa:.2f}"
    exp = int(np.floor(np.log10(kappa)))
    mant = kappa / 10**exp
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def _phase_label(t: float) -> str:
    if t < 0.001:
        return "raw columns  (no preconditioning)"
    if t > 0.999:
        return "fully Jacobi-normalised"
    return f"morphing — Jacobi level $t = {t:.2f}$"


def _setup_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("displacement along $v_{\\min}$  (long-axis direction)")
    ax.set_ylabel("displacement along $v_{\\max}$  (short-axis direction)")
    ax.plot(
        0,
        0,
        marker="*",
        color="white",
        markersize=18,
        markeredgecolor="black",
        markeredgewidth=0.9,
        zorder=10,
    )


def main() -> None:
    x_raw = _build_design_matrix()
    n_max = float(x_raw[:, 0].max())
    m_max = float(x_raw[:, 1].max())
    col_ratio_at_zero = m_max / n_max

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=100)
    fig.suptitle(
        "Jacobi normalisation morphs the OLS loss landscape\n"
        "Eigenvector-aligned view — visible aspect ratio $=\\sqrt{\\kappa(t)}$",
        fontsize=11,
        fontweight="bold",
    )

    n_grid = 280
    u_axis = np.linspace(-1.2, 1.2, n_grid)
    v_axis = np.linspace(-1.2, 1.2, n_grid)
    u_grid, v_grid = np.meshgrid(u_axis, v_axis)
    theta_circle = np.linspace(0, 2.0 * np.pi, 360)

    _setup_axes(ax)
    fig.tight_layout()

    artist_keys = [
        "contour_fill",
        "contour_lines",
        "ellipse_line",
        "ellipse_fill",
        "long_arrow",
        "short_arrow",
        "annot_block",
        "phase_label",
        "frame_tag",
        "warning_text",
    ]
    state: dict[str, object | None] = {k: None for k in artist_keys}

    def _clear_state() -> None:
        # ``QuadContourSet`` is an Artist in matplotlib ≥3.10 and exposes
        # ``.remove()`` directly; older ``.collections`` iteration was deprecated.
        for key in artist_keys:
            art = state[key]
            if art is None:
                continue
            art.remove()
            state[key] = None

    def update(frame: int) -> None:
        _clear_state()
        t = _t_at_frame(frame)
        kappa = _kappa_at_t(x_raw, n_max, m_max, t)

        quad = u_grid**2 + kappa * v_grid**2
        log_quad = np.log10(quad + 1.0)
        state["contour_fill"] = ax.contourf(u_grid, v_grid, log_quad, levels=18, cmap="viridis")
        state["contour_lines"] = ax.contour(
            u_grid,
            v_grid,
            log_quad,
            levels=18,
            colors="white",
            linewidths=0.4,
            alpha=0.30,
        )

        long_semi = 1.0
        short_semi = 1.0 / np.sqrt(kappa)
        ex = long_semi * np.cos(theta_circle)
        ey = short_semi * np.sin(theta_circle)
        state["ellipse_fill"] = ax.fill(ex, ey, color=C_RED, alpha=0.18, zorder=4)[0]
        state["ellipse_line"] = ax.plot(ex, ey, color=C_RED, lw=2.0, zorder=5)[0]

        state["long_arrow"] = ax.annotate(
            "",
            xy=(long_semi, 0),
            xytext=(-long_semi, 0),
            arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.5, alpha=0.85),
            zorder=6,
        )

        if short_semi > 0.04:
            state["short_arrow"] = ax.annotate(
                "",
                xy=(0, short_semi),
                xytext=(0, -short_semi),
                arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.5, alpha=0.85),
                zorder=6,
            )
        else:
            state["warning_text"] = ax.text(
                0.0,
                0.30,
                (f"short axis $\\approx {short_semi:.1e}$\n(below pixel resolution)"),
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

        col_ratio_t = col_ratio_at_zero ** (1.0 - t)
        annot_block = (
            f"$\\kappa = {_format_kappa(kappa)}$\n"
            f"$\\sqrt{{\\kappa}} \\approx {_format_kappa(np.sqrt(kappa))}$\n"
            f"col-max ratio $= {_format_kappa(col_ratio_t)}$"
        )
        state["annot_block"] = ax.text(
            0.04,
            0.96,
            annot_block,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.92,
                edgecolor="lightgray",
            ),
        )
        state["phase_label"] = ax.text(
            0.5,
            0.97,
            _phase_label(t),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="center",
            color="darkblue",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.92,
                edgecolor="lightgray",
            ),
            zorder=11,
        )

        # Per-frame tag prevents GIF optimiser from collapsing identical frames
        # (e.g. all hold frames at the same κ).  alpha=0.25 survives 256-colour
        # quantisation while staying visually unobtrusive.
        state["frame_tag"] = ax.text(
            0.999,
            0.001,
            str(frame),
            transform=ax.transAxes,
            fontsize=5,
            alpha=0.25,
            color="darkblue",
            ha="right",
            va="bottom",
        )

    anim = FuncAnimation(fig, update, frames=_N_FRAMES, interval=int(1000 / _FPS), repeat=False)
    path = save_animation(anim, "fig04_loss_landscape_animated", fps=_FPS, dpi=100)
    print(f"  saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
