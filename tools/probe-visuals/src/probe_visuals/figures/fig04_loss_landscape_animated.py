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
§5 — Animated Gradient Descent Trajectory

Builds up a 20-step gradient-descent trajectory on both panels simultaneously.
Left (raw, ill-conditioned): zig-zag.
Right (normalised): smoother convergence.

135 frames at 15 fps ≈ 9 s loop."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.animation import FuncAnimation

from probe_visuals.common import (
    A_FIT,
    B_FIT,
    C_BLUE,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    SIZE_2D,
    save_animation,
)

_N_TRAJ_STEPS = 20
_N_FRAMES = 135


def _run_gd_raw(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    start_a: float,
    start_b: float,
    lr: float,
    a_lo: float,
    a_hi: float,
    b_lo: float,
    b_hi: float,
) -> list[tuple[float, float]]:
    """Run gradient descent in raw parameter space; clip to grid bounds."""
    positions = [(start_a, start_b)]
    a, b = start_a, start_b
    for _ in range(_N_TRAJ_STEPS):
        residuals = y - a * x1 - b * x2
        da = float(-2.0 * (residuals * x1).sum())
        db = float(-2.0 * (residuals * x2).sum())
        a = float(np.clip(a - lr * da, a_lo, a_hi))
        b = float(np.clip(b - lr * db, b_lo, b_hi))
        positions.append((a, b))
    return positions


def _run_gd_norm(
    xi1: np.ndarray,
    xi2: np.ndarray,
    y: np.ndarray,
    start_alpha: float,
    start_beta: float,
    lr: float,
    al_lo: float,
    al_hi: float,
    be_lo: float,
    be_hi: float,
) -> list[tuple[float, float]]:
    """Run gradient descent in normalised parameter space; clip to grid bounds."""
    positions = [(start_alpha, start_beta)]
    alpha, beta = start_alpha, start_beta
    for _ in range(_N_TRAJ_STEPS):
        residuals = y - alpha * xi1 - beta * xi2
        da = float(-2.0 * (residuals * xi1).sum())
        db = float(-2.0 * (residuals * xi2).sum())
        alpha = float(np.clip(alpha - lr * da, al_lo, al_hi))
        beta = float(np.clip(beta - lr * db, be_lo, be_hi))
        positions.append((alpha, beta))
    return positions


def main() -> None:
    # Build design matrix (same as static fig04)
    x1_data = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_data = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    y_data = np.array([MEASURED_RSS_MB[shape] * 1024**2 for shape in PROBE_SHAPES], dtype=float)

    n_max = float(x1_data.max())
    m_max = float(x2_data.max())
    xi1 = x1_data / n_max
    xi2 = x2_data / m_max

    a_opt = A_FIT
    b_opt = B_FIT
    alpha_opt = A_FIT * n_max
    beta_opt = B_FIT * m_max

    # Grid extents for contour plots
    a_range = np.linspace(a_opt * 0.2, a_opt * 1.8, 300)
    b_range = np.linspace(b_opt * 0.2, b_opt * 1.8, 300)
    a_grid, b_grid = np.meshgrid(a_range, b_range)

    alpha_range = np.linspace(alpha_opt * 0.2, alpha_opt * 1.8, 300)
    beta_range = np.linspace(beta_opt * 0.2, beta_opt * 1.8, 300)
    al_grid, be_grid = np.meshgrid(alpha_range, beta_range)

    # Loss functions
    def loss_raw(a_g: np.ndarray, b_g: np.ndarray) -> np.ndarray:
        residuals = (
            y_data[:, None, None]
            - a_g[None, :, :] * x1_data[:, None, None]
            - b_g[None, :, :] * x2_data[:, None, None]
        )
        return (residuals**2).sum(axis=0)

    def loss_norm(alpha_g: np.ndarray, beta_g: np.ndarray) -> np.ndarray:
        residuals = (
            y_data[:, None, None]
            - alpha_g[None, :, :] * xi1[:, None, None]
            - beta_g[None, :, :] * xi2[:, None, None]
        )
        return (residuals**2).sum(axis=0)

    l_raw = np.log10(loss_raw(a_grid, b_grid) + 1)
    l_norm = np.log10(loss_norm(al_grid, be_grid) + 1)

    # Starting point (same as static fig)
    start_a_raw = a_opt * 1.5
    start_b_raw = b_opt * 0.3
    start_alpha = alpha_opt * 1.5
    start_beta = beta_opt * 0.3

    # Learning rates:
    # Raw: lr slightly above convergence threshold for b direction → zig-zag visible
    # Norm: lr well within convergence range → smoother trajectory
    lr_raw = 1.5e-16
    lr_norm = 0.25

    pos_raw = _run_gd_raw(
        x1_data,
        x2_data,
        y_data,
        start_a_raw,
        start_b_raw,
        lr_raw,
        float(a_range[0]),
        float(a_range[-1]),
        float(b_range[0]),
        float(b_range[-1]),
    )
    pos_norm = _run_gd_norm(
        xi1,
        xi2,
        y_data,
        start_alpha,
        start_beta,
        lr_norm,
        float(alpha_range[0]),
        float(alpha_range[-1]),
        float(beta_range[0]),
        float(beta_range[-1]),
    )

    # Colour map: blue (early) → red (late)
    cmap = plt.cm.coolwarm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE_2D, dpi=100)
    fig.suptitle(
        "OLS Loss Landscape — Gradient Descent Trajectory\n"
        "Raw (ill-conditioned) vs. Jacobi-normalised",
        fontsize=11,
        fontweight="bold",
    )

    log_levels = 25

    # Draw static contour backgrounds once
    ax1.contourf(a_grid, b_grid, l_raw, levels=log_levels, cmap="RdYlGn_r")
    ax1.contour(a_grid, b_grid, l_raw, levels=log_levels, colors="k", linewidths=0.3, alpha=0.4)
    ax1.plot(a_opt, b_opt, "w*", ms=14, zorder=5, label="Optimum")
    ax1.set_xlabel("$a$  (bytes / token-position)")
    ax1.set_ylabel("$b$  (bytes / token-position²)")
    ax1.set_title("Raw columns\n(ill-conditioned)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

    ax2.contourf(al_grid, be_grid, l_norm, levels=log_levels, cmap="RdYlGn_r")
    ax2.contour(al_grid, be_grid, l_norm, levels=log_levels, colors="k", linewidths=0.3, alpha=0.4)
    ax2.plot(alpha_opt, beta_opt, "w*", ms=14, zorder=5, label="Optimum")
    ax2.set_xlabel(r"$\alpha = a \cdot n_{\max}$")
    ax2.set_ylabel(r"$\beta = b \cdot m_{\max}$")
    ax2.set_title("Jacobi-normalised\n(better conditioned)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))

    fig.tight_layout()

    # Trajectory artist lists (managed per-frame)
    traj_raw: list = []
    traj_norm: list = []
    annot_raw: list = []
    annot_norm: list = []
    step_text: list = []

    def _clear_artists() -> None:
        for art_list in (traj_raw, traj_norm, annot_raw, annot_norm, step_text):
            for art in art_list:
                art.remove()
            art_list.clear()

    def update(frame: int) -> None:
        _clear_artists()

        step = min(frame // 3, _N_TRAJ_STEPS)

        # Draw trajectory segments up to current step
        for i in range(1, step + 1):
            t = (i - 1) / max(_N_TRAJ_STEPS - 1, 1)
            color = cmap(t)

            line_r = ax1.plot(
                [pos_raw[i - 1][0], pos_raw[i][0]],
                [pos_raw[i - 1][1], pos_raw[i][1]],
                color=color,
                lw=2.0,
                alpha=0.85,
                zorder=4,
            )[0]
            traj_raw.append(line_r)

            line_n = ax2.plot(
                [pos_norm[i - 1][0], pos_norm[i][0]],
                [pos_norm[i - 1][1], pos_norm[i][1]],
                color=color,
                lw=2.0,
                alpha=0.85,
                zorder=4,
            )[0]
            traj_norm.append(line_n)

        # Current-position dot
        pt_r = ax1.scatter(pos_raw[step][0], pos_raw[step][1], s=80, c=C_RED, zorder=8)
        traj_raw.append(pt_r)

        pt_n = ax2.scatter(pos_norm[step][0], pos_norm[step][1], s=80, c=C_RED, zorder=8)
        traj_norm.append(pt_n)

        # Step counter
        txt = fig.text(
            0.5,
            0.01,
            f"Step {step} / {_N_TRAJ_STEPS}",
            ha="center",
            fontsize=10,
            color="darkblue",
        )
        step_text.append(txt)

        # Hold-phase annotations (frames 63-134)
        if frame >= 63:
            alpha_fade = min(1.0, (frame - 62) / 15.0)

            ann_r = ax1.text(
                0.03,
                0.06,
                "Still oscillating",
                transform=ax1.transAxes,
                fontsize=9,
                color="firebrick",
                fontweight="bold",
                va="bottom",
                alpha=alpha_fade,
            )
            annot_raw.append(ann_r)

            ann_n = ax2.text(
                0.03,
                0.06,
                "Converging",
                transform=ax2.transAxes,
                fontsize=9,
                color="darkgreen",
                fontweight="bold",
                va="bottom",
                alpha=alpha_fade,
            )
            annot_norm.append(ann_n)

        # Start marker (always visible; re-add so it's above contour)
        st_r = ax1.scatter(pos_raw[0][0], pos_raw[0][1], s=80, c=C_BLUE, zorder=7, marker="o")
        traj_raw.append(st_r)

        st_n = ax2.scatter(pos_norm[0][0], pos_norm[0][1], s=80, c=C_BLUE, zorder=7, marker="o")
        traj_norm.append(st_n)

        # Subtle per-frame counter prevents Pillow GIF optimizer from collapsing
        # identical consecutive hold frames (e.g. frames 25-29 at step=20, alpha=1.0)
        # into a single compressed frame.  alpha=0.25 survives 256-colour quantisation.
        frame_lbl = ax1.text(
            0.999,
            0.001,
            str(frame),
            transform=ax1.transAxes,
            fontsize=5,
            alpha=0.25,
            color="darkblue",
            ha="right",
            va="bottom",
        )
        traj_raw.append(frame_lbl)

    anim = FuncAnimation(fig, update, frames=_N_FRAMES, interval=67, repeat=False)
    path = save_animation(anim, "fig04_loss_landscape_animated", fps=15, dpi=100)
    print(f"  saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
