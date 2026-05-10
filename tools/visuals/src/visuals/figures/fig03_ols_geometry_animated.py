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
fig03_ols_geometry_animated.py
§4 — Animated OLS Plane Build-Up

Data points appear one-by-one in sweep order, then the best-fit plane
materialises (alpha fade), then the view slowly rotates.

135 frames at 15 fps ≈ 9 s loop.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from visuals.common import (
    C_BLUE,
    C_GREEN,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    SIZE_3D,
    save_animation,
)

_N_SHAPES = len(PROBE_SHAPES)
_N_FRAMES = 135
_INIT_AZIM = 30.0
_INIT_ELEV = 25.0


def main() -> None:
    rng = np.random.default_rng(42)

    # Build design-matrix rows (same as static fig03)
    x1_data = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_data = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    y_data = np.array([MEASURED_RSS_MB[shape] * 1024**2 for shape in PROBE_SHAPES], dtype=float)
    y_noisy = y_data + rng.normal(0, 0.5e6, size=len(y_data))

    # Best-fit plane through origin
    design = np.column_stack([x1_data, x2_data])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_noisy, rcond=None)
    a_hat, b_hat = float(coeffs[0]), float(coeffs[1])

    # Grid for plane surface (coarser grid keeps GIF small)
    x1_lin = np.linspace(0, x1_data.max() * 1.1, 20)
    x2_lin = np.linspace(0, x2_data.max() * 1.1, 20)
    x1_grid, x2_grid = np.meshgrid(x1_lin, x2_lin)
    y_grid = a_hat * x1_grid + b_hat * x2_grid

    y_pred = a_hat * x1_data + b_hat * x2_data

    fig = plt.figure(figsize=SIZE_3D, dpi=100)

    def _draw_frame(frame: int) -> None:
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")

        # Determine state from frame index
        if frame < 21:
            # Frames 0-20: points appear one by one (3 frames per shape)
            n_show = frame // 3 + 1
            b_sz, s_sz = PROBE_SHAPES[frame // 3]
            title = f"Adding probe measurement {n_show}/{_N_SHAPES}: ({b_sz},{s_sz})"
            plane_alpha = 0.0
            show_residuals = False
        elif frame < 24:
            # Frames 21-23: all points, fitting hint
            n_show = _N_SHAPES
            title = "Fitting OLS plane…"
            plane_alpha = 0.0
            show_residuals = False
        elif frame < 45:
            # Frames 24-44: plane fades in over 7 steps (3 frames per step)
            n_show = _N_SHAPES
            fade_step = (frame - 24) // 3 + 1  # 1..7
            plane_alpha = (fade_step / 7) * 0.30
            show_residuals = True
            title = "Best-fit plane materialising…"
        else:
            # Frames 45-134: rotate azimuth 90°
            n_show = _N_SHAPES
            plane_alpha = 0.30
            show_residuals = True
            title = "$y = \\hat{a}\\cdot x_1 + \\hat{b}\\cdot x_2$  (no intercept)"

        # Origin marker (always present)
        ax.scatter([0], [0], [0], s=100, c="black", marker="*", zorder=6)

        # Scatter points visible so far
        for idx in range(n_show):
            ax.scatter(
                [x1_data[idx]],
                [x2_data[idx]],
                [y_noisy[idx] / 1e6],
                s=70,
                c=C_RED,
                zorder=5,
            )
            bv, sv = PROBE_SHAPES[idx]
            ax.text(
                x1_data[idx],
                x2_data[idx],
                y_noisy[idx] / 1e6 + 20,
                f"({bv},{sv})",
                fontsize=7,
                color="black",
                ha="center",
            )

        # Plane surface
        if plane_alpha > 0.0:
            ax.plot_surface(
                x1_grid,
                x2_grid,
                y_grid / 1e6,
                alpha=plane_alpha,
                color=C_BLUE,
                linewidth=0,
            )

        # Residual segments
        if show_residuals:
            for idx in range(_N_SHAPES):
                ax.plot(
                    [x1_data[idx], x1_data[idx]],
                    [x2_data[idx], x2_data[idx]],
                    [y_noisy[idx] / 1e6, y_pred[idx] / 1e6],
                    color=C_GREEN,
                    lw=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel("$x_1 = B\\cdot S$", labelpad=6)
        ax.set_ylabel("$x_2 = B\\cdot S^2$", labelpad=6)
        ax.set_zlabel("RSS (MiB)", labelpad=6)
        ax.set_title(title, fontsize=10)

        # Rotate during final phase
        if frame >= 45:
            rot = (frame - 45) * (90.0 / 89.0)
            ax.view_init(elev=_INIT_ELEV, azim=_INIT_AZIM + rot)
        else:
            ax.view_init(elev=_INIT_ELEV, azim=_INIT_AZIM)

        ax.text2D(
            0.02,
            0.02,
            f"a = {a_hat / 1024:.1f} KiB/tok  ·  b = {b_hat:.2f} B/tok²",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            color="navy",
        )

        # Per-frame counter prevents Pillow GIF optimizer from collapsing
        # identical consecutive hold frames (e.g. 3 frames per shape during
        # the appearance phase) into a single compressed frame.
        # alpha=0.25 darkblue on white survives 256-colour quantisation.
        ax.text2D(
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

    anim = FuncAnimation(fig, _draw_frame, frames=_N_FRAMES, interval=67)
    path = save_animation(anim, "fig03_ols_geometry_animated", fps=30, dpi=150)
    print(f"  saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
