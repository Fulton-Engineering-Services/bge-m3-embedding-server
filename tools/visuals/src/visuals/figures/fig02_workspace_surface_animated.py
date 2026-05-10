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
fig02_workspace_surface_animated.py
§2/§3 — Animated 360° rotation of the W(B,S) workspace surface.

The surface, budget plane, floor contour, colorbar, and labels are drawn
ONCE before the animation loop.  Only the view angle updates per frame.

135 frames at 15 fps ≈ 9 s loop.
  Frames   0–14  : static hold at initial view; title fades in.
  Frames  15–134 : 360° azimuth rotation (3° per frame).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

from visuals.common import A_FIT, B_FIT, C_BLUE, C_RED, SIZE_3D, save_animation

_N_FRAMES = 135
_INIT_AZIM = 45.0
_INIT_ELEV = 25.0


def main() -> None:
    B_arr = np.linspace(1, 16, 30)
    S_arr = np.linspace(64, 8192, 60)
    B_grid, S_grid = np.meshgrid(B_arr, S_arr)

    W_bytes = A_FIT * B_grid * S_grid + B_FIT * B_grid * S_grid**2
    W_GiB = W_bytes / (1024**3)

    MAX_WORKSPACE_GIB = 2.0

    fig = plt.figure(figsize=SIZE_3D, dpi=60)
    ax = fig.add_subplot(111, projection="3d")

    # Clip surface to 4× budget for visual clarity
    W_plot = np.clip(W_GiB, 0, 4 * MAX_WORKSPACE_GIB)

    surf = ax.plot_surface(
        B_grid,
        S_grid,
        W_plot,
        cmap=cm.viridis,
        alpha=0.75,
        linewidth=0,
        antialiased=True,
    )

    # Budget plane (translucent red horizontal slice at 2 GiB)
    B_plane = np.array([[B_arr[0], B_arr[-1]], [B_arr[0], B_arr[-1]]])
    S_plane = np.array([[S_arr[0], S_arr[0]], [S_arr[-1], S_arr[-1]]])
    W_plane = np.full_like(B_plane, MAX_WORKSPACE_GIB)
    ax.plot_surface(B_plane, S_plane, W_plane, color=C_RED, alpha=0.25)

    # Contour projection on the floor (z=0) — "fits" region
    floor_z = 0
    contour_levels = [MAX_WORKSPACE_GIB]
    ax.contourf(
        B_grid,
        S_grid,
        W_GiB,
        levels=[0, MAX_WORKSPACE_GIB],
        zdir="z",
        offset=floor_z,
        colors=[C_BLUE],
        alpha=0.3,
    )
    ax.contour(
        B_grid,
        S_grid,
        W_GiB,
        levels=contour_levels,
        zdir="z",
        offset=floor_z,
        colors=[C_RED],
        linewidths=2,
    )

    # Axis labels
    ax.set_xlabel("Batch size  B", labelpad=8)
    ax.set_ylabel("Sequence length  S  (tokens)", labelpad=8)
    ax.set_zlabel("Workspace  W  (GiB)", labelpad=8)

    # Title starts invisible; fades in over frames 0–14
    title_text = ax.set_title(
        "Workspace surface  $W(B,S) = a \\cdot B \\cdot S + b \\cdot B \\cdot S^2$\n"
        "Red plane = 2 GiB budget  ·  blue floor = fits region",
        fontsize=11,
        alpha=0.0,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("W (GiB, capped at 8)")

    ax.text2D(
        0.02,
        0.97,
        "Blue shaded floor: CostModel::fits() returns true",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="navy",
    )

    # Per-frame counter prevents Pillow GIF optimizer from collapsing
    # visually identical hold frames into a single compressed frame.
    # alpha=0.25 darkblue on white survives 256-colour quantisation.
    frame_counter = ax.text2D(
        0.999,
        0.001,
        "0",
        transform=ax.transAxes,
        fontsize=5,
        alpha=0.25,
        color="darkblue",
        ha="right",
        va="bottom",
    )

    # Set initial view angle before animation starts
    ax.view_init(elev=_INIT_ELEV, azim=_INIT_AZIM)

    def update(frame: int) -> None:
        if frame < 15:
            # Hold phase: fade title in from transparent to opaque
            title_text.set_alpha((frame + 1) / 15)
        else:
            title_text.set_alpha(1.0)
            # Rotation phase: 3° per frame → 120 frames × 3° = 360°
            azim = _INIT_AZIM + (frame - 15) * 3.0
            ax.view_init(elev=_INIT_ELEV, azim=azim)
        frame_counter.set_text(str(frame))

    anim = FuncAnimation(fig, update, frames=_N_FRAMES, interval=67, blit=False)
    path = save_animation(anim, "fig02_workspace_surface_animated", fps=15, dpi=60)
    print(f"  saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
