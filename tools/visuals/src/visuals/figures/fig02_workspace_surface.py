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
fig02_workspace_surface.py
§2/§3 — Geometric Intuition for the Budget

3D surface of W(B, S) = a·B·S + b·B·S² over B∈[1,16], S∈[64,8192].
Translucent horizontal plane at max_workspace ≈ 2 GiB.
Contour projection on the floor marks the "fits" region (W < budget).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

from probe_visuals.common import A_FIT, B_FIT, C_BLUE, C_RED, SIZE_3D, save


def main() -> None:
    B_arr = np.linspace(1, 16, 60)
    S_arr = np.linspace(64, 8192, 120)
    B_grid, S_grid = np.meshgrid(B_arr, S_arr)

    W_bytes = A_FIT * B_grid * S_grid + B_FIT * B_grid * S_grid**2
    W_GiB = W_bytes / (1024**3)

    MAX_WORKSPACE_GIB = 2.0  # ≈ 2 GiB budget plane

    fig = plt.figure(figsize=SIZE_3D)
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

    # Budget plane
    B_plane = np.array([[B_arr[0], B_arr[-1]], [B_arr[0], B_arr[-1]]])
    S_plane = np.array([[S_arr[0], S_arr[0]], [S_arr[-1], S_arr[-1]]])
    W_plane = np.full_like(B_plane, MAX_WORKSPACE_GIB)
    ax.plot_surface(B_plane, S_plane, W_plane, color=C_RED, alpha=0.25)

    # Contour projection on the floor (z=0) — "fits" contour
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

    # Labels
    ax.set_xlabel("Batch size  B", labelpad=8)
    ax.set_ylabel("Sequence length  S  (tokens)", labelpad=8)
    ax.set_zlabel("Workspace  W  (GiB)", labelpad=8)
    ax.set_title(
        "Workspace surface  $W(B,S) = a \\cdot B \\cdot S + b \\cdot B \\cdot S^2$\n"
        "Red plane = 2 GiB budget  ·  blue floor = fits region",
        fontsize=11,
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

    fig.tight_layout()
    path = save(fig, "fig02_workspace_surface")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
