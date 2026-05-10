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
fig03_ols_geometry.py
§4 — Ordinary Least Squares Without Intercept

3D scatter of the 7 probe shapes as (x1=B·S, x2=B·S², y=measured_rss).
Best-fit plane y = a·x1 + b·x2 (no intercept) rendered in semi-transparency.
Vertical residual segments from each point to the plane.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

from probe_visuals.common import (
    C_BLUE,
    C_GREEN,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    SIZE_3D,
    save,
)


def main() -> None:
    rng = np.random.default_rng(42)

    # Build design-matrix rows
    x1_data = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_data = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    # Convert measured MB → bytes (a/b coefficients are in bytes)
    y_data = np.array([MEASURED_RSS_MB[shape] * 1024**2 for shape in PROBE_SHAPES], dtype=float)

    # Add tiny noise so the fit matches "with noise" story
    y_noisy = y_data + rng.normal(0, 0.5e6, size=len(y_data))

    # ── Best-fit plane through origin ─────────────────────────────────────────
    # Solve min ||y - X @ [a,b]||²  s.t. no intercept
    X = np.column_stack([x1_data, x2_data])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_noisy, rcond=None)
    a_hat, b_hat = coeffs

    # ── Grid for plane surface ────────────────────────────────────────────────
    x1_lin = np.linspace(0, x1_data.max() * 1.1, 40)
    x2_lin = np.linspace(0, x2_data.max() * 1.1, 40)
    X1g, X2g = np.meshgrid(x1_lin, x2_lin)
    Yg = a_hat * X1g + b_hat * X2g

    fig = plt.figure(figsize=SIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    # Plane
    ax.plot_surface(X1g, X2g, Yg / 1e6, alpha=0.30, color=C_BLUE, linewidth=0)

    # Scatter points
    y_pred = a_hat * x1_data + b_hat * x2_data
    ax.scatter(
        x1_data,
        x2_data,
        y_noisy / 1e6,
        s=70,
        c=C_RED,
        zorder=5,
        label="Probe measurements",
    )

    # Residual segments
    for i in range(len(PROBE_SHAPES)):
        ax.plot(
            [x1_data[i], x1_data[i]],
            [x2_data[i], x2_data[i]],
            [y_noisy[i] / 1e6, y_pred[i] / 1e6],
            color=C_GREEN,
            lw=1.5,
            alpha=0.8,
        )

    # Origin marker (plane passes through origin)
    ax.scatter(
        [0],
        [0],
        [0],
        s=100,
        c="black",
        marker="*",
        zorder=6,
        label="Origin (no intercept)",
    )

    # Label each probe shape
    for i, (b, s) in enumerate(PROBE_SHAPES):
        label = f"({b},{s})"
        ax.text(
            x1_data[i],
            x2_data[i],
            y_noisy[i] / 1e6 + 20,
            label,
            fontsize=7,
            color="black",
            ha="center",
        )

    ax.set_xlabel("$x_1 = B \\cdot S$  (token-positions)", labelpad=8)
    ax.set_ylabel("$x_2 = B \\cdot S^2$  (token-positions²)", labelpad=8)
    ax.set_zlabel("RSS delta  (MiB)", labelpad=8)
    ax.set_title(
        "OLS Geometry: best-fit plane through origin\n"
        "$y = \\hat{a} \\cdot x_1 + \\hat{b} \\cdot x_2$  (no intercept)\n"
        "Green segments = residuals",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    ax.text2D(
        0.02,
        0.02,
        f"Fitted:  a = {a_hat/1024:.1f} KiB/tok  ·  b = {b_hat:.2f} B/tok²",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="navy",
    )

    fig.tight_layout()
    path = save(fig, "fig03_ols_geometry")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
