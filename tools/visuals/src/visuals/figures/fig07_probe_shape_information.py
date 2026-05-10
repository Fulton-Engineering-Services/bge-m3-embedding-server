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
fig07_probe_shape_information.py
§7 — Information Geometry for Two Coefficients

Log-log scatter of the 7 probe shapes in (B·S, B·S²) space.
(4,64) highlighted in contrasting color.
Arrow from (1,256) to (4,64): same x1 = B·S = 256, but very different x2 — isolates b.
Annotations explain the L-shape geometry.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from visuals.common import C_BLUE, C_GREEN, C_GREY, C_RED, PROBE_SHAPES, SIZE_2D, save


def main() -> None:
    x1_vals = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_vals = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    labels = [f"({b},{s})" for b, s in PROBE_SHAPES]

    off_arc_idx = labels.index("(4,64)")
    ref_idx = labels.index("(1,256)")  # same x1 as (4,64)

    colors = [C_RED if i == off_arc_idx else C_BLUE for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=SIZE_2D)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Draw the B=1 arc (x2 = x1²)
    S_arc = np.logspace(np.log10(64), np.log10(8192), 300)
    ax.plot(
        S_arc,
        S_arc**2,
        color=C_GREY,
        lw=1.2,
        ls="--",
        label="$B=1$ arc: $x_2 = x_1^2$",
        zorder=1,
    )

    # Scatter
    ax.scatter(x1_vals, x2_vals, c=colors, s=100, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl,
            (x1_vals[i], x2_vals[i]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color="darkred" if i == off_arc_idx else "black",
        )

    # ── Horizontal bracket: same x1=256 → different x2 ───────────────────────
    x1_pair = x1_vals[off_arc_idx]  # = 256
    x2_off = x2_vals[off_arc_idx]  # (4,64)  → 4*64² = 16384
    x2_ref = x2_vals[ref_idx]  # (1,256) → 256² = 65536

    # Arrow from (1,256) down to (4,64) at fixed x1=256
    ax.annotate(
        "",
        xy=(x1_pair * 1.02, x2_off),
        xytext=(x1_pair * 1.02, x2_ref),
        arrowprops=dict(
            arrowstyle="<->",
            color=C_GREEN,
            lw=2.0,
            connectionstyle="arc3,rad=0.0",
        ),
    )
    ax.text(
        x1_pair * 1.25,
        np.sqrt(x2_off * x2_ref),
        "Same $x_1$,\ndifferent $x_2$\n→ near-pure\nmeasurement of $b$",
        fontsize=8,
        color="darkgreen",
        va="center",
    )

    ax.text(
        0.03,
        0.97,
        "$B=1$ arc gives joint $(a,b)$ leverage.\nOff-arc point $(4,64)$ isolates $b$.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
    )

    ax.set_xlabel("$x_1 = B \\cdot S$  (token-positions, log scale)")
    ax.set_ylabel("$x_2 = B \\cdot S^2$  (token-positions², log scale)")
    ax.set_title(
        "Probe Shape Information Geometry\n"
        "7 shapes in $(B \\cdot S,\\; B \\cdot S^2)$ space  ·  "
        "red = off-arc point $(4, 64)$",
        fontsize=11,
    )

    # Legend
    b1_patch = mpatches.Patch(color=C_BLUE, label="$B = 1$ shapes (on arc)")
    off_patch = mpatches.Patch(color=C_RED, label="$(4, 64)$  off-arc — isolates $b$")
    ax.legend(handles=[b1_patch, off_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    path = save(fig, "fig07_probe_shape_information")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
