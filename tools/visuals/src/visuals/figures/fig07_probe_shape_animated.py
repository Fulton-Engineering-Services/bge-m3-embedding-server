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
fig07_probe_shape_animated.py
§7 — Animated Information Accumulation

Probe shapes appear one-by-one in sweep order.  After each shape appears the
normalised Gram-matrix determinant counter updates.  Shows how design coverage
grows and why the off-arc (4,64) point matters.

168 frames at 15 fps ≈ 11.2 s loop.
  Frames   0– 23: shape 1 visible
  Frames  24– 47: shapes 1-2 visible
  ...
  Frames 144–167: all 7 shapes visible (stability annotation)
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from probe_visuals.common import (
    C_BLUE,
    C_GREEN,
    C_GREY,
    C_RED,
    PROBE_SHAPES,
    SIZE_2D,
    save_animation,
)

_N_SHAPES = len(PROBE_SHAPES)
_FRAMES_PER_SHAPE = 24
_N_FRAMES = _N_SHAPES * _FRAMES_PER_SHAPE  # 168


def _compute_gram_dets(
    xi1: np.ndarray,
    xi2: np.ndarray,
) -> list[float]:
    """Return normalised Gram-matrix determinant after adding each shape in order."""
    raw_dets = []
    for k in range(1, _N_SHAPES + 1):
        xi1_k = xi1[:k]
        xi2_k = xi2[:k]
        g11 = float(np.dot(xi1_k, xi1_k))
        g22 = float(np.dot(xi2_k, xi2_k))
        g12 = float(np.dot(xi1_k, xi2_k))
        raw_dets.append(g11 * g22 - g12 * g12)

    max_det = max(abs(d) for d in raw_dets)
    if max_det == 0.0:
        return [0.0] * _N_SHAPES
    return [d / max_det for d in raw_dets]


def main() -> None:
    x1_vals = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_vals = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    labels = [f"({b},{s})" for b, s in PROBE_SHAPES]

    n_max = float(x1_vals.max())
    m_max = float(x2_vals.max())
    xi1 = x1_vals / n_max
    xi2 = x2_vals / m_max

    det_values = _compute_gram_dets(xi1, xi2)

    off_arc_idx = labels.index("(4,64)")

    # B=1 arc for background
    s_arc = np.logspace(np.log10(64), np.log10(8192), 300)
    x1_arc = s_arc
    x2_arc = s_arc**2

    fig, ax = plt.subplots(figsize=SIZE_2D, dpi=100)

    def _setup_axes() -> None:
        ax.cla()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$x_1 = B\\cdot S$  (token-positions, log scale)")
        ax.set_ylabel("$x_2 = B\\cdot S^2$  (token-positions², log scale)")
        # Draw B=1 arc (always visible)
        ax.plot(
            x1_arc,
            x2_arc,
            color=C_GREY,
            lw=1.2,
            ls="--",
            label="$B=1$ arc: $x_2 = x_1^2$",
            zorder=1,
        )

    def update(frame: int) -> None:
        _setup_axes()

        # How many shapes visible this frame
        num_shown = min(frame // _FRAMES_PER_SHAPE + 1, _N_SHAPES)
        show_stability = num_shown == _N_SHAPES  # frames 144-167

        ax.set_title(
            f"Probe Shape Information Geometry — shape {num_shown}/{_N_SHAPES} added\n"
            "red = off-arc point $(4, 64)$ that isolates $b$",
            fontsize=10,
        )

        # Scatter points added so far
        for i in range(num_shown):
            color = C_RED if i == off_arc_idx else C_BLUE
            ax.scatter([x1_vals[i]], [x2_vals[i]], c=color, s=100, zorder=5)
            ax.annotate(
                labels[i],
                (x1_vals[i], x2_vals[i]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color="darkred" if i == off_arc_idx else "black",
            )

        # Gram-det counter (top-left)
        det_val = det_values[num_shown - 1]
        ax.text(
            0.03,
            0.97,
            f"Shapes seen: {num_shown}/{_N_SHAPES}\nnorm. det(G) = {det_val:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.90},
        )

        # Arrow showing same-x1 information (visible once both (1,64) and (4,64) shown)
        if num_shown >= 3:
            # (4,64) and (1,256) both visible from shape index 2 onward
            idx_464 = 1  # PROBE_SHAPES index
            idx_1256 = 2
            x1_pair = float(x1_vals[idx_464])
            x2_464 = float(x2_vals[idx_464])
            x2_1256 = float(x2_vals[idx_1256])
            ax.annotate(
                "",
                xy=(x1_pair * 1.03, x2_464),
                xytext=(x1_pair * 1.03, x2_1256),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": C_GREEN,
                    "lw": 2.0,
                    "connectionstyle": "arc3,rad=0.0",
                },
            )
            ax.text(
                x1_pair * 1.3,
                float(np.sqrt(x2_464 * x2_1256)),
                "Same $x_1$,\ndiff. $x_2$\n→ isolates $b$",
                fontsize=8,
                color="darkgreen",
                va="center",
            )

        # Stability annotation (hold frames)
        if show_stability:
            ax.text(
                0.03,
                0.03,
                "Fit is numerically stable [OK]",
                transform=ax.transAxes,
                fontsize=10,
                color="darkgreen",
                fontweight="bold",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "honeydew", "alpha": 0.92},
            )

        # Legend
        b1_patch = mpatches.Patch(color=C_BLUE, label="$B = 1$ shapes (on arc)")
        off_patch = mpatches.Patch(color=C_RED, label="$(4, 64)$  off-arc")
        ax.legend(handles=[b1_patch, off_patch], fontsize=9, loc="lower right")

        # Subtle per-frame counter in axes corner prevents Pillow GIF optimizer from
        # collapsing identical consecutive hold frames into a single compressed frame.
        # alpha=0.25 darkblue on white background → pale lavender, survives 256-colour
        # palette quantisation unlike near-invisible alpha≈0.02 text.
        ax.text(
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

        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=_N_FRAMES, interval=67, repeat=False)
    path = save_animation(anim, "fig07_probe_shape_animated", fps=30, dpi=150)
    print(f"  saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
