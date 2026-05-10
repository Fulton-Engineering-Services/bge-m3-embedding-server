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
fig06_jacobi_transformation.py
§6 — Column Normalisation as a Coordinate Change

Two-panel scatter.
Panel A: probe shapes in raw (x1, x2) space (log-log axes).
Panel B: same shapes in normalised (ξ1, ξ2) ∈ [0,1]² space (linear axes).
The (4,64) off-arc point is highlighted.
The scaling matrix D = diag(n_max, m_max) is shown as a text box.
"""

import matplotlib.pyplot as plt
import numpy as np

from probe_visuals.common import C_BLUE, C_GREY, C_RED, PROBE_SHAPES, SIZE_2D, save


def main() -> None:
    x1_vals = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_vals = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)
    n_max = x1_vals.max()
    m_max = x2_vals.max()
    xi1 = x1_vals / n_max
    xi2 = x2_vals / m_max

    labels = [f"({b},{s})" for b, s in PROBE_SHAPES]
    off_arc_idx = labels.index("(4,64)")  # the point that breaks the B=1 arc

    colors = [C_RED if i == off_arc_idx else C_BLUE for i in range(len(labels))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE_2D)
    fig.suptitle(
        "Jacobi Scaling as a Coordinate Change\n"
        r"$D = \mathrm{diag}(n_{\max},\; m_{\max})$  maps raw features to $[0,1]^2$",
        fontsize=12,
        fontweight="bold",
    )

    # ── Panel A: raw log-log ──────────────────────────────────────────────────
    ax1.scatter(x1_vals, x2_vals, c=colors, s=80, zorder=5)
    for i, lbl in enumerate(labels):
        ax1.annotate(
            lbl,
            (x1_vals[i], x2_vals[i]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
            color="darkred" if i == off_arc_idx else "black",
        )

    # Draw the B=1 arc as a guide (x2 = x1²  when B=1 and S = x1)
    S_arc = np.logspace(np.log10(64), np.log10(8192), 200)
    ax1.plot(S_arc, S_arc**2, color=C_GREY, lw=1.2, ls="--", label="$B=1$ arc: $x_2=x_1^2$")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("$x_1 = B \\cdot S$  (token-positions, log scale)")
    ax1.set_ylabel("$x_2 = B \\cdot S^2$  (token-positions², log scale)")
    ax1.set_title("Raw feature space\n(log–log  —  hard to see structure)")
    ax1.legend(fontsize=9)
    ax1.annotate(
        "(4,64) off arc\n→ isolates b",
        xy=(x1_vals[off_arc_idx], x2_vals[off_arc_idx]),
        xytext=(x1_vals[off_arc_idx] * 0.25, x2_vals[off_arc_idx] * 10),
        fontsize=8,
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )

    # ── Panel B: normalised linear ────────────────────────────────────────────
    ax2.scatter(xi1, xi2, c=colors, s=80, zorder=5)
    for i, lbl in enumerate(labels):
        ax2.annotate(
            lbl,
            (xi1[i], xi2[i]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
            color="darkred" if i == off_arc_idx else "black",
        )

    # B=1 normalised arc
    xi1_arc = S_arc / n_max
    xi2_arc = (S_arc**2) / m_max
    ax2.plot(xi1_arc, xi2_arc, color=C_GREY, lw=1.2, ls="--", label="$B=1$ arc")

    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlabel(r"$\xi_1 = x_1 / n_{\max}$")
    ax2.set_ylabel(r"$\xi_2 = x_2 / m_{\max}$")
    ax2.set_title(r"Normalised $[0,1]^2$ space" + "\n(linear  —  structure visible)")
    ax2.legend(fontsize=9)
    ax2.set_aspect("equal")

    # Transformation box
    ax1.text(
        0.02,
        0.97,
        r"$D = \mathrm{diag}(n_{\max},\;m_{\max})$" + "\n" + r"$\xi = D^{-1} x$",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Arrow between panels (use plain text — \xrightarrow is not in mathtext)
    fig.text(
        0.505,
        0.5,
        r"$D^{-1}$" + "\n" + r"$\longrightarrow$",
        ha="center",
        va="center",
        fontsize=13,
        color="navy",
    )

    fig.tight_layout()
    path = save(fig, "fig06_jacobi_transformation")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
