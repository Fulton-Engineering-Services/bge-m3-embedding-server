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
fig05_column_magnitudes.py
§5 — Why Scale Matters

Two-panel bar chart over the 7 probe shapes.
Panel A (log-y): B·S vs B·S² side-by-side — shows ~8000× ratio at (1,8192).
Panel B (linear): same after Jacobi normalisation → both columns in [0,1].
"""

import matplotlib.pyplot as plt
import numpy as np

from visuals.common import C_BLUE, C_RED, PROBE_SHAPES, SIZE_2D, save


def main() -> None:
    labels = [f"({b},{s})" for b, s in PROBE_SHAPES]
    x1_vals = np.array([b * s for b, s in PROBE_SHAPES], dtype=float)
    x2_vals = np.array([b * s**2 for b, s in PROBE_SHAPES], dtype=float)

    n_max = x1_vals.max()
    m_max = x2_vals.max()

    xi1 = x1_vals / n_max
    xi2 = x2_vals / m_max

    x = np.arange(len(labels))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE_2D)
    fig.suptitle(
        "Column Magnitudes Before and After Jacobi Scaling",
        fontsize=13,
        fontweight="bold",
    )

    # ── Panel A: raw, log-y ──────────────────────────────────────────────────
    ax1.bar(x - w / 2, x1_vals, width=w, color=C_BLUE, label="$x_1 = B \\cdot S$")
    ax1.bar(x + w / 2, x2_vals, width=w, color=C_RED, label="$x_2 = B \\cdot S^2$")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_xlabel("Probe shape  (B, S)")
    ax1.set_ylabel("Column value  (log scale)")
    ax1.set_title("Raw columns\n(up to ~8000× ratio)")
    ax1.legend()

    # Annotate the maximum-ratio pair (1, 8192)
    idx_max = labels.index("(1,8192)")
    ratio = x2_vals[idx_max] / x1_vals[idx_max]
    ax1.annotate(
        f"ratio ≈ {ratio:,.0f}×",
        xy=(x[idx_max] + w / 2, x2_vals[idx_max]),
        xytext=(x[idx_max] - 1.2, x2_vals[idx_max] * 0.3),
        fontsize=8,
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )

    # ── Panel B: normalised, linear ──────────────────────────────────────────
    ax2.bar(x - w / 2, xi1, width=w, color=C_BLUE, label=r"$\xi_1 = x_1 / n_{\max}$")
    ax2.bar(x + w / 2, xi2, width=w, color=C_RED, label=r"$\xi_2 = x_2 / m_{\max}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_xlabel("Probe shape  (B, S)")
    ax2.set_ylabel("Normalised column value  (0 – 1)")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Jacobi-normalised columns\n(both axes in [0, 1])")
    ax2.legend()
    ax2.axhline(1.0, color="grey", lw=0.8, ls="--")

    # Big arrow between panels — text annotation
    ax1.text(
        1.08,
        0.5,
        "÷ $D$\n→ 1×",
        transform=ax1.transAxes,
        fontsize=11,
        ha="center",
        va="center",
        color="navy",
        fontweight="bold",
    )

    fig.tight_layout()
    path = save(fig, "fig05_column_magnitudes")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
