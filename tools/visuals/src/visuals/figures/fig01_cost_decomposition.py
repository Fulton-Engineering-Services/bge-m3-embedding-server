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
fig01_cost_decomposition.py
§2 — Where the Quadratic Term Comes From

Two-panel chart at B=1:
  Panel A (linear axes): linear term a·S, quadratic term b·S², and total W(S).
                         Vertical dashed line marks the crossover S = a/b.
  Panel B (log-log axes): same data — straight lines of slope 1 and 2 confirm
                          the two regimes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from visuals.common import A_FIT, B_FIT, C_BLUE, C_GREEN, C_GREY, C_RED, SIZE_2D, save


def main() -> None:
    S = np.linspace(1, 8192, 4000)
    linear = A_FIT * S
    quadratic = B_FIT * S**2
    total = linear + quadratic

    crossover_S = A_FIT / B_FIT  # ≈ 2973

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE_2D)
    fig.suptitle(
        "BGE-M3 Workspace Cost Decomposition  (B = 1)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # ── Panel A: linear axes ─────────────────────────────────────────────────
    ax1.plot(S, linear / 1e6, color=C_BLUE, lw=2, label=r"$a \cdot S$ (linear)")
    ax1.plot(S, quadratic / 1e6, color=C_RED, lw=2, label=r"$b \cdot S^2$ (quadratic)")
    ax1.plot(S, total / 1e6, color=C_GREEN, lw=2.5, ls="--", label="Total  $W(S)$")

    ax1.axvline(crossover_S, color=C_GREY, lw=1.4, ls=":", zorder=0)
    ax1.annotate(
        f"crossover\n$S^* = a/b \\approx {crossover_S:,.0f}$",
        xy=(crossover_S, (A_FIT * crossover_S + B_FIT * crossover_S**2) / 2e6),
        xytext=(crossover_S + 600, 300),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="grey"),
        color="grey",
    )

    # regime labels
    ax1.text(1100, 70, "projection-\ndominated", fontsize=9, color=C_BLUE, ha="center")
    ax1.text(
        6200,
        (B_FIT * 6200**2) / 1e6 * 0.60,
        "attention-\ndominated",
        fontsize=9,
        color=C_RED,
        ha="center",
    )

    ax1.set_xlabel("Sequence length  S  (tokens)")
    ax1.set_ylabel("Workspace  (MiB)")
    ax1.set_title("Linear axes")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_xlim(0, 8192)
    ax1.set_ylim(0)

    # ── Panel B: log-log axes ─────────────────────────────────────────────────
    ax2.loglog(S, linear, color=C_BLUE, lw=2, label=r"$a \cdot S$  (slope 1)")
    ax2.loglog(S, quadratic, color=C_RED, lw=2, label=r"$b \cdot S^2$  (slope 2)")
    ax2.loglog(S, total, color=C_GREEN, lw=2.5, ls="--", label="Total")
    ax2.axvline(crossover_S, color=C_GREY, lw=1.4, ls=":", zorder=0)

    ax2.set_xlabel("Sequence length  S  (tokens, log scale)")
    ax2.set_ylabel("Workspace  (bytes, log scale)")
    ax2.set_title("Log–log axes  (slopes 1 and 2 visible)")
    ax2.legend(loc="upper left")

    # Annotate slopes
    ax2.text(100, A_FIT * 100 * 3, "slope 1", fontsize=9, color=C_BLUE, rotation=22)
    ax2.text(200, B_FIT * 200**2 * 0.25, "slope 2", fontsize=9, color=C_RED, rotation=40)

    fig.tight_layout()
    path = save(fig, "fig01_cost_decomposition")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
