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
fig09_fit_quality.py
§8/§13 — From Measurement to Cost Model

Measured RSS deltas at B=1 plotted as scatter points.
Overlay: fitted curve  a_fit·S + b_fit·S²
         dashed curve  a_cons·S + b_cons·S²  (conservative defaults)

Shows: fitted model tracks data; conservative defaults are systematically
       too pessimistic at low S and roughly right at high S.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from probe_visuals.common import (
    A_CONS,
    A_FIT,
    B_CONS,
    B_FIT,
    C_BLUE,
    C_GREEN,
    C_GREY,
    C_RED,
    MEASURED_RSS_MB,
    PROBE_SHAPES,
    SIZE_2D,
    save,
)


def main() -> None:
    # Only B=1 shapes for the 1-D chart
    b1_shapes = [(b, s) for b, s in PROBE_SHAPES if b == 1]
    seqs = np.array([s for _, s in b1_shapes], dtype=float)
    rss_mib = np.array([MEASURED_RSS_MB[(1, s)] for _, s in b1_shapes], dtype=float)

    S_smooth = np.linspace(1, 8192, 2000)
    fitted_bytes = A_FIT * S_smooth + B_FIT * S_smooth**2
    cons_bytes = A_CONS * S_smooth + B_CONS * S_smooth**2
    fitted_mib = fitted_bytes / 1024**2
    cons_mib = cons_bytes / 1024**2

    fig, ax = plt.subplots(figsize=SIZE_2D)

    ax.scatter(seqs, rss_mib, s=90, color=C_RED, zorder=5, label="Measured RSS delta  (from §12)")

    ax.plot(
        S_smooth,
        fitted_mib,
        color=C_BLUE,
        lw=2.5,
        label=(f"Fitted: $a_{{fit}}·S + b_{{fit}}·S^2$\n$a$ = {A_FIT:,} B  $b$ = {B_FIT} B/tok²"),
    )

    ax.plot(
        S_smooth,
        cons_mib,
        color=C_GREEN,
        lw=2.0,
        ls="--",
        label=(
            f"Conservative: $a_{{cons}}·S + b_{{cons}}·S^2$\n"
            f"$a$ = {A_CONS:,} B  $b$ = {B_CONS} B/tok²"
        ),
    )

    # Residual lines
    for b, s in b1_shapes:
        y_m = MEASURED_RSS_MB[(b, s)]
        y_f = (A_FIT * s + B_FIT * s**2) / 1024**2
        ax.vlines(s, min(y_m, y_f), max(y_m, y_f), colors=C_GREY, lw=1.2, alpha=0.6)

    # Annotate each measured point
    for b, s in b1_shapes:
        rss = MEASURED_RSS_MB[(b, s)]
        ax.annotate(f"{rss} MiB", (s, rss), textcoords="offset points", xytext=(8, 2), fontsize=8)

    # Mark crossover
    crossover_S = A_FIT / B_FIT
    ax.axvline(crossover_S, color=C_GREY, lw=1.0, ls=":", zorder=0)
    ylim = ax.get_ylim()
    ax.text(
        crossover_S + 120,
        ylim[1] * 0.1 if ylim[1] > 0 else 30,
        f"$S^*={crossover_S:,.0f}$",
        fontsize=8,
        color="grey",
    )

    ax.set_xlabel("Sequence length  S  (tokens,  B = 1)")
    ax.set_ylabel("RSS delta  (MiB)")
    ax.set_title(
        "Probe Fit Quality: Measured vs. Fitted vs. Conservative Defaults\n"
        "(conservative defaults systematically over-predict at low S)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 8500)
    ax.set_ylim(0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))

    ax.text(
        0.97,
        0.30,
        "Conservative defaults\novershoot at low S\n(safe but wastes budget)",
        transform=ax.transAxes,
        fontsize=8,
        va="center",
        ha="right",
        color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", alpha=0.85),
    )

    fig.tight_layout()
    path = save(fig, "fig09_fit_quality")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
