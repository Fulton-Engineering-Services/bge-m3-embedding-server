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
fig10_clamp_asymmetry.py
§9 — Asymmetric Clamping

Two stacked panels showing the piecewise-linear clamp functions.

Top panel: raw a_raw → output a
  Clamp band: [4 KiB, 256 KiB]  (= [4096, 262144] bytes)
  Negative a_raw → clamped to floor (4 KiB), NOT rejected.

Bottom panel: raw b_raw → output b
  Clamp band: [0.01, 50000] bytes/tok²
  Negative b_raw → REJECT → fall back to conservative defaults.

The asymmetry is the whole point: under-packing is slow; over-packing crashes.
"""

import matplotlib.pyplot as plt
import numpy as np

from visuals.common import A_FIT, B_FIT, C_BLUE, C_GREEN, C_GREY, C_RED, SIZE_2D, save

# ── Clamp bounds ──────────────────────────────────────────────────────────
A_LO, A_HI = 4_096.0, 262_144.0  # [4 KiB, 256 KiB]
B_LO, B_HI = 0.01, 50_000.0  # [0.01, 50k]


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(SIZE_2D[0], SIZE_2D[1] + 2))
    fig.suptitle(
        "Asymmetric Clamping of Fitted Coefficients $a$ and $b$\n"
        "Under-pack → slow   ·   Over-pack → OOM / crash",
        fontsize=12,
        fontweight="bold",
    )

    # ── Top: a clamp ──────────────────────────────────────────────────────────
    a_lo_plot = -A_HI * 0.6
    a_hi_plot = A_HI * 1.4
    a_raw = np.linspace(a_lo_plot, a_hi_plot, 2000)

    def clamp_a(x: np.ndarray) -> np.ndarray:
        return np.clip(x, A_LO, A_HI)

    a_out = clamp_a(a_raw)

    ax_a.axvspan(a_lo_plot, 0, alpha=0.08, color=C_GREY, zorder=0)
    ax_a.axvspan(0, A_LO, alpha=0.12, color=C_RED, zorder=0, label="Below floor → clamped to 4 KiB")
    ax_a.axvspan(
        A_LO, A_HI, alpha=0.10, color=C_GREEN, zorder=0, label="Valid band [4 KiB, 256 KiB]"
    )
    ax_a.axvspan(A_HI, a_hi_plot, alpha=0.10, color=C_RED, zorder=0)

    ax_a.plot(a_raw, a_out, color=C_BLUE, lw=2.5, label="$a_{out} = \\mathrm{clamp}(a_{raw})$")
    ax_a.axhline(A_LO, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_a.axhline(A_HI, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_a.axvline(0, color=C_GREY, lw=1.0, ls=":", alpha=0.7)
    ax_a.axvline(A_LO, color=C_GREEN, lw=1.0, ls=":", alpha=0.7)
    ax_a.axvline(A_HI, color=C_GREEN, lw=1.0, ls=":", alpha=0.7)

    ax_a.scatter(
        [A_FIT],
        [clamp_a(np.array([A_FIT]))[0]],
        s=80,
        color=C_RED,
        zorder=5,
        label=f"$a_{{fit}}$ = {A_FIT:,} B  (in-band)",
    )

    ax_a.text(
        -A_HI * 0.45,
        (A_LO + A_HI) * 0.7,
        "Negative $a_{raw}$\nclamped to floor\n(rc8 fix)",
        fontsize=8,
        color="navy",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightcyan", alpha=0.8),
    )

    ax_a.set_xlabel("$a_{raw}$  (bytes / token-position)")
    ax_a.set_ylabel("$a_{out}$  (bytes / token-position)")
    ax_a.set_title("Coefficient $a$: symmetric floor clamp (negatives → 4 KiB, NOT rejected)")
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y / 1024:.0f} KiB"))
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1024:.0f} KiB"))
    ax_a.set_xlim(a_lo_plot, a_hi_plot)

    # ── Bottom: b clamp ───────────────────────────────────────────────────────
    b_lo_plot = -B_HI * 0.3
    b_hi_plot = B_HI * 1.4
    b_raw = np.linspace(b_lo_plot, b_hi_plot, 2000)

    def clamp_b(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, np.nan, np.clip(x, B_LO, B_HI))

    b_out = clamp_b(b_raw)

    ax_b.axvspan(b_lo_plot, 0, alpha=0.15, color="firebrick", zorder=0)
    ax_b.axvspan(0, B_LO, alpha=0.12, color=C_RED, zorder=0)
    ax_b.axvspan(B_LO, B_HI, alpha=0.10, color=C_GREEN, zorder=0, label="Valid band [0.01, 50k]")
    ax_b.axvspan(B_HI, b_hi_plot, alpha=0.10, color=C_RED, zorder=0)

    # Plot valid portion only (no nan)
    mask = b_raw >= 0
    ax_b.plot(
        b_raw[mask],
        b_out[mask],
        color=C_BLUE,
        lw=2.5,
        label="$b_{out} = \\mathrm{clamp}(b_{raw})$",
    )
    ax_b.axhline(B_LO, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_b.axhline(B_HI, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_b.axvline(0, color="firebrick", lw=1.5, ls="-", alpha=0.7)

    # REJECT region label
    ax_b.text(
        b_lo_plot * 0.5,
        B_HI * 0.5,
        "REJECT\n(negative $b_{raw}$)\n→ conservative\ndefaults",
        fontsize=9,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="firebrick", alpha=0.85),
    )

    ax_b.scatter(
        [B_FIT],
        [clamp_b(np.array([B_FIT]))[0]],
        s=80,
        color=C_RED,
        zorder=5,
        label=f"$b_{{fit}}$ = {B_FIT} B/tok²  (in-band)",
    )

    ax_b.set_xlabel("$b_{raw}$  (bytes / token-position²)")
    ax_b.set_ylabel("$b_{out}$  (bytes / token-position²)")
    ax_b.set_title(
        "Coefficient $b$: REJECT negatives → fall back to conservative defaults\n"
        "(negative $b$ → model predicts shrinking memory — physically impossible)"
    )
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.set_xlim(b_lo_plot, b_hi_plot)

    fig.tight_layout()
    path = save(fig, "fig10_clamp_asymmetry")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
