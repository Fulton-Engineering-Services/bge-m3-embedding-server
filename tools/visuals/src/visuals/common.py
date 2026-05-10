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
common.py — shared style, constants, and helpers for probe-visuals figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# rcParams — mathtext only, no LaTeX binary required
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)

# ---------------------------------------------------------------------------
# Figure sizes
# ---------------------------------------------------------------------------
SIZE_2D = (10, 6)  # standard 2-D figures
SIZE_3D = (9, 7)  # 3-D surface figures

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Paul Tol "bright" subset, 7 distinct colours)
# ---------------------------------------------------------------------------
COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]
C_BLUE, C_RED, C_GREEN, C_YELLOW, C_CYAN, C_PURPLE, C_GREY = COLORS

# ---------------------------------------------------------------------------
# Source-of-truth model constants (mirror §2/§9 of docs/startup-probe.md)
# ---------------------------------------------------------------------------
A_FIT = 18432.0  # bytes / token-position  (fitted)
B_FIT = 6.2  # bytes / token-position² (fitted)
A_CONS = 16384.0  # conservative default for a
B_CONS = 8.0  # conservative default for b
MAX_SEQ = 8192

# 7 probe shapes (batch, seq_len)
PROBE_SHAPES = [(1, 64), (4, 64), (1, 256), (1, 1024), (1, 2048), (1, 4096), (1, 8192)]

# Measured RSS deltas from §12 of docs/startup-probe.md (in MiB)
MEASURED_RSS_MB = {
    (1, 64): 2,
    (4, 64): 8,
    (1, 256): 6,
    (1, 1024): 27,
    (1, 2048): 68,
    (1, 4096): 210,
    (1, 8192): 720,
}

# ---------------------------------------------------------------------------
# Output directory — tools/probe-visuals/out/
# ---------------------------------------------------------------------------
# __file__ is tools/probe-visuals/src/probe_visuals/common.py
# .parent.parent.parent → tools/probe-visuals/
OUT_DIR: Path = Path(__file__).parent.parent.parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, name: str, out_dir: Path | None = None) -> str:
    """Write fig to <out_dir>/<name>.png with tight layout; return the full path."""
    target = (out_dir if out_dir is not None else OUT_DIR) / (name + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target), bbox_inches="tight")
    plt.close(fig)
    return str(target)


def save_animation(
    anim: FuncAnimation,
    name: str,
    fps: int = 15,
    out_dir: Path | None = None,
    dpi: int = 100,
) -> str:
    """Write anim to <out_dir>/<name>.gif using PillowWriter; return full path."""
    target = (out_dir if out_dir is not None else OUT_DIR) / (name + ".gif")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps, metadata={"loop": 0})
    anim.save(str(target), writer=writer, dpi=dpi)
    return str(target)
