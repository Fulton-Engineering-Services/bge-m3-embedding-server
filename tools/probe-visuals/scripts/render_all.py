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
render_all.py — render all 10 probe-visual figures and report results.

Run from tools/probe-visuals/ with:
    uv run python scripts/render_all.py
"""

import sys
import time
import traceback

from probe_visuals.common import OUT_DIR
from probe_visuals.figures import (
    fig01_cost_decomposition,
    fig02_workspace_surface,
    fig03_ols_geometry,
    fig04_loss_landscape_conditioning,
    fig05_column_magnitudes,
    fig06_jacobi_transformation,
    fig07_probe_shape_information,
    fig08_collinearity_failure,
    fig09_fit_quality,
    fig10_clamp_asymmetry,
)

FIGURES = [
    ("fig01_cost_decomposition", fig01_cost_decomposition.main),
    ("fig02_workspace_surface", fig02_workspace_surface.main),
    ("fig03_ols_geometry", fig03_ols_geometry.main),
    ("fig04_loss_landscape_conditioning", fig04_loss_landscape_conditioning.main),
    ("fig05_column_magnitudes", fig05_column_magnitudes.main),
    ("fig06_jacobi_transformation", fig06_jacobi_transformation.main),
    ("fig07_probe_shape_information", fig07_probe_shape_information.main),
    ("fig08_collinearity_failure", fig08_collinearity_failure.main),
    ("fig09_fit_quality", fig09_fit_quality.main),
    ("fig10_clamp_asymmetry", fig10_clamp_asymmetry.main),
]


def main() -> int:
    print("=" * 60)
    print("render_all.py — BGE-M3 probe visual companions")
    print("=" * 60)

    failures = []
    total_start = time.perf_counter()

    for name, fn in FIGURES:
        print(f"\n▶ Running {name} …")
        t0 = time.perf_counter()
        try:
            fn()
            dt = time.perf_counter() - t0
            print(f"  ✓  ({dt:.1f}s)")
        except Exception as exc:
            dt = time.perf_counter() - t0
            print(f"  ✗ FAILED  ({dt:.1f}s): {exc}", file=sys.stderr)
            traceback.print_exc()
            failures.append(name)

    total_dt = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    n_succeeded = len(FIGURES) - len(failures)
    print(f"Completed in {total_dt:.1f}s  —  {n_succeeded}/{len(FIGURES)} succeeded")

    if failures:
        print(f"FAILURES: {failures}", file=sys.stderr)

    # Report PNG sizes
    print(f"\nPNGs in {OUT_DIR}:")
    pngs = sorted(OUT_DIR.glob("*.png"))
    for p in pngs:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:50s}  {size_kb:8.1f} KiB")

    if len(pngs) == len(FIGURES) and not failures:
        print("\nAll 10 PNGs generated successfully.")
        return 0

    missing = len(FIGURES) - len(pngs)
    print(f"\nWARNING: expected {len(FIGURES)} PNGs, found {len(pngs)}.", file=sys.stderr)
    if missing > 0:
        print(f"  {missing} PNG(s) missing — check FAILURES above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
