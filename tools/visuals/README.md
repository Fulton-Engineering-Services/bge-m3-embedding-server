# Startup-Probe Visualisations

This directory contains ten Python figure scripts that produce the mathematical diagrams
supplementing [`docs/startup-probe.md`](../../docs/startup-probe.md). Each figure illustrates
a distinct concept from the startup workspace probe: the quadratic cost model, ordinary least
squares geometry, Jacobi column normalisation, probe shape information theory, fit quality, and
the asymmetric clamping policy. The scripts are self-contained, headless (Matplotlib Agg
backend), and render PNGs to `out/`.

---

## Prerequisites

- **Python ≥ 3.13**
- **uv ≥ 0.4**

### Install uv

```bash
# macOS (Homebrew)
brew install uv

# Linux / macOS (official installer)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quick start

```bash
cd tools/visuals

# Install dependencies into an isolated .venv
uv sync

# Render all 10 figures → out/fig01_*.png … out/fig10_*.png
uv run python scripts/render_all.py
```

PNGs appear in `out/`. The script prints each figure name, elapsed time, and final file sizes.

---

## Figure index

| Figure | File | Section in startup-probe.md | Description |
|--------|------|-----------------------------|-------------|
| fig01 | `figures/fig01_cost_decomposition.py` | §2 — Where the Quadratic Term Comes From | Linear vs. quadratic workspace terms on linear and log-log axes; crossover point S* = a/b ≈ 2,973 |
| fig02 | `figures/fig02_workspace_surface.py` | §2/§3 — Geometric Intuition for the Budget | 3-D surface W(B,S) with 2 GiB budget plane and floor contour showing `CostModel::fits()` region |
| fig03 | `figures/fig03_ols_geometry.py` | §4 — Ordinary Least Squares Without Intercept | 3-D scatter of 7 probe shapes in (x₁,x₂,y) space; best-fit plane through origin; residual segments |
| fig04 | `figures/fig04_loss_landscape_conditioning.py` | §5 — The Conditioning Problem | Side-by-side OLS loss contours: elongated (raw) vs. near-circular (Jacobi-normalised); gradient step comparison |
| fig05 | `figures/fig05_column_magnitudes.py` | §5 — Why Scale Matters | Bar chart: raw column magnitudes (up to 8,000× ratio) vs. normalised columns in [0,1] |
| fig06 | `figures/fig06_jacobi_transformation.py` | §6 — Column Normalisation as a Coordinate Change | Scatter in raw log-log space vs. normalised [0,1]² space; (4,64) off-arc point highlighted |
| fig07 | `figures/fig07_probe_shape_information.py` | §7 — Information Geometry for Two Coefficients | Log-log scatter annotated with the (4,64)↔(1,256) bracket isolating the b coefficient |
| fig08 | `figures/fig08_collinearity_failure.py` | §7 — Why Not Just Sweep One Direction | Loss landscape: chosen shapes (compact ellipse) vs. all-B=1 shapes (degenerate valley); Gram determinant labels |
| fig09 | `figures/fig09_fit_quality.py` | §8/§13 — From Measurement to Cost Model | Measured RSS deltas vs. fitted curve vs. conservative defaults; over-prediction at low S visible |
| fig10 | `figures/fig10_clamp_asymmetry.py` | §9 — Asymmetric Clamping | Piecewise clamp functions for a (floor, no reject) and b (negative → reject → conservative defaults) |

---

## Running a single figure

```bash
uv run python src/visuals/figures/fig01_cost_decomposition.py
```

Each figure script has a `main()` function and a standard `if __name__ == "__main__": main()`
guard, so they can be run directly or imported without side effects.

---

## Lint / format

```bash
# Check for lint errors
uv run ruff check src scripts

# Auto-fix safe issues
uv run ruff check --fix src scripts

# Format
uv run ruff format src scripts

# Format check (CI gate)
uv run ruff format --check src scripts
```

---

## Adding a new figure

1. Create `src/visuals/figures/figNN_descriptive_name.py` following the existing pattern:
   - Import shared constants/helpers from `visuals.common`
   - Wrap all plotting code inside `def main() -> None:`
   - Add `if __name__ == "__main__": main()` at the bottom
   - Call `save(fig, "figNN_descriptive_name")` to write to `out/`
2. Register the new module in `scripts/render_all.py`:
   - Add the import at the top
   - Add an entry to the `FIGURES` list
3. Update the figure index table in this README
4. Optionally cross-reference the new figure from `docs/startup-probe.md`

---

## Output

`out/` is **gitignored** — regenerate on demand with `uv run python scripts/render_all.py`.

**Follow-on plan:** once the figures have been reviewed, the PNGs will be moved to
`docs/figures/startup-probe/` and embedded in `docs/startup-probe.md` using standard Markdown
image links.

---

## Notebooks

Interactive ipywidgets notebooks for hands-on exploration. Require Jupyter and ipywidgets.

### Available notebooks

| Notebook | Description |
|---|---|
| `01_cost_decomposition_explorer.ipynb` | Sliders for `a` and `b` — live crossover point, preset buttons for fitted vs conservative defaults |
| `02_workspace_budget_calculator.ipynb` | Deployment sizing tool — workers, model RSS, available memory, safety factor → utilization traffic light |
| `03_conditioning_visualiser.ipynb` | Column scale ratio slider — morphs OLS loss landscape from circular to elongated, shows condition number |

GitHub renders the markdown and code cells statically. For interactive use, run locally with the terminal path below.

Notebooks use the `ipympl` backend — each widget interaction redraws the live matplotlib canvas in-cell. Pan and zoom work on all figure panels.

---

## Running the Notebooks

### Terminal + Browser (recommended — always works)

This is the confirmed reliable path. Run from the `tools/visuals/` directory:

```bash
# 1. Install all notebook dependencies (first time only)
uv sync --group notebooks

# 2. Register the kernel so Jupyter can find it (first time only, or after recreating the venv)
uv run python -m ipykernel install --user --name bge-m3-visuals --display-name "BGE-M3 Visuals"

# 3. Launch Jupyter in the browser
uv run jupyter notebook notebooks/
```

Jupyter opens at `http://localhost:8888` with a file browser. Click any notebook to open it, then **Run → Run All Cells** (or Shift+Enter through each cell). The widgets appear inline — sliders are interactive immediately.

**Why this path:** `uv run jupyter notebook` launches Jupyter with the project venv's Python automatically. No kernel selection needed.

### VS Code / Cursor

After running the `ipykernel install` command above, VS Code offers **"BGE-M3 Visuals"** in the kernel picker (top-right of the notebook). Select it, then **Run All**.

If VS Code does not show the kernel: `Ctrl+Shift+P` → **Developer: Reload Window**, then reopen the notebook.

**Note:** If the kernel picker shows a pyenv Python that gets killed (SIGKILL), the project venv is backed by Homebrew Python. Recreate the venv with:

```bash
rm -rf .venv
uv venv --python /opt/homebrew/bin/python3.13
uv sync --group dev --group notebooks
uv run python -m ipykernel install --user --name bge-m3-visuals --display-name "BGE-M3 Visuals"
```

### Migrating from the old `bge-m3-probe-visuals` kernel

The kernel was renamed from `bge-m3-probe-visuals` to `bge-m3-visuals` (display name
"BGE-M3 Visuals") when the tool moved from `tools/probe-visuals/` to `tools/visuals/`.
If you previously registered the old kernel, remove it and register the new one:

```bash
uv run jupyter kernelspec remove -y bge-m3-probe-visuals
uv run python -m ipykernel install --user --name bge-m3-visuals --display-name "BGE-M3 Visuals"
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'matplotlib'` | Wrong kernel selected | Select "BGE-M3 Visuals" kernel or use terminal path |
| `'widget' is not a recognised backend name` | ipympl not in active kernel | Same as above |
| `FigureCanvasAgg is non-interactive` | Old cached output | Restart kernel and run all cells |
| Figure clipped in browser | Browser window too narrow | Widen browser window or zoom out |
| SIGKILL on Python binary | pyenv binary blocked by Gatekeeper | Recreate venv with Homebrew Python (see above) |

---

## Source-of-truth constants

All figures share the constants defined in `src/visuals/common.py`.

| Constant | Value | Meaning |
|----------|-------|---------|
| `A_FIT` | `18432.0` | Fitted linear coefficient — bytes per token-position (a in W = a·B·S + b·B·S²) |
| `B_FIT` | `6.2` | Fitted quadratic coefficient — bytes per token-position² |
| `A_CONS` | `16384.0` | Conservative fallback for a (used when probe cannot run or b < 0) |
| `B_CONS` | `8.0` | Conservative fallback for b |
| `MAX_SEQ` | `8192` | Maximum tokenized sequence length supported by the server |

These values mirror the constants documented in `docs/startup-probe.md` §2 and §9.
`common.py` is the single source of truth — update it there and every figure updates automatically.
