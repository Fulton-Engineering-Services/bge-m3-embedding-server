# 7. Measurement

A measurement-driven cost model is only as good as its measurements. The probe must answer two questions for each shape: what to feed the model, and how to measure how much memory the resulting `session.run()` actually allocated. Both are subtler than they first appear.

The first question matters because real text has substructure. Natural English averages roughly four characters per token, but synthetic strings of a single repeated character can tokenise very differently due to BPE's run-length-aware merges. Using fake inputs would cause the probe to measure a different ORT execution path than real traffic — repetitive sequences trigger different attention patterns and may take different fast paths through the kernel. The probe therefore pulls real strings from a curated corpus and repeat-trims them to the desired length.

The second question matters because ORT does not expose its workspace allocator state. The OS provides Resident Set Size (RSS) — the total resident memory of the process — and the change in RSS across one `session.run()` call must be attributed to the workspace that call requested. The arena allocator retains pages across calls, OS page granularity introduces noise, and other workers in the process can perturb the reading. The probe handles this with a two-layer RSS scheme: one measurement at worker startup to establish the model + arena baseline, and one measurement around each `session.run()` to capture the per-shape workspace delta.

## Fit quality as a calibration check

![Figure 9 — Scatter: x-axis is sequence length S; y-axis is RSS delta in MB; seven measured probe points (dots) overlaid on the fitted quadratic curve y = a·B·S + b·B·S² (solid line) and the conservative-defaults curve (dashed); residuals visible as the gap between dots and curve.](../figures/startup-probe/fig09_fit_quality.png)

Figure 9 plots the seven probe measurements (filled dots) against the cost model's prediction. The solid curve is $y = a \cdot B \cdot S + b \cdot B \cdot S^2$ with the fitted $(a, b)$. The dashed curve uses the conservative defaults $(\text{CONSERVATIVE\_A}, \text{CONSERVATIVE\_B})$ for comparison. Each dot's vertical distance to the solid curve is the residual that OLS minimises.

A good fit appears as dots scattered tightly around the solid curve with no systematic over- or under-prediction at any sequence-length range. The fitted curve sits well below the conservative-defaults curve at most points, which represents the throughput win — the bin-packer can pack more aggressively with the fitted model than with the conservative one.

This figure is the calibration check. After every probe run, the question "do the measurements actually fit the model?" can be answered by inspection. Dots that wander far from the curve indicate a misspecified model, noisy measurements, or hardware behaviour the model does not account for; §8 covers what happens when the fit goes off the rails.

## Synthesising inputs

The probe needs $B$ texts each tokenising to approximately $S$ tokens. The probe does not have a tokeniser handy (it lives in the worker), so it approximates: at ${\sim}4$ characters per token for natural English, an $S$-token input is ${\sim}4S$ characters. The probe synthesises batches by repeating curated corpus snippets and trimming:

```53:69:src/probe/corpus.rs
fn synthesize_texts(corpus: &[String], batch: usize, target_seq: usize) -> Vec<String> {
    let target_chars = target_seq.saturating_mul(4).max(16);
    (0..batch)
        .map(|i| {
            let base = &corpus[i % corpus.len()];
            // Repeat the base text until we have enough characters.
            let repeated = base.repeat((target_chars / base.len().max(1)).max(2) + 1);
            // Trim to target_chars bytes (not chars, but close enough for probing).
            let trimmed = if repeated.len() > target_chars {
                &repeated[..target_chars]
            } else {
                &repeated
            };
            trimmed.to_string()
        })
        .collect()
}
```

The corpus is the same fixture used by the benchmarks (`benches/fixtures/corpus.json`) — real production-shaped strings drawn from three databases. Synthetic strings of a single repeated character can tokenise very differently from natural language (run-length compression, fewer subword splits), and the workspace measurement should reflect realistic ORT execution paths.

The resulting tokenised lengths are approximate. The tokeniser truncates to the configured `max_seq_length` upper bound, so the probe shape $(B, S)$ becomes "$B$ texts, each padded to at most $S$ tokens" — exactly what the bin-packer needs to predict.

### The four-characters-per-token approximation

XLM-RoBERTa's BPE vocabulary is multilingual ($250\,\mathrm{k}$ tokens). For natural English prose, the empirical mean tokens-per-character lies near $0.25$ — i.e., four characters per token. The probe uses this as a rough conversion factor: a $1024$-token target becomes a $4096$-character string, which the tokeniser then trims to exactly $1024$ tokens during the `session.run()` call.

The conversion is not precise — non-English text, code, or text with many short words tokenises denser, while text with long technical terms tokenises sparser. But the probe does not need exact $S$; it needs $S$ within a few percent, which the trim-and-pad behaviour of the tokeniser guarantees. The cost model fits the actual padded sequence length from the tokeniser, not the requested one.

## Measuring RSS across two layers

RSS is measured at two points in the startup sequence, serving two distinct purposes.

### Layer 1: per-shape workspace measurement

Inside `probe_run_dense`, the probe wraps each `session.run()` with RSS reads taken immediately before and after the call:

```85:109:src/embedder/worker.rs
pub(crate) fn probe_run_dense(
    session: &mut ort::session::Session,
    ids_array: &ndarray::Array2<i64>,
    mask_array: &ndarray::Array2<i64>,
) -> Result<ProbeResult> {
    let rss_before = sysinfo::read_process_rss_bytes().unwrap_or(0);
    let ids_tensor = TensorRef::from_array_view(ids_array.view()).map_err(ort_err)?;
    let mask_tensor = TensorRef::from_array_view(mask_array.view()).map_err(ort_err)?;
    // Run inference (output discarded — we only care about RSS).
    let _outputs = session
        .run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })
        .map_err(ort_err)?;
    let rss_after = sysinfo::read_process_rss_bytes().unwrap_or(rss_before);
    Ok(ProbeResult {
        rss_before,
        rss_after,
    })
}
```

These per-shape deltas (`rss_after - rss_before`) feed the OLS fit and determine $(a, b)$. The probe holds all worker permits during the sweep (§10), so no concurrent traffic perturbs the readings.

### Layer 2: per-worker model-weight + arena-baseline footprint

Each worker measures its own RSS immediately *before* `load_models()` and *after* a per-worker arena-priming `session.run()`, inside the `spawn_blocking` thread where the ORT session is actually created:

```rust
let pre_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(0);
let initial_models = match load_models(...) {
    Ok(mut models) => {
        // Prime the ORT session arena with a tiny session.run() BEFORE
        // measuring post-load RSS. ORT lazily allocates ~1 GiB of arena
        // bookkeeping on the first run() call regardless of input size;
        // priming here folds that allocation into the per-worker model
        // RSS measurement so the workspace-budget math sees the realistic
        // per-worker memory footprint, AND so the probe sweep's per-shape
        // rss_delta readings reflect only the incremental workspace
        // attributable to that shape.
        let prime_ids = ndarray::Array2::<i64>::zeros((1, 8));
        let prime_mask = ndarray::Array2::<i64>::ones((1, 8));
        let _ = probe_run_dense(&mut models.0, &prime_ids, &prime_mask);
        models
    }
    Err(e) => { /* … */ }
};
let post_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(pre_load_rss);
let rss_delta = post_load_rss.saturating_sub(pre_load_rss);
let _ = rt.block_on(ready_tx.send(Ok(rss_delta)));
```

The `ready_tx` channel carries `Result<usize>` (the delta in bytes). `EmbedPool::spawn` collects all worker deltas and stores the *median* via `Arc<AtomicUsize>` — the median is robust to one outlier from page-cache settling or ORT arena jitter. The caller reads `state.pool.model_rss_per_worker_bytes()` to get the per-worker model + arena footprint used in the workspace-budget formula:

```
total_workspace = available − N × model_rss_per_worker − OS_HEADROOM
per_worker_workspace = total_workspace × safety_factor / N
```

### Why both layers are necessary

The alternative — reading RSS twice immediately after all workers have finished loading, or measuring before any `session.run()` — under-counts each worker's contribution by ${\sim}1\,\text{GiB}$. With `model_rss_per_worker` ${\approx}1.4\,\text{GiB}$ and 4 workers on 28 GiB, the budget formula over-budgets each worker by ${\sim}1\,\text{GiB}$. Real $(1, 8192)$ requests subsequently grow each worker's arena to its high-water mark (${\sim}5\text{–}6\,\text{GiB}$ total per worker), which fits the corrected budget at 4 workers but not at 7 workers. Per-worker priming was added in v0.15.0-rc7 specifically to make the budget reflect arena baselines.

The two layers correspond to two distinct quantities:

- **Layer 2** captures what each worker costs simply by existing: model weights plus the ${\sim}1\,\text{GiB}$ ORT arena baseline that the first `session.run()` allocates regardless of input size.
- **Layer 1** captures what each *shape* adds on top of the existing arena: the incremental per-call workspace attributable to a particular $(B, S)$.

If Layer 2 priming is skipped, the *first* per-shape measurement in Layer 1 includes the ${\sim}1\,\text{GiB}$ arena initialisation, dwarfing the actual workspace cost and corrupting the OLS fit. By priming each worker's session before measuring, the arena initialisation is folded into the per-worker baseline (where it belongs), and Layer 1 reports clean per-shape deltas.

## Reading `/proc/self/statm`

`read_process_rss_bytes()` parses field 1 of `/proc/self/statm` (RSS in pages) and multiplies by the page size ($4096$ on Linux/x86_64 and Linux/aarch64). The delta `rss_after - rss_before` is the RSS growth attributable to the call.

This is *not* the same as the peak workspace allocated during the call. RSS is a high-water mark as observed at the moment of reading; it captures pages still resident immediately after the call returns. ORT's arena allocator typically retains its high-water buffer rather than releasing pages back to the kernel, so the RSS delta is a good proxy for peak workspace on the *first* allocation at a given shape and a near-zero floor on repeat calls (the arena is already big enough).

The probe uses each shape exactly once for this reason: the first call at a shape grows the arena to that shape's peak, which the RSS delta captures. Subsequent calls at the same shape would show ${\sim}$zero delta and be useless data points.

### Other allocations during the call

The probe holds all worker permits (§10) during the sweep, so no concurrent inference can grow the arena. Some baseline RSS noise remains:

- Tokio worker threads allocating small per-task buffers
- Kernel page-cache reads for the model weights (already resident, but bookkeeping changes)
- Tracing/log allocations from the probe's own info-level events

These are all in the kilobyte range. A typical probe shape produces an RSS delta in the tens to hundreds of megabytes, so the noise floor is comfortably below the signal. The OLS fit absorbs what remains.

## Linux-only constraints

Both cgroup memory detection and `/proc/self/statm` are Linux-specific. On macOS:

- `detect_available_memory()` falls back to `sysctl hw.memsize` for total host RAM.
- `read_process_rss_bytes()` returns `None`, because reading task RSS requires `task_info` FFI, which conflicts with `unsafe_code = "forbid"`.

The probe logs a warning and uses conservative defaults on macOS. The native macOS deployment (LaunchAgent build) instead uses the CoreML-tuned plist defaults — see [deployment.md](../deployment.md). For Docker-on-macOS (Apple Silicon dev), the probe runs inside a Linux container so RSS reads work, but MLAS-only inference makes the probe sweep slow (several minutes), so most developers set `BGE_M3_DISABLE_AUTO_BUDGET=1` for fast iteration.

## Median across workers

Layer 2 reports one RSS delta per worker. With seven workers, that is seven numbers. The pool stores the *median* rather than the mean or maximum for two reasons:

1. **Robustness to outliers.** Page-cache settling or ORT arena jitter can produce one anomalously high reading. The mean would be dragged by that outlier; the median is unaffected.
2. **Robustness to measurement contamination.** Workers load *sequentially* (not in parallel) for exactly this reason — `/proc/self/statm` reports process-wide RSS, so concurrent loads would inflate every reading after the first. The median across sequentially measured workers gives a per-worker number that is robust to the residual noise.

The median is what flows into the budget formula and what appears in `/health` as `model_rss_bytes_per_worker`.

## Common failure modes

| Symptom | Likely cause | What the probe does |
|---------|--------------|----------------------|
| `model_rss_bytes_per_worker` is 0 in `/health` | Non-Linux platform or `/proc/self/statm` unreadable | Budget formula deducts 0 for model weights; over-budgets workers; operator should set `BGE_M3_MEMORY_SAFETY_FACTOR=0.5` as a stopgap |
| Layer 1 RSS deltas are all ${\sim}0$ | Workers loading in parallel (regression bug) — process-wide RSS is contaminated | Probe fails; conservative defaults applied; investigate `EmbedPool::spawn` |
| Layer 1 RSS delta for $(1, 64)$ is ${\sim}1\,\text{GiB}$ | Arena warm-up did not run, or the wrong measurement layer is being read | First-shape `RSS_delta` includes the lazy arena init; OLS fit will see one outlier and may produce a bad $(a, b)$ |
| Probe times out on a slow architecture | aarch64 MLAS is slow at large $S$; probe sweep takes > 5 minutes | Switch to a different model variant or set `BGE_M3_DISABLE_AUTO_BUDGET=1` and rely on conservative defaults |

---

← [Previous: Probe shapes](06-probe-shapes.md) | [↑ Series overview](../startup-probe.md) | [Next: Clamps & fallback →](08-clamps-fallback.md)
