# 7. Measurement — Synthesizing Inputs and Reading RSS

> The probe needs `B` realistic-looking texts each tokenizing to about `S` tokens, and it needs to know how much memory the resulting `session.run()` actually allocated. Both of those sound simple. Both have subtleties.

## Intuition

A measurement-driven cost model is only as good as its measurements. Two questions dominate:

1. **What do you feed the model?** Real text has substructure — natural English averages roughly 4 characters per token, but synthetic strings of a single repeated character can tokenize very differently due to BPE's run-length-aware merges. Using fake inputs would cause the probe to measure a different ORT execution path than real traffic — for instance, repetitive sequences trigger different attention patterns and may take different fast paths through the kernel.
2. **How do you measure how much memory it used?** ORT doesn't expose its workspace allocator state. The OS gives us *Resident Set Size* (RSS) — the total resident memory of the process — and we have to attribute the change in RSS across one `session.run()` call to the workspace that call requested. That's harder than it sounds, because ORT's arena allocator retains pages across calls, OS page granularity introduces noise, and other workers in the process can perturb the reading.

The probe handles both with care. For inputs, it pulls real strings from a curated corpus and repeat-trims them to the desired length. For measurements, it uses a *two-layer* RSS scheme: one measurement at worker startup to establish the model+arena baseline, and a second measurement around each `session.run()` call to capture the per-shape workspace delta.

## The figure

![Scatter plot: x-axis is sequence length S; y-axis is RSS delta in MB; seven measured probe points (dots) overlaid on the fitted quadratic curve y = a·B·S + b·B·S² (solid line) and the conservative-defaults curve (dashed); residuals visible as the gap between dots and curve](../figures/startup-probe/fig09_fit_quality.png)

**What you're looking at:** the seven probe measurements (filled dots) plotted against the cost model's prediction. The solid curve is `y = a·B·S + b·B·S²` with the *fitted* `(a, b)`. The dashed curve uses the *conservative* defaults `(CONSERVATIVE_A, CONSERVATIVE_B)` for comparison. Each dot's vertical distance to the solid curve is the **residual** — what OLS minimizes.

A good fit looks like this: dots scattered tightly around the solid curve, with no systematic over- or under-prediction at any sequence-length range. The fitted curve sits well below the conservative-defaults curve at most points, which is the throughput win — the bin-packer can pack more aggressively with the fitted model than with the conservative one.

**Why it matters:** this figure is the **calibration check**. After every probe run, you can ask "do the measurements actually fit the model?" If the dots wander far from the curve, something's wrong — the model is misspecified, the measurements are noisy, or the hardware is doing something the model doesn't account for. Page [Clamps & fallback](08-clamps-fallback.md) covers what happens when the fit goes off the rails.

## Synthesizing inputs

The probe needs `B` texts each tokenizing to approximately `S` tokens. The probe doesn't have a tokenizer handy (it lives in the worker), so it approximates: at ~4 chars/token for natural English, an `S`-token input is ~`4S` characters. The probe synthesizes batches by repeating curated corpus snippets and trimming:

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

The corpus is the same fixture used by the benchmarks (`benches/fixtures/corpus.json`) — real production-shaped strings drawn from three databases. This matters: synthetic strings of a single repeated character can tokenize very differently from natural language (run-length compression, fewer subword splits), and we want the workspace measurement to reflect realistic ORT execution paths.

The resulting tokenized lengths are *approximate*. The tokenizer truncates to the configured `max_seq_length` upper bound, so the probe shape `(B, S)` becomes "B texts, each padded to at most S tokens" — which is exactly what the bin-packer needs to predict.

### Why ~4 characters per token

XLM-RoBERTa's BPE vocabulary is multilingual (250k tokens). For natural English prose, the empirical mean tokens-per-character lands near 0.25 — i.e., 4 characters per token. The probe uses this as a rough conversion factor: a 1024-token target becomes a 4096-character string, which the tokenizer then trims to exactly 1024 tokens during the actual `session.run()` call.

The conversion isn't precise — non-English text, code, or text with many short words can tokenize denser; text with long technical terms tokenizes sparser. But the probe doesn't need exact `S` — it needs `S ± a few percent`, which the trim-and-pad behavior of the tokenizer guarantees. The cost model fits *the actual padded sequence length* from the tokenizer, not the requested one.

## Measuring RSS — two layers

RSS is measured at two points in the startup sequence, serving two different purposes.

### Layer 1 — Per-shape workspace measurement

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

These per-shape deltas (`rss_after - rss_before`) feed the OLS fit and determine `(a, b)`. The probe holds all worker permits during the sweep (see [Execution](10-execution.md)), so no concurrent traffic perturbs the readings.

### Layer 2 — Per-worker model-weight + arena-baseline footprint

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

The `ready_tx` channel carries `Result<usize>` (the delta in bytes). `EmbedPool::spawn` collects all worker deltas and stores the **median** via `Arc<AtomicUsize>` — median is robust to one outlier from page-cache settling or ORT arena jitter. The caller reads `state.pool.model_rss_per_worker_bytes()` to get the per-worker model+arena footprint used in the workspace-budget formula:

```
total_workspace = available − N × model_rss_per_worker − OS_HEADROOM
per_worker_workspace = total_workspace × safety_factor / N
```

### Why both layers are necessary

The alternative (reading RSS twice immediately after all workers have finished loading, OR measuring before any `session.run()`) under-counts each worker's contribution by ~1 GiB. With `model_rss_per_worker ≈ 1.4 GiB` and 4 workers on 28 GiB, the budget formula over-budgets each worker by ~1 GiB. Real `(1, 8192)` requests subsequently grow each worker's arena to its high-water mark (~5–6 GiB total per worker) — which fits the corrected budget at 4 workers but does NOT fit at 7 workers. Per-worker priming was added in v0.15.0-rc7 specifically to make the budget reflect arena baselines.

You can think of it this way:

- **Layer 2 captures "what does this worker cost just by existing?"** — model weights plus the ~1 GiB ORT arena baseline that the first `session.run()` allocates regardless of input size.
- **Layer 1 captures "what does this *shape* add on top of the existing arena?"** — the incremental per-call workspace attributable to a particular `(B, S)`.

If you skip Layer 2 priming, then the *first* per-shape measurement in Layer 1 will include the ~1 GiB arena initialization, dwarfing the actual workspace cost and corrupting the OLS fit. By priming each worker's session before measuring, the arena initialization is folded into the per-worker baseline (where it belongs), and Layer 1 can report clean per-shape deltas.

## Reading `/proc/self/statm`

`read_process_rss_bytes()` parses field 1 of `/proc/self/statm` (RSS in pages) and multiplies by the page size (4096 on Linux/x86_64 and Linux/aarch64). The delta `rss_after - rss_before` is the RSS growth attributable to the call.

This is *not* the same as the peak workspace allocated during the call. RSS is a high-water mark *as observed at the moment of reading* — it captures pages still resident immediately after the call returns. ORT's arena allocator typically retains its high-water buffer rather than releasing pages back to the kernel, so the RSS delta is a good proxy for peak workspace **on the first allocation at a given shape** and a near-zero floor on repeat calls (the arena is already big enough).

The probe uses each shape exactly once for this reason: the first call at a shape grows the arena to that shape's peak, which the RSS delta captures. Subsequent calls at the same shape would show ~zero delta and be useless data points.

### What about other allocations during the call?

The probe holds all worker permits (page [Execution](10-execution.md)) during the sweep, so no concurrent inference can grow the arena. But there's still some baseline RSS noise from:

- **Tokio worker threads** allocating small per-task buffers
- **Kernel page-cache reads** for the model weights (already-resident, but bookkeeping changes)
- **Tracing/log allocations** from the probe's own info-level events

These are all in the kilobyte range. A typical probe shape produces an RSS delta in the tens-to-hundreds of megabytes, so the noise floor is comfortably below the signal. The OLS fit absorbs what's left.

## The Linux-only constraint

Both `cgroup` memory detection and `/proc/self/statm` are Linux-specific. On macOS:

- `detect_available_memory()` falls back to `sysctl hw.memsize` for total host RAM.
- `read_process_rss_bytes()` returns `None`, because reading task RSS requires `task_info` FFI, which conflicts with `unsafe_code = "forbid"`.

The probe logs a warning and uses conservative defaults on macOS. The native macOS deployment (LaunchAgent build) instead uses the CoreML-tuned plist defaults — see [deployment.md](../deployment.md).

For Docker-on-macOS (Apple Silicon dev), the probe runs inside a Linux container so RSS reads work — but MLAS-only inference makes the probe sweep slow (several minutes), so most devs set `BGE_M3_DISABLE_AUTO_BUDGET=1` for fast iteration.

## Why median across workers

Layer 2 reports one RSS delta per worker. With 7 workers, that's 7 numbers. The pool stores the **median** rather than the mean or max for two reasons:

1. **Robustness to outliers.** Page-cache settling or ORT arena jitter can produce one anomalously high reading. The mean would be dragged by that outlier; the median is unaffected.
2. **Robustness to measurement contamination.** Workers load *sequentially* (not in parallel) for exactly this reason — `/proc/self/statm` reports process-wide RSS, so concurrent loads would inflate every reading after the first. The median across sequentially-measured workers gives a per-worker number that's robust to the residual noise.

The median is what flows into the budget formula and what shows up in `/health` as `model_rss_bytes_per_worker`.

## Common failure modes

| Symptom | Likely cause | What the probe does |
|---------|--------------|----------------------|
| `model_rss_bytes_per_worker` is 0 in `/health` | Non-Linux platform or `/proc/self/statm` unreadable | Budget formula deducts 0 for model weights; over-budgets workers; operator should set `BGE_M3_MEMORY_SAFETY_FACTOR=0.5` as stopgap |
| Layer 1 RSS deltas are all ~0 | Workers are loading parallel (regression bug) — process-wide RSS is contaminated | Probe `_failed`; conservative defaults applied; investigate `EmbedPool::spawn` |
| Layer 1 RSS delta for `(1, 64)` is ~1 GiB | Arena warm-up didn't run, or the wrong measurement layer is being read | First-shape `RSS_delta` includes the lazy arena init; OLS fit will see one outlier and may produce a bad `(a, b)` |
| Probe times out on a slow architecture | aarch64 MLAS is slow at large `S`; probe sweep takes > 5 minutes | Switch to a different model variant or set `BGE_M3_DISABLE_AUTO_BUDGET=1` and rely on conservative defaults |

## What's next

The next page covers what happens when the OLS fit produces something unreasonable — coefficients outside physical bounds, fits that fail the singular-system check, capability errors that can't run a probe shape at all. The probe's failure handling is asymmetric for a principled reason.

---

← [Previous: Probe shapes](06-probe-shapes.md) | [↑ Series overview](../startup-probe.md) | [Next: Clamps & fallback →](08-clamps-fallback.md)
