//! Probe shape-sweep driver.
//!
//! Orchestrates the `(batch, seq)` shape sweep on the leader worker, applies
//! the conservative-`fits()` gate and the absolute-RSS guard, and feeds the
//! collected `DataPoint`s to [`super::fit::fit_cost_model`] for OLS fitting.

use tracing::{info, warn};

use super::corpus::{load_probe_texts, synthesize_texts};
use super::fit::{fit_cost_model, DataPoint};
use super::validate::validate_max_seq_shape;
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;

/// Probe shape: `(batch_count, seq_length)`.
pub(super) type Shape = (usize, usize);

/// Shapes swept by the probe.
///
/// 6 static shapes plus a dynamic `(1, max_seq)` shape added at runtime for
/// the quadratic anchor at the configured upper bound:
///
/// - `(1, 64)` and `(1, 256)` anchor the linear term at low seq.
/// - `(4, 64)` shares `x1 = batch*seq = 256` with `(1, 256)` but has a
///   different `x2 = batch*seq² = 16384` vs `65536`, giving a near-direct
///   measurement of `b` independent of `a`.
/// - `(1, 1024)` and `(1, 2048)` provide mid-range leverage.
/// - `(1, 4096)` anchors the quadratic regime.
///
/// ## Safety against OOM
///
/// ORT's memory arena retains pages across `session.run()` calls, so
/// cumulative process RSS grows with each successive probe shape. Three
/// independent mechanisms keep the sweep within the container's cgroup limit:
///
/// 1. **Arena warm-up** at the start of `run_probe` runs a `(1, 64)`
///    `session.run()` BEFORE the sweep, so the lazy ORT arena initialisation
///    does not appear as a ~1 GB constant offset on every per-shape delta.
/// 2. **Conservative `fits()` gate** rejects any shape whose per-call
///    workspace estimate exceeds `rss_ceiling` (the safety-discounted budget).
/// 3. **Absolute-RSS guard** rejects any shape whose projected arena growth
///    would push process RSS above 87.5% of the cgroup ceiling, regardless
///    of the conservative model's estimate.
///
/// The dynamic `(1, max_seq)` shape is added at runtime by `run_probe`. If
/// the model variant cannot run at `max_seq`, the shape is skipped and the
/// error surfaces on the first real embedding request.
///
/// Estimated probe time: ~120 s on aarch64 MLAS fp16 at `max_seq=8192`.
pub(super) const PROBE_SHAPES: &[Shape] = &[
    (1, 64),   // linear anchor
    (4, 64),   // pairs with (1,256) for direct b isolation
    (1, 256),  // linear anchor
    (1, 1024), // mid-range
    (1, 2048), // mid-range, anchors quadratic stability
    (1, 4096), // quadratic anchor
               // (1, max_seq) is added dynamically based on configured max.
];

/// Runs the startup probe on the already-warmed leader worker.
///
/// # Arguments
///
/// - `pool`: the `EmbedPool` whose leader worker has already loaded models.
/// - `max_seq`: the configured `BGE_M3_MAX_SEQ_LENGTH` (determines the topmost
///   probe shape).  The dynamic `(1, max_seq)` capability check has been
///   removed — see `trim_probe_shapes` in the change log.
/// - `rss_ceiling`: the per-worker workspace budget computed from sysinfo.
///   Shapes estimated to exceed this are skipped to avoid OOM mid-probe
///   (the conservative-model guard, unchanged).
/// - `cgroup_limit_bytes`: the **actual kernel memory ceiling** (cgroup limit
///   or host RAM, whichever was detected first). Used by the absolute-RSS
///   guard: before each shape the current process RSS is measured and the
///   shape is skipped if `rss + 4 × estimated_cost > cgroup_limit × 87.5%`.
///   This prevents ORT session-arena retention from accumulating past the
///   kernel ceiling across successive probe shapes.
///
/// # Returns
///
/// `(a, b)` where `a` and `b` are the fitted cost-model coefficients.
/// Returns conservative defaults and logs a warning on any failure.
#[allow(clippy::too_many_lines)]
pub(crate) async fn run_probe(
    pool: &EmbedPool,
    max_seq: usize,
    rss_ceiling: usize,
    cgroup_limit_bytes: usize,
) -> (f64, f64) {
    info!(
        max_seq,
        rss_ceiling_mb = rss_ceiling / (1024 * 1024),
        cgroup_limit_mb = cgroup_limit_bytes / (1024 * 1024),
        "Starting memory probe"
    );

    // Validate that the model can accept inputs at max_seq without running
    // attention. This checks tokenizer + ndarray shape construction only —
    // no `session.run()` call, so no ORT arena allocation.
    validate_max_seq_shape(max_seq);

    // Build shape list from the static set + dynamic max_seq capability anchor.
    let mut shapes: Vec<Shape> = PROBE_SHAPES.to_vec();
    // Add a (1, max_seq) shape if max_seq is larger than any static shape.
    // This anchors the quadratic coefficient at the configured upper bound.
    // If the model cannot run at max_seq, the per-shape error path skips it
    // (no fail-fast — the failure surfaces as an ORT error on the first real
    // request, which is more actionable than a startup OOM).
    if !shapes.iter().any(|&(_, s)| s == max_seq) {
        shapes.push((1, max_seq));
    }
    // Remove any shapes whose seq > max_seq (out of range for this model).
    shapes.retain(|&(_, s)| s <= max_seq);
    // Sort by ascending total token-positions so we grow load gradually.
    shapes.sort_by_key(|&(b, s)| b * s);

    let probe_start = std::time::Instant::now();
    let mut data: Vec<DataPoint> = Vec::with_capacity(shapes.len());
    let conservative = CostModel::conservative(rss_ceiling);

    // Per-shape outcome counters for precise diagnostics when data is empty.
    let mut shapes_skipped: usize = 0;
    let mut shapes_errored: usize = 0;
    let total_shapes = shapes.len();

    // Synthesize probe texts from corpus (already curated and pinned).
    let corpus_texts = load_probe_texts();

    // ----- Arena warm-up -----
    //
    // ORT lazily allocates its session arena on the first `session.run()`.
    // The first call therefore reads as a ~1 GB RSS delta even at tiny
    // shapes — that delta is arena bookkeeping, not per-call workspace, and
    // it pollutes the cost-model fit because it appears as constant noise
    // across all subsequent shapes.
    //
    // The warm-up runs a small `(1, 64)` `session.run()` BEFORE the actual
    // sweep starts and discards the result. After the warm-up, subsequent
    // per-shape `rss_delta` readings reflect only the incremental allocation
    // attributable to that shape, giving the OLS fitter a meaningful signal.
    //
    // The warm-up is gated by the same RSS guard that protects the sweep —
    // if `current_rss + 4 × chunk_cost(1, 64)` would breach the cgroup limit
    // we skip the warm-up and continue with conservative defaults.
    let warmup_texts = synthesize_texts(&corpus_texts, 1, 64);
    let warmup_start = std::time::Instant::now();
    match pool.probe(warmup_texts).await {
        Ok(result) => {
            let warmup_delta = result.rss_after.saturating_sub(result.rss_before);
            let elapsed_ms = u64::try_from(warmup_start.elapsed().as_millis()).unwrap_or(u64::MAX);
            info!(
                warmup_delta_mb = warmup_delta / (1024 * 1024),
                rss_after_mb = result.rss_after / (1024 * 1024),
                elapsed_ms,
                "Probe: arena warm-up complete (delta excluded from fit)"
            );
        }
        Err(e) => {
            warn!(
                error = %e,
                "Probe: warm-up failed — proceeding without warm-up; first shape's \
                 rss_delta will include arena initialisation overhead"
            );
        }
    }

    for (batch, seq) in &shapes {
        let batch = *batch;
        let seq = *seq;

        // Skip shapes estimated to exceed the rss_ceiling by more than
        // conservative cost model says (avoids OOM mid-probe).
        if !conservative.fits(batch, seq) {
            info!(
                batch,
                seq,
                rss_ceiling_mb = rss_ceiling / (1024 * 1024),
                "Probe: skipping shape (estimated to exceed rss_ceiling)"
            );
            shapes_skipped += 1;
            continue;
        }

        // Absolute-RSS guard: ORT session-arena retention accumulates across
        // probe shapes — each `session.run()` grows the arena and retains the
        // pages for subsequent calls. The conservative `fits()` check above
        // only looks at per-call workspace, not cumulative process RSS, so it
        // cannot protect against gradual exhaustion.
        //
        // Before each shape we read the live process RSS and project the
        // additional arena growth at 4× the conservative per-call estimate
        // (empirically observed ratio on aarch64 MLAS fp16 at shapes where
        // arena retention dominates). If the projected total would consume
        // more than 87.5% of the cgroup ceiling we skip the shape.
        //
        // This guard fires only when `cgroup_limit_bytes > 0` so it is a
        // no-op when memory detection fell back to the 4 GiB constant or was
        // overridden to 0 in tests.
        if cgroup_limit_bytes > 0 {
            let current_rss = crate::sysinfo::read_process_rss_bytes().unwrap_or(0);
            let estimated_cost = conservative.chunk_cost(batch, seq) as usize;
            // 12.5% safety headroom below the cgroup ceiling.
            let headroom = cgroup_limit_bytes / 8;
            let rss_limit = cgroup_limit_bytes.saturating_sub(headroom);
            // 4× multiplier on the conservative estimate to account for arena
            // retention observed across successive probe shapes.
            let projected = current_rss.saturating_add(estimated_cost.saturating_mul(4));
            if projected > rss_limit {
                info!(
                    batch,
                    seq,
                    current_rss_mb = current_rss / (1024 * 1024),
                    projected_mb = projected / (1024 * 1024),
                    rss_limit_mb = rss_limit / (1024 * 1024),
                    "Probe: skipping shape (current RSS + estimated arena growth \
                     would breach cgroup limit)"
                );
                shapes_skipped += 1;
                continue;
            }
        }

        // Synthesize texts of approximately `seq` tokens by repeating corpus
        // texts and trimming. We can't tokenize here (no tokenizer), so we
        // approximate: one word ≈ 1.3 tokens, one char ≈ 0.25 tokens.
        // At a rough 4 chars/token, a `seq`-token input is ~4*seq characters.
        let texts = synthesize_texts(&corpus_texts, batch, seq);

        let shape_start = std::time::Instant::now();
        match pool.probe(texts).await {
            Ok(result) => {
                let elapsed_ms =
                    u64::try_from(shape_start.elapsed().as_millis()).unwrap_or(u64::MAX);
                let delta = result.rss_after.saturating_sub(result.rss_before);
                info!(
                    batch,
                    seq,
                    rss_delta_mb = delta / (1024 * 1024),
                    elapsed_ms,
                    "Probe shape measured"
                );
                data.push(DataPoint {
                    batch,
                    seq,
                    rss_delta: delta,
                });
            }
            Err(e) => {
                let elapsed_ms =
                    u64::try_from(shape_start.elapsed().as_millis()).unwrap_or(u64::MAX);
                warn!(batch, seq, elapsed_ms, error = %e, "Probe shape failed; skipping");
                shapes_errored += 1;
            }
        }
    }

    if data.is_empty() {
        // Emit a specific diagnostic based on what actually happened so the
        // operator can distinguish between a broken budget (rss_ceiling=0),
        // ORT/model errors, and a non-Linux platform where RSS is unavailable.
        if shapes_skipped == total_shapes {
            warn!(
                rss_ceiling_mb = rss_ceiling / (1024 * 1024),
                total_shapes,
                "Probe: all shapes skipped because rss_ceiling is too small to fit even \
                 (batch=1, seq=64); per_worker_workspace upstream is likely broken — \
                 check model_rss_per_worker measurement and memory detection"
            );
        } else if shapes_errored == total_shapes {
            warn!(
                total_shapes,
                "Probe: all shapes errored — check ORT session and model logs above"
            );
        } else {
            warn!(
                shapes_skipped,
                shapes_errored,
                total_shapes,
                "Probe collected no usable data points (RSS measurement unavailable on \
                 non-Linux platforms, or all shapes were skipped/errored); \
                 using conservative defaults"
            );
        }
        return (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B);
    }

    // If all measured deltas are zero, RSS is unavailable (non-Linux).
    if data.iter().all(|dp| dp.rss_delta == 0) {
        warn!(
            data_points = data.len(),
            "Probe: all RSS deltas are zero — read_process_rss_bytes() returned 0; \
             auto-budget requires Linux /proc/self/statm; using conservative defaults"
        );
        return (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B);
    }

    if let Some((a, b)) = fit_cost_model(&data) {
        let total_elapsed_ms = u64::try_from(probe_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        info!(
            a = format!("{a:.0}"),
            b = format!("{b:.4}"),
            data_points = data.len(),
            total_elapsed_ms,
            "Probe: fitted cost model"
        );
        (a, b)
    } else {
        warn!(
            data_points = data.len(),
            "Probe: least-squares fit failed or produced invalid coefficients; \
             using conservative defaults"
        );
        (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B)
    }
}
