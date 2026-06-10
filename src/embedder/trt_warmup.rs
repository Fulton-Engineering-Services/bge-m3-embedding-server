// Copyright (c) 2026 J. Patrick Fulton
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `TensorRT` engine pre-warming: compiles and caches engine files during
//! worker startup so the first real request hits a cached engine instead
//! of triggering an on-demand 30–170 s compile.
//!
//! ## Durability
//!
//! After each shape compiles, the engine cache directory is fsynced so an
//! unexpected SIGKILL (ECS OOM-kill, host crash) cannot strand a
//! partially-written engine plan in the page cache.  See `trt_cache.rs`.
//!
//! ## Cache-hit fast path (warm cache skip)
//!
//! ORT's TRT EP caches engines with per-dimension `[min, max]` ranges — not
//! one engine per shape.  A `session.run()` is a cache **hit** (fast, no
//! compile) when every input dimension falls within the cached `[min, max]`
//! range; it is a cache **miss** (slow compile) only when a dimension falls
//! outside that range and the engine must be rebuilt with an extended range.
//!
//! After a full first-deploy warmup sweep, the cached profile records:
//! `input_ids.dim_0 ∈ [min_batch, max_batch]` and
//! `input_ids.dim_1 ∈ [min_seq, max_seq]` — covering every shape in the
//! warmup grid.  On subsequent container starts, every warmup
//! `session.run()` is a cache hit and finishes in ≤ 3 s.
//!
//! Rather than paying 24 × 1–3 s = 24–72 s of redundant cache-hit loads,
//! `trt_prewarm` runs at most **4 "dimensional-extreme" shapes** (the shapes
//! that exercise the minimum and maximum of each input dimension
//! independently) and measures wall-clock time.  If all extremes complete
//! under [`CACHE_HIT_THRESHOLD_MS`], the profile is guaranteed to cover the
//! entire shard and the remaining shapes are skipped.
//!
//! ### Why this has zero false positives
//!
//! For shape `(b, s)` to be a TRT cache hit it must satisfy:
//! ```text
//! profile.min_batch ≤ b ≤ profile.max_batch   (batch dimension)
//! profile.min_seq   ≤ s ≤ profile.max_seq      (sequence dimension)
//! ```
//!
//! The four extreme shapes bound all four inequalities independently:
//!
//! | Check shape          | Fact established when it is a cache hit  |
//! |----------------------|------------------------------------------|
//! | `(min_batch, any_s)` | `profile.min_batch ≤ min_batch`          |
//! | `(max_batch, any_s)` | `profile.max_batch ≥ max_batch`          |
//! | `(any_b, min_seq)`   | `profile.min_seq   ≤ min_seq`            |
//! | `(any_b, max_seq)`   | `profile.max_seq   ≥ max_seq`            |
//!
//! Together these four facts guarantee that every shard shape `(b, s)` with
//! `b ∈ [min_batch, max_batch]` and `s ∈ [min_seq, max_seq]` is a cache
//! hit.  If any extreme shape is **slow** (≥ `CACHE_HIT_THRESHOLD_MS`) the
//! engine must be rebuilt for that dimension → the fast path is suppressed
//! and all remaining shapes are compiled normally.

use std::path::Path;

use super::trt_cache;

mod postcondition;
mod runner;
#[cfg(test)]
mod tests;

pub(super) use postcondition::{
    prewarm_persistence_postcondition_failed, prewarm_persistence_suspicious_undercount,
};
// Constants are re-exported only for the sibling `tests` module; production
// callers reach them transitively through the postcondition predicates above.
#[cfg(test)]
pub(super) use postcondition::SUSPICIOUS_UNDERCOUNT_MIN_FRESH;
use runner::run_warmup_shape;

/// Threshold (ms) below which a `session.run()` is classified as a TRT
/// engine cache hit (loaded from disk) rather than a fresh compile.
///
/// Cold compiles take 30 000–170 000 ms; warm cache loads finish in
/// ≤ 3 000 ms even for `32 × 8192` shapes.  A 5 000 ms threshold gives a
/// comfortable margin above the observed maximum warm-load time while
/// remaining well below the minimum observed cold-compile time.
pub(super) const CACHE_HIT_THRESHOLD_MS: u64 = 5_000;

/// Aggregate per-worker statistics returned by [`trt_prewarm`].
///
/// `total_compile_ms` and `total_fsync_ms` sum across only the shapes that
/// completed successfully on this worker's shard, whether they were cache
/// hits or fresh compiles.  They are intended for the `"TensorRT pre-warm
/// complete"` summary log emitted by the worker.
pub(super) struct PrewarmStats {
    pub warmed: usize,
    pub total_compile_ms: u64,
    pub total_fsync_ms: u64,
    /// `true` when the dimensional-extreme coverage check determined the
    /// entire shard was already cached and the remaining shapes were skipped.
    /// `false` on cold cache, on a fresh compile, or when the check phase
    /// detected at least one slow (≥ `CACHE_HIT_THRESHOLD_MS`) shape.
    pub fully_cached: bool,
    /// Number of shapes in the shard that were skipped because
    /// `fully_cached` was determined to be true.  Zero on cold cache or
    /// when all shapes were run.
    pub skipped: usize,
    /// Number of shapes that reported a fresh compile (`!cache_hit` and
    /// `Ok(_)` from `session.run()`).  Used together with `engine_count_delta`
    /// by the worker to decide whether the on-disk artifacts match what the
    /// per-shape logs claimed.
    pub fresh_compiles: usize,
    /// Net `.engine` file count change across this worker's prewarm sweep
    /// (`count_after_last_shape - count_before_first_shape`). SM-filtered:
    /// reflects only plans matching the worker's GPU compute capability,
    /// so a stale `_sm89.engine` next to a fresh `_sm120.engine` does not
    /// silently zero out the delta on a Blackwell worker. Compared against
    /// `fresh_compiles` to detect compile-success-without-persistence.
    pub engine_count_delta: i64,
    /// `.engine` file count observed in `engine_cache_dir` before the worker
    /// ran any of its shard's shapes. SM-filtered: counts only plans matching
    /// the worker's GPU compute capability (see the `sm` parameter on
    /// [`trt_prewarm`]). When SM detection failed and `sm == None`, falls
    /// back to the legacy unfiltered count.
    pub engine_count_before: usize,
    /// `.engine` file count observed in `engine_cache_dir` after the worker
    /// finished its shard (post final fsync). SM-filtered with the same
    /// semantics as `engine_count_before`.
    pub engine_count_after: usize,
    /// Largest sequence length among the shapes this worker successfully
    /// warmed (fresh compile **or** warm-cache hit). Zero when no shape
    /// succeeded (e.g. every compile failed, the worker-3 `seq=8192` failure
    /// mode). Folded into the pool-wide `warmed_seq_ceiling` atomic by the
    /// worker so [`super::jit_guard::TrtJitGuard`] knows the highest sequence
    /// tier with a persisted engine plan. See `worker.rs`.
    pub max_warmed_seq: usize,
}

/// Selects the minimal set of shapes needed to verify that an ORT TRT EP
/// cached profile covers all shapes in `shapes`.
///
/// ORT's TRT EP stores engine profiles as per-dimension `[min, max]` ranges.
/// Verifying complete coverage requires bounding all four dimension extremes
/// independently.  This function returns at most 4 representative shapes —
/// one with `min_batch`, one with `max_batch`, one with `min_seq`, one with
/// `max_seq` — deduplicated so the same shape is never run twice.
///
/// When the shard has only one unique extreme in a dimension (e.g. all
/// shapes share the same batch size), the duplicates collapse and the
/// returned set is smaller.
pub(super) fn coverage_check_shapes(shapes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if shapes.is_empty() {
        return vec![];
    }
    let min_batch = shapes.iter().map(|(b, _)| *b).min().expect("non-empty");
    let max_batch = shapes.iter().map(|(b, _)| *b).max().expect("non-empty");
    let min_seq = shapes.iter().map(|(_, s)| *s).min().expect("non-empty");
    let max_seq = shapes.iter().map(|(_, s)| *s).max().expect("non-empty");

    // Pick the first shape in the shard that carries each dimensional extreme.
    // "First" is stable (same shape list order across workers on the same host)
    // so logs are reproducible.
    let rep_min_batch = *shapes
        .iter()
        .find(|(b, _)| *b == min_batch)
        .expect("non-empty");
    let rep_max_batch = *shapes
        .iter()
        .find(|(b, _)| *b == max_batch)
        .expect("non-empty");
    let rep_min_seq = *shapes
        .iter()
        .find(|(_, s)| *s == min_seq)
        .expect("non-empty");
    let rep_max_seq = *shapes
        .iter()
        .find(|(_, s)| *s == max_seq)
        .expect("non-empty");

    // Deduplicate while preserving the discovery order (min_batch → max_batch
    // → min_seq → max_seq) so the log is consistent across runs.
    let mut result: Vec<(usize, usize)> = Vec::with_capacity(4);
    for s in [rep_min_batch, rep_max_batch, rep_min_seq, rep_max_seq] {
        if !result.contains(&s) {
            result.push(s);
        }
    }
    result
}

/// Partitions `shapes` into a per-worker shard using a stride assignment.
///
/// Worker `worker_index` receives shapes at positions
/// `worker_index, worker_index + worker_count, worker_index + 2*worker_count, …`
/// in the input slice order.
///
/// **Why stride and not contiguous blocks?**\
/// The default warmup grid is ordered batch-major:
/// `{1,2,4,8,16,32} × {128,512,2048,8192}`. Each consecutive group of four shapes
/// belongs to one batch size, and within a group the sequence length grows
/// monotonically. Stride assignment therefore spreads the work so each GPU
/// receives one shape from each batch group at a different sequence length.
/// The expensive `_×8192` shapes land on different workers than each other
/// (e.g. with 4 workers, worker 3 gets all 8192-seq shapes, which compile in
/// parallel with the cheaper shapes on workers 0–2). Total wall-clock time is
/// approximately the serial compile time for worker 3's four shapes, compared
/// to the serial time for all 24 - a rough 4× speedup on 4 GPUs.
///
/// Returns all shapes unchanged when `worker_count ≤ 1`.
pub(super) fn shard_shapes(
    shapes: &[(usize, usize)],
    worker_index: usize,
    worker_count: usize,
) -> Vec<(usize, usize)> {
    if worker_count <= 1 {
        return shapes.to_vec();
    }
    shapes
        .iter()
        .enumerate()
        .filter(|(i, _)| i % worker_count == worker_index)
        .map(|(_, &s)| s)
        .collect()
}

/// Runs a dummy `session.run()` for each `(batch, seq)` shape in
/// `warmup_shapes` so the `TensorRT` EP compiles and caches engine files
/// before the first real request arrives.
///
/// ## SM-aware cache accounting
///
/// `sm` selects which engine plans count toward `engine_count_before`,
/// `engine_count_after`, the coverage-check fast-path trigger, and the
/// per-shape persistence WARN. Pass `Some("smXY")` (e.g. `Some("sm120")`
/// for Blackwell) so a heterogeneous cache containing plans for other GPU
/// compute capabilities — typical when a fleet is mid-deploy or an EFS
/// volume was previously used by a different instance family — never
/// produces a false `cache_hit:true` signal. Pass `None` for the legacy
/// unfiltered behaviour (only when SM detection failed; see the WARN
/// emitted in `run_worker`).
///
/// ## Warm-cache fast path
///
/// When `.engine` files **matching `sm`** already exist in the cache
/// directory, the function first runs only the dimensional-extreme shapes
/// (≤ 4) to probe whether the cached profile covers the full shard. If all
/// extreme shapes complete in under [`CACHE_HIT_THRESHOLD_MS`] the
/// remaining shapes are **skipped** — they are guaranteed to be cache hits
/// by the range-based ORT TRT EP profile logic (see module-level
/// documentation for the proof). If any extreme shape is slow the fast path
/// is suppressed and all remaining shapes are compiled normally.
///
/// ## Cold cache
///
/// When no `.engine` files matching `sm` exist the coverage-check phase is
/// bypassed and every shape is compiled in sequence.  Each may take
/// 30–170 s on the very first deploy; subsequent starts reuse the cached
/// `.engine` files for this SM.
///
/// Progress is logged at `INFO` with `compile_ms`, `fsync_ms`, and
/// `cache_hit` (whether the run was under `CACHE_HIT_THRESHOLD_MS`) for
/// each shape.  After each successful run the engine cache directory is
/// fsynced so the plan file survives an unexpected OOM-kill — see
/// `trt_cache::fsync_cache_dir`.
///
/// Returns aggregate statistics including `fully_cached` (whether the
/// shard was served entirely from cache **for this SM**) and `skipped`
/// (shapes not run).
#[allow(clippy::too_many_lines)]
pub(super) fn trt_prewarm(
    session: &mut ort::session::Session,
    warmup_shapes: &[(usize, usize)],
    worker_id: usize,
    cache_dir: &Path,
    sm: Option<&str>,
) -> PrewarmStats {
    let engine_cache_dir = trt_cache::engine_cache_path(cache_dir);

    let mut warmed = 0usize;
    let mut fresh_compiles = 0usize;
    let mut total_compile_ms: u64 = 0;
    let mut total_fsync_ms: u64 = 0;
    let mut max_warmed_seq = 0usize;
    let shape_total = warmup_shapes.len();
    let engine_count_before = trt_cache::count_engine_files_for_sm(&engine_cache_dir, sm);

    if warmup_shapes.is_empty() {
        return PrewarmStats {
            warmed: 0,
            total_compile_ms: 0,
            total_fsync_ms: 0,
            fully_cached: false,
            skipped: 0,
            fresh_compiles: 0,
            engine_count_delta: 0,
            engine_count_before,
            engine_count_after: engine_count_before,
            max_warmed_seq: 0,
        };
    }

    // ── Coverage-check fast path ──────────────────────────────────────────
    // If any engines already exist on disk, run only the dimensional-extreme
    // shapes to determine whether the full shard is already cached.
    let check_shapes = if engine_count_before > 0 {
        coverage_check_shapes(warmup_shapes)
    } else {
        Vec::new()
    };

    // Run check shapes (at most 4).
    let mut shape_idx = 0usize;
    let mut all_checks_fast = !check_shapes.is_empty(); // false when check_shapes is empty
    for &(batch, seq) in &check_shapes {
        shape_idx += 1;
        let r = run_warmup_shape(
            session,
            batch,
            seq,
            worker_id,
            shape_idx,
            shape_total,
            &engine_cache_dir,
            sm,
        );
        if r.succeeded {
            warmed += 1;
            max_warmed_seq = max_warmed_seq.max(seq);
            total_compile_ms = total_compile_ms.saturating_add(r.compile_ms);
            total_fsync_ms = total_fsync_ms.saturating_add(r.fsync_ms);
            if !r.cache_hit {
                fresh_compiles += 1;
            }
        }
        if !r.cache_hit || !r.succeeded {
            all_checks_fast = false;
        }
    }

    if all_checks_fast {
        // Every dimensional extreme was a sub-threshold cache hit: the ORT TRT
        // EP's stored profile covers the full shard range.  Skip remaining shapes.
        let skipped = shape_total.saturating_sub(check_shapes.len());
        tracing::info!(
            worker_id,
            checked = check_shapes.len(),
            skipped,
            total = shape_total,
            detected_sm = sm.unwrap_or("unfiltered"),
            cache_hit_threshold_ms = CACHE_HIT_THRESHOLD_MS,
            "TensorRT pre-warm: shard fully cached \
             (all dimensional-extreme checks fast), skipping remaining shapes"
        );
        // One final fsync covers sidecar files (timing cache, `.profile`)
        // that may have been touched during the check phase.
        trt_cache::fsync_cache_dir(&engine_cache_dir);
        let engine_count_after = trt_cache::count_engine_files_for_sm(&engine_cache_dir, sm);
        return PrewarmStats {
            warmed,
            total_compile_ms,
            total_fsync_ms,
            fully_cached: true,
            skipped,
            fresh_compiles,
            engine_count_delta: i64::try_from(engine_count_after).unwrap_or(i64::MAX)
                - i64::try_from(engine_count_before).unwrap_or(i64::MAX),
            engine_count_before,
            engine_count_after,
            // The dimensional-extreme checks include the shard's max-seq shape;
            // a fully-cached shard therefore has coverage up to its max seq.
            max_warmed_seq,
        };
    }

    // ── Full compile path ─────────────────────────────────────────────────
    // Cold cache OR at least one extreme was slow → run all shapes not
    // already executed in the check phase.
    for &(batch, seq) in warmup_shapes {
        // Skip shapes already run as coverage checks (avoid double-running).
        if check_shapes.contains(&(batch, seq)) {
            continue;
        }
        shape_idx += 1;
        let r = run_warmup_shape(
            session,
            batch,
            seq,
            worker_id,
            shape_idx,
            shape_total,
            &engine_cache_dir,
            sm,
        );
        if r.succeeded {
            warmed += 1;
            max_warmed_seq = max_warmed_seq.max(seq);
            total_compile_ms = total_compile_ms.saturating_add(r.compile_ms);
            total_fsync_ms = total_fsync_ms.saturating_add(r.fsync_ms);
            if !r.cache_hit {
                fresh_compiles += 1;
            }
        }
    }

    // Final sweep covers any sidecar files (timing cache, `.profile`) that
    // were touched during the warmup but not associated with a single shape.
    trt_cache::fsync_cache_dir(&engine_cache_dir);

    let engine_count_after = trt_cache::count_engine_files_for_sm(&engine_cache_dir, sm);
    PrewarmStats {
        warmed,
        total_compile_ms,
        total_fsync_ms,
        fully_cached: false,
        skipped: 0,
        fresh_compiles,
        engine_count_delta: i64::try_from(engine_count_after).unwrap_or(i64::MAX)
            - i64::try_from(engine_count_before).unwrap_or(i64::MAX),
        engine_count_before,
        engine_count_after,
        max_warmed_seq,
    }
}
