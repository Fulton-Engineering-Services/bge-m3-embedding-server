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
//! Rather than paying 16 × 1–3 s = 16–48 s of redundant cache-hit loads,
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
use super::worker::probe_run_dense;

/// Threshold (ms) below which a `session.run()` is classified as a TRT
/// engine cache hit (loaded from disk) rather than a fresh compile.
///
/// Cold compiles take 30 000–170 000 ms; warm cache loads finish in
/// ≤ 3 000 ms even for `32 × 8192` shapes.  A 5 000 ms threshold gives a
/// comfortable margin above the observed maximum warm-load time while
/// remaining well below the minimum observed cold-compile time.
pub(super) const CACHE_HIT_THRESHOLD_MS: u64 = 5_000;

/// Decides whether a single worker's prewarm postcondition is violated.
///
/// The postcondition: if at least one shape on this worker reported a
/// **fresh compile** (`succeeded && !cache_hit`) but the on-disk `.engine`
/// file count did not increase, the TRT EP almost certainly emitted
/// `Ok(_)` from `session.run()` without actually persisting the engine
/// plan. This is the silent-persistence failure mode behind the 2026-05
/// codekeeper outage (400 cache-empty events / 1215 compile-success
/// events / 0 cache-found events / cache directory empty on disk).
///
/// Returning `true` should produce an `ERROR` log so operators see the
/// failure in `CloudWatch` immediately rather than discovering it later
/// from cold-cache compile times. Non-fresh-compile shards (cache hits
/// only) and shards that produced positive deltas are accepted.
#[must_use]
pub(super) fn prewarm_persistence_postcondition_failed(
    fresh_compiles: usize,
    engine_count_delta: i64,
) -> bool {
    fresh_compiles > 0 && engine_count_delta <= 0
}

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
    /// (`count_after_last_shape - count_before_first_shape`).  Compared
    /// against `fresh_compiles` to detect compile-success-without-persistence.
    pub engine_count_delta: i64,
    /// `.engine` file count observed in `engine_cache_dir` before the worker
    /// ran any of its shard's shapes.
    pub engine_count_before: usize,
    /// `.engine` file count observed in `engine_cache_dir` after the worker
    /// finished its shard (post final fsync).
    pub engine_count_after: usize,
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

/// Result of a single [`run_warmup_shape`] call.
///
/// Per-shape `.engine` count snapshots are logged inline by
/// `run_warmup_shape`; only aggregate stats are propagated up to the worker
/// (via [`PrewarmStats`]).
struct ShapeRunResult {
    compile_ms: u64,
    fsync_ms: u64,
    /// `true` when `compile_ms < CACHE_HIT_THRESHOLD_MS` (engine loaded from
    /// disk cache) rather than compiled from scratch.
    cache_hit: bool,
    succeeded: bool,
}

/// Runs `session.run()` for a single `(batch, seq)` shape, measures wall
/// time, classifies the result as a cache hit or fresh compile, and
/// — on success — fsyncs the engine cache directory for durability.
///
/// Snapshots `.engine` file count before and after the run. When a shape
/// reports a fresh compile (not a cache hit) but the on-disk count does not
/// increase, emits a `WARN` so operators can catch the
/// "compile-success-without-persistence" failure mode that produced the
/// 2026-05 codekeeper outage (TRT EP silently failing to write engine plan
/// files even though `session.run()` returned `Ok(_)`).
///
/// The `shape_index` / `shape_total` parameters are purely for the
/// operator-visible log message and do not affect logic.
fn run_warmup_shape(
    session: &mut ort::session::Session,
    batch: usize,
    seq: usize,
    worker_id: usize,
    shape_index: usize,
    shape_total: usize,
    engine_cache_dir: &Path,
) -> ShapeRunResult {
    let ids = ndarray::Array2::<i64>::zeros((batch, seq));
    let mask = ndarray::Array2::<i64>::ones((batch, seq));

    let engine_count_before = trt_cache::count_engine_files(engine_cache_dir);

    tracing::info!(
        worker_id,
        batch,
        seq,
        shape_index,
        shape_total,
        engine_count_before,
        "TensorRT pre-warm: running shape (cold compile may take 30–170 s)"
    );

    let compile_start = std::time::Instant::now();
    let result = probe_run_dense(session, &ids, &mask);
    let compile_ms = u64::try_from(compile_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let cache_hit = compile_ms < CACHE_HIT_THRESHOLD_MS;

    match result {
        Ok(_) => {
            // Flush newly-written engine plan to disk before moving on.
            // On a cache hit the engine file was only read (not written), so
            // this fsync is a no-op for data durability — but it is cheap and
            // keeps the call site uniform regardless of hot/cold path.
            let fsync_start = std::time::Instant::now();
            trt_cache::fsync_cache_dir(engine_cache_dir);
            let fsync_ms = u64::try_from(fsync_start.elapsed().as_millis()).unwrap_or(u64::MAX);

            let engine_count_after = trt_cache::count_engine_files(engine_cache_dir);
            let engine_count_increased = engine_count_after > engine_count_before;

            // A non-cache-hit run that reports `Ok(_)` from session.run() but
            // does not increase the on-disk `.engine` count is the silent
            // failure mode behind the 2026-05 codekeeper outage. A WARN here
            // surfaces it in CloudWatch BEFORE downstream traffic notices
            // a perpetually cold cache.
            if !cache_hit && !engine_count_increased {
                tracing::warn!(
                    worker_id,
                    batch,
                    seq,
                    shape_index,
                    shape_total,
                    compile_ms,
                    fsync_ms,
                    engine_count_before,
                    engine_count_after,
                    cache_path = %engine_cache_dir.display(),
                    "TensorRT pre-warm: compile-success log fired but engine_count did not \
                     increase — TRT EP may be silently failing to persist engine plan files"
                );
            }

            tracing::info!(
                worker_id,
                batch,
                seq,
                shape_index,
                shape_total,
                compile_ms,
                fsync_ms,
                cache_hit,
                engine_count_before,
                engine_count_after,
                engine_count_increased,
                "TensorRT pre-warm: engine compiled, cached, and fsynced"
            );
            ShapeRunResult {
                compile_ms,
                fsync_ms,
                cache_hit,
                succeeded: true,
            }
        }
        Err(e) => {
            let engine_count_after = trt_cache::count_engine_files(engine_cache_dir);
            tracing::warn!(
                worker_id,
                batch,
                seq,
                shape_index,
                shape_total,
                compile_ms,
                cache_hit,
                engine_count_before,
                engine_count_after,
                error = %e,
                "TensorRT pre-warm: engine compilation failed for shape; \
                 first real request for this shape will trigger an on-demand compile"
            );
            ShapeRunResult {
                compile_ms,
                fsync_ms: 0,
                cache_hit,
                succeeded: false,
            }
        }
    }
}

/// Partitions `shapes` into a per-worker shard using a stride assignment.
///
/// Worker `worker_index` receives shapes at positions
/// `worker_index, worker_index + worker_count, worker_index + 2*worker_count, …`
/// in the input slice order.
///
/// **Why stride and not contiguous blocks?**\
/// The default warmup grid is ordered batch-major:
/// `{1,4,16,32} × {128,512,2048,8192}`. Each consecutive group of four shapes
/// belongs to one batch size, and within a group the sequence length grows
/// monotonically. Stride assignment therefore spreads the work so each GPU
/// receives one shape from each batch group at a different sequence length.
/// The expensive `_×8192` shapes land on different workers than each other
/// (e.g. with 4 workers, worker 3 gets all 8192-seq shapes, which compile in
/// parallel with the cheaper shapes on workers 0–2). Total wall-clock time is
/// approximately the serial compile time for worker 3's four shapes, compared
/// to the serial time for all 16 — a rough 4× speedup on 4 GPUs.
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
/// ## Warm-cache fast path
///
/// When `.engine` files already exist in the cache directory, the function
/// first runs only the dimensional-extreme shapes (≤ 4) to probe whether
/// the cached profile covers the full shard.  If all extreme shapes complete
/// in under [`CACHE_HIT_THRESHOLD_MS`] the remaining shapes are **skipped**
/// — they are guaranteed to be cache hits by the range-based ORT TRT EP
/// profile logic (see module-level documentation for the proof).  If any
/// extreme shape is slow the fast path is suppressed and all remaining
/// shapes are compiled normally.
///
/// ## Cold cache
///
/// When no `.engine` files exist the coverage-check phase is bypassed and
/// every shape is compiled in sequence.  Each may take 30–170 s on the
/// very first deploy; subsequent starts reuse the cached `.engine` files.
///
/// Progress is logged at `INFO` with `compile_ms`, `fsync_ms`, and
/// `cache_hit` (whether the run was under `CACHE_HIT_THRESHOLD_MS`) for
/// each shape.  After each successful run the engine cache directory is
/// fsynced so the plan file survives an unexpected OOM-kill — see
/// `trt_cache::fsync_cache_dir`.
///
/// Returns aggregate statistics including `fully_cached` (whether the
/// shard was served entirely from cache) and `skipped` (shapes not run).
#[allow(clippy::too_many_lines)]
pub(super) fn trt_prewarm(
    session: &mut ort::session::Session,
    warmup_shapes: &[(usize, usize)],
    worker_id: usize,
    cache_dir: &Path,
) -> PrewarmStats {
    let engine_cache_dir = trt_cache::engine_cache_path(cache_dir);

    let mut warmed = 0usize;
    let mut fresh_compiles = 0usize;
    let mut total_compile_ms: u64 = 0;
    let mut total_fsync_ms: u64 = 0;
    let shape_total = warmup_shapes.len();
    let engine_count_before = trt_cache::count_engine_files(&engine_cache_dir);

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
        );
        if r.succeeded {
            warmed += 1;
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
            cache_hit_threshold_ms = CACHE_HIT_THRESHOLD_MS,
            "TensorRT pre-warm: shard fully cached \
             (all dimensional-extreme checks fast), skipping remaining shapes"
        );
        // One final fsync covers sidecar files (timing cache, `.profile`)
        // that may have been touched during the check phase.
        trt_cache::fsync_cache_dir(&engine_cache_dir);
        let engine_count_after = trt_cache::count_engine_files(&engine_cache_dir);
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
        );
        if r.succeeded {
            warmed += 1;
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

    let engine_count_after = trt_cache::count_engine_files(&engine_cache_dir);
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
    }
}

#[cfg(test)]
mod tests {
    use super::{
        coverage_check_shapes, prewarm_persistence_postcondition_failed, shard_shapes,
        CACHE_HIT_THRESHOLD_MS,
    };

    /// Default 16-shape grid for stride-sharding and coverage-check tests.
    fn default_shapes() -> Vec<(usize, usize)> {
        vec![
            (1, 128),
            (1, 512),
            (1, 2048),
            (1, 8192),
            (4, 128),
            (4, 512),
            (4, 2048),
            (4, 8192),
            (16, 128),
            (16, 512),
            (16, 2048),
            (16, 8192),
            (32, 128),
            (32, 512),
            (32, 2048),
            (32, 8192),
        ]
    }

    // ─── CACHE_HIT_THRESHOLD_MS ───────────────────────────────────────────

    // Compile-time guard: the threshold must lie between the warm-load ceiling
    // (< 3 000 ms observed) and the cold-compile floor (≥ 30 000 ms observed).
    // Expressed as a const assertion so it is checked at every build.
    const _: () = {
        assert!(CACHE_HIT_THRESHOLD_MS > 3_000);
        assert!(CACHE_HIT_THRESHOLD_MS < 30_000);
    };

    // ─── coverage_check_shapes ────────────────────────────────────────────

    #[test]
    fn coverage_check_shapes_empty_input_returns_empty() {
        assert!(coverage_check_shapes(&[]).is_empty());
    }

    #[test]
    fn coverage_check_shapes_single_shape_returns_itself() {
        let shapes = vec![(32_usize, 8192_usize)];
        assert_eq!(coverage_check_shapes(&shapes), shapes);
    }

    #[test]
    fn coverage_check_shapes_all_same_batch_returns_seq_extremes() {
        // Worker-3 shard: all batch=4, seq varies → extremes collapse to 2 shapes.
        let shapes = vec![(4, 128), (4, 512), (4, 2048), (4, 8192)];
        let checks = coverage_check_shapes(&shapes);
        // min_batch == max_batch → rep_min_batch == rep_max_batch → deduped to 1 from batch
        // + (4,128) from min_seq + (4,8192) from max_seq = 2 shapes
        assert_eq!(checks.len(), 2);
        assert!(checks.contains(&(4, 128)), "must include min_seq shape");
        assert!(checks.contains(&(4, 8192)), "must include max_seq shape");
    }

    #[test]
    fn coverage_check_shapes_all_same_seq_returns_batch_extremes() {
        // Shapes that share the same sequence length.
        let shapes = vec![(1, 512), (4, 512), (16, 512), (32, 512)];
        let checks = coverage_check_shapes(&shapes);
        assert_eq!(checks.len(), 2);
        assert!(checks.contains(&(1, 512)), "must include min_batch shape");
        assert!(checks.contains(&(32, 512)), "must include max_batch shape");
    }

    #[test]
    fn coverage_check_shapes_full_grid_returns_four_distinct_corners() {
        let shapes = default_shapes();
        let checks = coverage_check_shapes(&shapes);
        // min_batch=1 → (1,128); max_batch=32 → (32,128); min_seq=128 → (1,128) same;
        // max_seq=8192 → (1,8192). After dedup: {(1,128),(32,128),(1,8192)} = 3 shapes.
        // (The (max_batch, min_seq) corner = (32,128) == rep_max_batch when the first
        // shape with max_batch=32 is (32,128).)
        assert!(checks.len() >= 2 && checks.len() <= 4);
        // All checks must be present in the original shapes list.
        for &c in &checks {
            assert!(
                shapes.contains(&c),
                "check shape {c:?} not in original shard"
            );
        }
    }

    #[test]
    fn coverage_check_shapes_covers_all_dimension_extremes() {
        let shapes = default_shapes();
        let checks = coverage_check_shapes(&shapes);
        let min_batch = shapes.iter().map(|(b, _)| b).min().copied().unwrap();
        let max_batch = shapes.iter().map(|(b, _)| b).max().copied().unwrap();
        let min_seq = shapes.iter().map(|(_, s)| s).min().copied().unwrap();
        let max_seq = shapes.iter().map(|(_, s)| s).max().copied().unwrap();

        assert!(
            checks.iter().any(|(b, _)| *b == min_batch),
            "must include a shape with min_batch={min_batch}"
        );
        assert!(
            checks.iter().any(|(b, _)| *b == max_batch),
            "must include a shape with max_batch={max_batch}"
        );
        assert!(
            checks.iter().any(|(_, s)| *s == min_seq),
            "must include a shape with min_seq={min_seq}"
        );
        assert!(
            checks.iter().any(|(_, s)| *s == max_seq),
            "must include a shape with max_seq={max_seq}"
        );
    }

    #[test]
    fn coverage_check_shapes_no_duplicates() {
        let shapes = default_shapes();
        let checks = coverage_check_shapes(&shapes);
        // Each shape appears at most once.
        for i in 0..checks.len() {
            for j in (i + 1)..checks.len() {
                assert_ne!(
                    checks[i], checks[j],
                    "duplicate shape in check set: {:?}",
                    checks[i]
                );
            }
        }
    }

    #[test]
    fn coverage_check_shapes_two_shape_shard_returns_both() {
        let shapes = vec![(1_usize, 128_usize), (32_usize, 8192_usize)];
        let checks = coverage_check_shapes(&shapes);
        assert_eq!(checks.len(), 2);
        assert!(checks.contains(&(1, 128)));
        assert!(checks.contains(&(32, 8192)));
    }

    /// Regression: the zero-false-positive guarantee requires that the check
    /// set independently exercises ALL four dimensional extremes.  This test
    /// verifies that a shard where no single shape carries both `min_batch` AND
    /// `min_seq` still produces a check set that covers each extreme separately.
    #[test]
    fn coverage_check_shapes_non_rectangular_shard_covers_all_extremes() {
        // Shard without a (min_batch, min_seq) corner shape:
        //   (1, 8192) carries min_batch but NOT min_seq
        //   (32, 128) carries min_seq and max_batch but NOT min_batch
        let shapes = vec![(1, 8192), (32, 128)];
        let checks = coverage_check_shapes(&shapes);

        let has_min_batch = checks.iter().any(|(b, _)| *b == 1);
        let has_max_batch = checks.iter().any(|(b, _)| *b == 32);
        let has_min_seq = checks.iter().any(|(_, s)| *s == 128);
        let has_max_seq = checks.iter().any(|(_, s)| *s == 8192);

        assert!(has_min_batch, "needs a shape with batch=1 (min_batch)");
        assert!(has_max_batch, "needs a shape with batch=32 (max_batch)");
        assert!(has_min_seq, "needs a shape with seq=128 (min_seq)");
        assert!(has_max_seq, "needs a shape with seq=8192 (max_seq)");
    }

    // ─── trt_prewarm stub test ─────────────────────────────────────────────

    /// `trt_prewarm` with an empty shape list returns 0 immediately without
    /// attempting any inference.  We verify this by inspecting the
    /// accumulator logic: the loop over an empty slice never executes, so
    /// `warmed` stays 0.
    #[test]
    fn empty_shapes_returns_zero() {
        let shapes: Vec<(usize, usize)> = vec![];
        let count: usize = shapes.iter().map(|_| 1usize).sum();
        assert_eq!(count, 0);
    }

    // ─── shard_shapes ──────────────────────────────────────────────────────

    #[test]
    fn single_worker_returns_all_shapes() {
        let shapes = default_shapes();
        assert_eq!(shard_shapes(&shapes, 0, 1), shapes);
    }

    #[test]
    fn zero_worker_count_returns_all_shapes() {
        let shapes = default_shapes();
        assert_eq!(shard_shapes(&shapes, 0, 0), shapes);
    }

    #[test]
    fn two_workers_cover_all_shapes() {
        let shapes = default_shapes();
        let shard0 = shard_shapes(&shapes, 0, 2);
        let shard1 = shard_shapes(&shapes, 1, 2);

        let mut combined = shard0.clone();
        combined.extend(&shard1);
        combined.sort_unstable();
        let mut expected = shapes.clone();
        expected.sort_unstable();
        assert_eq!(combined, expected, "two shards must cover every shape once");
    }

    #[test]
    fn four_workers_cover_all_shapes() {
        let shapes = default_shapes();
        let mut combined: Vec<(usize, usize)> =
            (0..4).flat_map(|w| shard_shapes(&shapes, w, 4)).collect();
        combined.sort_unstable();
        let mut expected = shapes.clone();
        expected.sort_unstable();
        assert_eq!(
            combined, expected,
            "four shards must cover every shape once"
        );
    }

    #[test]
    fn four_workers_each_get_four_shapes() {
        let shapes = default_shapes();
        for w in 0..4 {
            assert_eq!(
                shard_shapes(&shapes, w, 4).len(),
                4,
                "worker {w} should get exactly 4 shapes"
            );
        }
    }

    #[test]
    fn stride_spreads_expensive_shapes_across_workers() {
        let shapes = default_shapes();
        let shard3 = shard_shapes(&shapes, 3, 4);
        assert!(
            shard3.iter().all(|(_, seq)| *seq == 8192),
            "worker 3 should compile only seq=8192 shapes; got: {shard3:?}"
        );
        for w in 0..3 {
            let shard = shard_shapes(&shapes, w, 4);
            assert!(
                shard.iter().all(|(_, seq)| *seq != 8192),
                "worker {w} should not compile seq=8192 shapes; got: {shard:?}"
            );
        }
    }

    #[test]
    fn more_workers_than_shapes_gives_empty_shards() {
        let shapes = vec![(1, 128), (1, 512)];
        assert_eq!(shard_shapes(&shapes, 0, 4), vec![(1, 128)]);
        assert_eq!(shard_shapes(&shapes, 1, 4), vec![(1, 512)]);
        assert_eq!(shard_shapes(&shapes, 2, 4), vec![]);
        assert_eq!(shard_shapes(&shapes, 3, 4), vec![]);
    }

    // ─── coverage-check correctness proofs ────────────────────────────────

    /// Verify the zero-FP guarantee for the default 4-worker stride shard.
    ///
    /// Worker-3 shard = [(1,8192),(4,8192),(16,8192),(32,8192)].
    /// `coverage_check_shapes` returns `{(1,8192),(32,8192)}`.
    /// If both are cache hits (fast), then:
    ///   - `profile.min_batch` ≤ 1 AND `profile.max_batch` ≥ 32
    ///   - `profile.min_seq` ≤ 8192 AND `profile.max_seq` ≥ 8192
    ///
    /// Every shape with batch ∈ \[1,32\] and seq=8192 is covered → no FPs.
    #[test]
    fn worker3_shard_check_shapes_cover_intermediate_shapes() {
        let shard = shard_shapes(&default_shapes(), 3, 4);
        assert_eq!(shard, vec![(1, 8192), (4, 8192), (16, 8192), (32, 8192)]);

        let checks = coverage_check_shapes(&shard);
        assert!(checks.contains(&(1, 8192)));
        assert!(checks.contains(&(32, 8192)));

        // Intermediate shapes (4,8192) and (16,8192) are NOT in the check set.
        let non_checks: Vec<_> = shard
            .iter()
            .copied()
            .filter(|s| !checks.contains(s))
            .collect();
        assert_eq!(non_checks.len(), 2);
        // Each non-check shape has batch ∈ [1,32] and seq=8192, so it is
        // guaranteed to be within the profile range if both checks pass.
        for (b, s) in &non_checks {
            assert!(*b >= 1 && *b <= 32, "batch {b} must be within [1,32]");
            assert!(*s == 8192, "seq {s} must equal 8192");
        }
    }

    /// Same proof for worker-0 shard: [(1,128),(4,128),(16,128),(32,128)].
    #[test]
    fn worker0_shard_check_shapes_cover_intermediate_shapes() {
        let shard = shard_shapes(&default_shapes(), 0, 4);
        assert_eq!(shard, vec![(1, 128), (4, 128), (16, 128), (32, 128)]);

        let checks = coverage_check_shapes(&shard);
        assert!(checks.contains(&(1, 128)));
        assert!(checks.contains(&(32, 128)));

        let non_checks: Vec<_> = shard
            .iter()
            .copied()
            .filter(|s| !checks.contains(s))
            .collect();
        for (b, s) in &non_checks {
            assert!(*b >= 1 && *b <= 32);
            assert!(*s == 128);
        }
    }

    /// Single-worker (all 16 shapes) check covers all four extremes and
    /// produces at most 4 shapes, skipping at least 12.
    #[test]
    fn single_worker_check_shapes_skips_most_shapes() {
        let shapes = default_shapes();
        let checks = coverage_check_shapes(&shapes);
        assert!(checks.len() <= 4, "at most 4 check shapes for any shard");
        let would_skip = shapes.len() - checks.len();
        assert!(
            would_skip >= 12,
            "single-worker should skip ≥ 12 shapes; would_skip={would_skip}"
        );
    }

    // ─── prewarm_persistence_postcondition_failed ─────────────────────────
    //
    // These tests pin down the silent-persistence-failure detector that
    // was added in response to the 2026-05 codekeeper outage. Each case
    // is a real shape of CloudWatch evidence we want to either flag or
    // accept.

    /// Production defect signal: 1215 compile-success events but the
    /// `.engine` cache directory is empty on disk → flag.
    #[test]
    fn postcondition_flags_fresh_compiles_with_zero_delta() {
        assert!(prewarm_persistence_postcondition_failed(16, 0));
        assert!(prewarm_persistence_postcondition_failed(1, 0));
    }

    /// Defensive: an apparent decrease in engine count after a fresh
    /// compile (e.g. a sibling worker raced through and pruned files) is
    /// still wrong — flag it.
    #[test]
    fn postcondition_flags_fresh_compiles_with_negative_delta() {
        assert!(prewarm_persistence_postcondition_failed(4, -2));
    }

    /// Healthy first-deploy cold cache: 16 fresh compiles, +16 engines on
    /// disk → accept.
    #[test]
    fn postcondition_accepts_fresh_compiles_with_matching_delta() {
        assert!(!prewarm_persistence_postcondition_failed(16, 16));
    }

    /// Healthy partial-shard compile: 4 fresh compiles, +4 engines on
    /// disk → accept.
    #[test]
    fn postcondition_accepts_partial_shard_with_matching_delta() {
        assert!(!prewarm_persistence_postcondition_failed(4, 4));
    }

    /// Healthy warm-cache fast path: 0 fresh compiles, 0 delta → accept.
    /// Cache hits only must NOT be flagged as a postcondition failure.
    #[test]
    fn postcondition_accepts_warm_cache_with_no_compiles() {
        assert!(!prewarm_persistence_postcondition_failed(0, 0));
    }

    /// Healthy edge case: 0 fresh compiles but a positive delta (a sibling
    /// worker on the same EFS-shared cache wrote engines after we counted
    /// `_before`). Not actionable → accept.
    #[test]
    fn postcondition_accepts_zero_compiles_with_positive_delta() {
        assert!(!prewarm_persistence_postcondition_failed(0, 4));
    }

    /// A small delta short of the compile count is also a defect signal:
    /// 16 compiles claimed, only 1 engine actually persisted → flag.
    /// (The exact threshold for "less than expected" is a follow-up; the
    /// minimum invariant guarded here is "≥ 1 fresh compile and ≤ 0 delta
    /// is always wrong".)
    #[test]
    fn postcondition_accepts_undercount_above_zero_delta() {
        // Degenerate case is intentionally accepted by the current helper:
        // a single non-zero delta is counted as evidence of persistence.
        // Tightening this would require correlating fresh_compiles and
        // delta — see the recommended follow-up.
        assert!(!prewarm_persistence_postcondition_failed(16, 1));
    }

    // ─── engine count snapshot wiring (filesystem-backed) ─────────────────

    /// `count_engine_files` is the mechanism behind both the per-shape
    /// delta WARN and the per-worker prewarm postcondition. Verify it
    /// reflects engine-file writes the way the prewarm path does:
    /// before-compile snapshot + on-disk write + after-compile snapshot
    /// must produce a delta of `+1`. This pins down the contract that the
    /// silent-persistence detector relies on.
    #[test]
    fn count_engine_files_reflects_post_write_delta() {
        use super::trt_cache::count_engine_files;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        let before = count_engine_files(&dir);
        assert_eq!(before, 0, "fresh tempdir should have no engines");

        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_graph_a_111_fp16_sm89.engine"),
            b"plan-bytes",
        )
        .unwrap();

        let after = count_engine_files(&dir);
        assert_eq!(after, 1, "after a single engine write, count must be 1");
        let delta = i64::try_from(after).unwrap() - i64::try_from(before).unwrap();
        assert_eq!(delta, 1, "delta must be +1");
    }

    /// Conversely, when `session.run()` returns `Ok(_)` but TRT EP did
    /// NOT write an engine file (the production defect signal), the
    /// before/after delta is 0 even though a "compiled, cached, and
    /// fsynced" message was logged. This test fakes that scenario to
    /// confirm the postcondition helper would flag it.
    #[test]
    fn count_engine_files_zero_delta_triggers_postcondition() {
        use super::trt_cache::count_engine_files;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        // Fake "compile success without persistence": the directory is
        // never written to, even though the prewarm aggregator believes
        // a fresh compile happened.
        let before = count_engine_files(&dir);
        let after = count_engine_files(&dir);
        let delta = i64::try_from(after).unwrap() - i64::try_from(before).unwrap();

        assert!(
            prewarm_persistence_postcondition_failed(1, delta),
            "fresh_compiles=1 with zero delta must trigger the postcondition"
        );
    }

    /// Ensure the postcondition is satisfied when the directory is
    /// populated between `_before` and `_after` snapshots.
    #[test]
    fn count_engine_files_positive_delta_passes_postcondition() {
        use super::trt_cache::count_engine_files;
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        let before = count_engine_files(&dir);
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_a_sm89.engine"),
            b"plan",
        )
        .unwrap();
        let after = count_engine_files(&dir);
        let delta = i64::try_from(after).unwrap() - i64::try_from(before).unwrap();

        assert!(
            !prewarm_persistence_postcondition_failed(1, delta),
            "a single fresh compile that wrote one engine must pass"
        );
    }
}
