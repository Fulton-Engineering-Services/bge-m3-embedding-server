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

//! Per-shape `TensorRT` warmup runner.
//!
//! Extracted from `trt_warmup.rs` so the parent module can stay at facade
//! length while keeping all per-shape logging, cache-hit classification, and
//! engine-count snapshotting in one focused unit.
//!
//! [`run_warmup_shape`] is invoked once per `(batch, seq)` shape by
//! `trt_prewarm` (both during the dimensional-extreme coverage check and
//! during the full compile path).  [`ShapeRunResult`] is the per-call
//! summary the caller aggregates into the worker-scoped `PrewarmStats`.

use std::path::Path;

use super::super::trt_cache;
use super::super::worker::probe_run_dense;
use super::CACHE_HIT_THRESHOLD_MS;

/// Result of a single [`run_warmup_shape`] call.
///
/// Per-shape `.engine` count snapshots are logged inline by
/// `run_warmup_shape`; only aggregate stats are propagated up to the worker
/// (via `PrewarmStats`).
pub(super) struct ShapeRunResult {
    pub(super) compile_ms: u64,
    pub(super) fsync_ms: u64,
    /// `true` when `compile_ms < CACHE_HIT_THRESHOLD_MS` (engine loaded from
    /// disk cache) rather than compiled from scratch.
    pub(super) cache_hit: bool,
    pub(super) succeeded: bool,
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
/// `sm` selects which engine plans count toward the before/after snapshots:
/// `Some("smXY")` filters to plans matching this worker's GPU compute
/// capability so a heterogeneous cache (e.g. stale `sm89` plans next to
/// fresh `sm120` plans) is never miscounted; `None` is a passthrough that
/// counts every `.engine` file (legacy behaviour, used when detection failed).
/// See [`super::super::trt_cache::engine_files_for_sm`] for the filter
/// semantics.
///
/// The `shape_index` / `shape_total` parameters are purely for the
/// operator-visible log message and do not affect logic.
///
/// `#[allow(clippy::too_many_arguments)]` is acceptable here because every
/// argument is logically distinct — `(batch, seq)` already has its own
/// pair-of-`usize` shape, and bundling the remaining diagnostic positional
/// fields (`worker_id`, `shape_index`, `shape_total`, `sm`) into an
/// auxiliary struct would obscure the per-shape ergonomics for the only
/// caller, `trt_prewarm`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_warmup_shape(
    session: &mut ort::session::Session,
    batch: usize,
    seq: usize,
    worker_id: usize,
    shape_index: usize,
    shape_total: usize,
    engine_cache_dir: &Path,
    sm: Option<&str>,
) -> ShapeRunResult {
    let ids = ndarray::Array2::<i64>::zeros((batch, seq));
    let mask = ndarray::Array2::<i64>::ones((batch, seq));

    let engine_count_before = trt_cache::count_engine_files_for_sm(engine_cache_dir, sm);

    tracing::info!(
        worker_id,
        batch,
        seq,
        shape_index,
        shape_total,
        engine_count_before,
        detected_sm = sm.unwrap_or("unfiltered"),
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

            let engine_count_after = trt_cache::count_engine_files_for_sm(engine_cache_dir, sm);
            let engine_count_increased = engine_count_after > engine_count_before;

            // A non-cache-hit run that reports `Ok(_)` from session.run() but
            // leaves the on-disk `.engine` count at zero is the silent failure
            // mode behind the 2026-05 codekeeper outage.
            //
            // NOTE: The condition is `engine_count_after == 0`, NOT
            // `!engine_count_increased`. ORT's TRT EP writes one profile-based
            // engine file that covers all (batch, seq) shapes via [min, max]
            // ranges — it rewrites that file in-place as the profile expands,
            // so `engine_count_before == engine_count_after` (delta == 0) is
            // the normal steady-state after the first compile. A WARN on every
            // delta==0 shape is a false positive; WARN only when no file
            // exists at all.
            if !cache_hit && engine_count_after == 0 {
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
                    detected_sm = sm.unwrap_or("unfiltered"),
                    cache_path = %engine_cache_dir.display(),
                    "TensorRT pre-warm: compile-success log fired but engine_count is still \
                     zero — TRT EP may not be persisting engine plan files"
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
                detected_sm = sm.unwrap_or("unfiltered"),
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
            let engine_count_after = trt_cache::count_engine_files_for_sm(engine_cache_dir, sm);
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
                detected_sm = sm.unwrap_or("unfiltered"),
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
