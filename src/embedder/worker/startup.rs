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

//! Worker startup: model load, optional cache GC, TRT prewarm, readiness signal.

use std::path::Path;
use std::sync::atomic::Ordering;

use anyhow::Result;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tracing::{info, info_span};

use super::config::WorkerConfig;
use super::prewarm_strict::should_fail_readiness;
use super::probe::probe_run_dense;
use crate::config::EpSelection;
use crate::embedder::session::{GpuSessionConfig, load_models};
use crate::embedder::sm_detect::detect_sm_for_device;
use crate::embedder::trt_cache;
#[cfg(feature = "cache-gc")]
use crate::embedder::trt_cache_gc;
use crate::embedder::trt_warmup::{
    prewarm_persistence_postcondition_failed, prewarm_persistence_suspicious_undercount,
    trt_prewarm,
};
use crate::sysinfo;

/// Outcome of the blocking startup sequence before the request loop begins.
pub(super) struct StartupOutcome {
    pub initial_models: (ort::session::Session, tokenizers::Tokenizer),
    pub detected_sm: Option<String>,
}

/// Loads models, runs optional TRT prewarm, and signals readiness.
#[allow(clippy::too_many_lines)]
pub(super) fn startup_worker(
    id: usize,
    cache_dir: &Path,
    ready_tx: &mpsc::Sender<Result<usize>>,
    config: &WorkerConfig,
    rt: &Handle,
) -> Result<StartupOutcome> {
    let span = info_span!("worker", id = id);
    let _span_guard = span.enter();

    tracing::debug!(
        worker_id = id,
        gpu_device = config.device_id,
        ep = %config.ep,
        "worker assigned to GPU device"
    );
    info!("Loading models (worker {id})...");
    let load_start = std::time::Instant::now();

    let pre_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(0);
    let mut initial_models = match load_models(
        &GpuSessionConfig {
            cache_dir,
            model_variant: config.model_variant,
            max_seq_length: config.max_seq_length,
            intra_threads: config.intra_threads,
            ep: config.ep,
            device_id: config.device_id,
            trt_max_workspace_bytes: config.trt_max_workspace_bytes,
            gpu_mem_limit_bytes: config.gpu_mem_limit_bytes,
        },
        id == 0,
    ) {
        Ok(mut models) => {
            // Prime the ORT session arena with a tiny session.run() BEFORE
            // measuring post-load RSS. ORT lazily allocates ~1 GiB of arena
            // bookkeeping on the first run() call regardless of input size;
            // priming here folds that allocation into the per-worker model
            // RSS measurement so the workspace-budget math on the main thread
            // sees the realistic per-worker memory footprint, AND so the
            // probe sweep's per-shape `rss_delta` readings reflect only the
            // incremental workspace attributable to that shape.
            //
            // Without per-worker priming, the probe could dispatch shapes to
            // workers that have not yet done a session.run(), and each such
            // first-touch contributes ~1 GiB of arena init noise to its
            // delta — which buries the per-shape workspace signal in the
            // OLS fit.
            let prime_ids = ndarray::Array2::<i64>::zeros((1, 8));
            let prime_mask = ndarray::Array2::<i64>::ones((1, 8));
            match probe_run_dense(&mut models.0, &prime_ids, &prime_mask) {
                Ok(_) => {
                    tracing::debug!("Worker {id} arena primed");
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Worker {id} arena prime failed; first probe shape on this \
                         worker will include arena init delta"
                    );
                }
            }

            let post_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(pre_load_rss);
            tracing::info!(
                elapsed_ms = load_start.elapsed().as_millis(),
                rss_delta_mb = post_load_rss.saturating_sub(pre_load_rss) / (1024 * 1024),
                "Models loaded (worker {id})"
            );
            models
        }
        Err(e) => {
            let _ =
                rt.block_on(ready_tx.send(Err(anyhow::anyhow!("Worker {id} failed to load: {e}"))));
            return Err(e);
        }
    };

    // Destructive stale-SM cache GC (feature-gated `cache-gc` + runtime
    // `BGE_M3_TRT_CACHE_GC_ENABLED=1`). Both gates must be on for any
    // deletion to occur. When the feature is OFF this entire block is
    // physically absent from the binary. See `trt_cache_gc.rs` for the
    // multi-SM ASG hazard model.
    // Only the leader worker (id == 0) runs GC. Workers load sequentially
    // (CLAUDE.md invariant), so worker 0 always finishes GC before worker 1
    // starts — this guard is race-free and prevents spurious WARN cascades on
    // workers 1..N that would otherwise each attempt (and log) the same sweep.
    #[cfg(feature = "cache-gc")]
    if id == 0 && config.trt_cache_gc_enabled && config.ep != EpSelection::Cpu {
        let engine_cache_dir = trt_cache::engine_cache_path(cache_dir);
        match detect_sm_for_device(config.device_id) {
            Some(current_sm) => {
                tracing::warn!(
                    target: "bge_m3_embedding_server::trt_cache_gc",
                    worker_id = id,
                    gpu_device = config.device_id,
                    current_sm = %current_sm,
                    cache_path = %engine_cache_dir.display(),
                    sidecar_suffixes = ?trt_cache_gc::ENGINE_SIDE_SUFFIXES,
                    "destructive cache GC about to run: will delete every \
                     `_smXX.engine` whose XX != current SM, plus aligned \
                     sidecars. DO NOT use this build against a shared EFS \
                     engine cache in a multi-SM ASG"
                );
                let stats = trt_cache_gc::gc_stale_sm_plans(&engine_cache_dir, &current_sm);
                tracing::warn!(
                    target: "bge_m3_embedding_server::trt_cache_gc",
                    worker_id = id,
                    gpu_device = config.device_id,
                    current_sm = %current_sm,
                    plans_deleted = stats.plans_deleted,
                    bytes_freed = stats.bytes_freed,
                    other_sms_observed = ?stats.other_sms_observed,
                    cache_path = %engine_cache_dir.display(),
                    "destructive cache GC ran: deleted other-SM engine plans \
                     from shared cache"
                );
            }
            None => {
                tracing::warn!(
                    target: "bge_m3_embedding_server::trt_cache_gc",
                    worker_id = id,
                    gpu_device = config.device_id,
                    "destructive cache GC requested but current GPU compute \
                     capability could not be detected (no nvidia-smi or \
                     unparseable output); skipping GC"
                );
            }
        }
    }

    // Detect this worker's GPU compute capability (e.g. `sm89`, `sm120`) once,
    // before TRT prewarm, so every subsequent cache enumeration restricts
    // itself to plans matching this GPU. Without this, a worker on `sm120`
    // would count a stale `_sm89.engine` plan toward `cache_hit` and report
    // a misleading `engine_count_before:3` on a fresh Blackwell deploy with
    // leftover L40S plans on the shared EFS cache — a common heterogeneous-SM
    // false-positive failure mode. `None` (detection failed) keeps legacy
    // unfiltered semantics so an operator missing `nvidia-smi` mid-deploy
    // does not see a hard regression.
    let detected_sm: Option<String> = if config.ep == EpSelection::TensorRt {
        let sm = detect_sm_for_device(config.device_id);
        if sm.is_none() {
            tracing::warn!(
                worker_id = id,
                gpu_device = config.device_id,
                "trt cache: nvidia-smi compute-capability detection failed; \
                 falling back to unfiltered engine cache counts. This re-introduces \
                 the heterogeneous-SM false-positive risk — install nvidia-smi on \
                 the container or check that the GPU is accessible."
            );
        }
        sm
    } else {
        None
    };

    // TensorRT engine pre-warming: compile engine files for each configured
    // shape before signaling readiness.  This runs BEFORE ready_tx.send() so
    // the worker is not marked ready until all TRT engines are cached —
    // `/health` correctly returns `503 loading` during the compile window.
    //
    // Each shape may take 30–120 s on first deploy; subsequent starts reuse
    // the cached `.engine` files from `{cache_dir}/trt-engines/` (seconds).
    if config.ep == EpSelection::TensorRt && !config.trt_warmup_shapes.is_empty() {
        let engine_cache_dir = trt_cache::engine_cache_path(cache_dir);
        let total_engine_count = trt_cache::count_engine_files(&engine_cache_dir);
        let matching_engine_count =
            trt_cache::count_engine_files_for_sm(&engine_cache_dir, detected_sm.as_deref());
        // Greppable structured log line (target=bge_m3_embedding_server::trt_cache):
        // operators searching CloudWatch for `detected_sm` / `matching_engine_count`
        // get an immediate picture of the heterogeneous-cache situation. A line
        // showing `matching_engine_count:0, total_engine_count:3` is the visual
        // signature of the bug this commit fixes.
        tracing::info!(
            target: "bge_m3_embedding_server::trt_cache",
            worker_id = id,
            device_id = config.device_id,
            detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
            cache_path = %engine_cache_dir.display(),
            matching_engine_count,
            total_engine_count,
            "trt cache: SM-filtered engine plan enumeration"
        );
        tracing::info!(
            worker_id = id,
            gpu_device = config.device_id,
            detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
            shape_count = config.trt_warmup_shapes.len(),
            shapes = ?config.trt_warmup_shapes,
            "TensorRT pre-warm: worker compiling shard \
             (first run per shape takes 30–170 s; subsequent starts reuse cache)"
        );
        trt_cache::log_engine_basenames_before_prewarm_for_sm(
            &engine_cache_dir,
            detected_sm.as_deref(),
        );
        let stats = trt_prewarm(
            &mut initial_models.0,
            &config.trt_warmup_shapes,
            id,
            cache_dir,
            detected_sm.as_deref(),
        );
        // Per-worker postcondition: if the shard reported one or more fresh
        // (non-cache-hit) compiles but the on-disk engine count did not
        // increase, surface an ERROR. This is the in-process counterpart to
        // the postcondition check at the end of the warmup-only path in
        // lib.rs, intended to catch the silent-persistence failure mode that
        // produced silent-persistence startup failures in production.
        if prewarm_persistence_postcondition_failed(stats.fresh_compiles, stats.engine_count_after)
        {
            tracing::error!(
                worker_id = id,
                gpu_device = config.device_id,
                detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                engine_count_delta = stats.engine_count_delta,
                cache_path = %engine_cache_dir.display(),
                "TensorRT pre-warm postcondition failed: \
                 compile-success events present but no .engine files on disk \
                 for this SM; TRT EP may be silently failing to persist engine \
                 plan files (or building plans for a different SM than this \
                 worker's device)"
            );
        } else if prewarm_persistence_suspicious_undercount(
            stats.fresh_compiles,
            stats.engine_count_after,
        ) {
            // Non-fatal: TRT EP can legitimately reuse a single `.engine`
            // file across many input shapes (engine plans are keyed by
            // fused-subgraph identity + precision + GPU SM, not by
            // `(batch, seq)`). A 1:2 ratio is tolerated silently; this
            // WARN fires only when delta * 2 < fresh_compiles AND the
            // postcondition above did not already trigger. Greppable
            // tag: "engine_count_delta is suspiciously low".
            tracing::warn!(
                worker_id = id,
                gpu_device = config.device_id,
                detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                engine_count_delta = stats.engine_count_delta,
                cache_path = %engine_cache_dir.display(),
                "TensorRT pre-warm: engine_count_delta is suspiciously low \
                 relative to fresh_compiles (delta * 2 < fresh_compiles); \
                 some engine plans may not have persisted to disk despite \
                 session.run() reporting Ok — investigate cache path \
                 resolution and EFS mount durability"
            );
        }
        tracing::info!(
            worker_id = id,
            detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
            warmed = stats.warmed,
            skipped = stats.skipped,
            fully_cached = stats.fully_cached,
            fresh_compiles = stats.fresh_compiles,
            engine_count_before = stats.engine_count_before,
            engine_count_after = stats.engine_count_after,
            engine_count_delta = stats.engine_count_delta,
            total = config.trt_warmup_shapes.len(),
            total_compile_ms = stats.total_compile_ms,
            total_fsync_ms = stats.total_fsync_ms,
            max_warmed_seq = stats.max_warmed_seq,
            "TensorRT pre-warm complete"
        );

        // Raise the pool-wide in-band JIT guard ceiling to the highest
        // sequence tier this worker successfully warmed. With seq-homogeneous
        // sharding (worker N gets one full seq tier) the union across workers
        // reconstructs the full grid coverage; a tier that failed on every
        // worker leaves the ceiling below it so real requests at that tier are
        // refused (HTTP 503) instead of triggering a process-killing in-band
        // JIT. See `jit_guard.rs`.
        let prev_ceiling = config
            .warmed_seq_ceiling
            .fetch_max(stats.max_warmed_seq, Ordering::AcqRel);
        tracing::info!(
            target: "bge_m3_embedding_server::trt_warmup",
            worker_id = id,
            max_warmed_seq = stats.max_warmed_seq,
            warmed_seq_ceiling = prev_ceiling.max(stats.max_warmed_seq),
            "in-band JIT guard: warmed-seq ceiling updated after prewarm"
        );

        // Strict-mode escalation (BGE_M3_PREWARM_STRICT): when the prewarm
        // postcondition signals that compile-success events occurred but no
        // engine plan files persisted, refuse to signal ready. The pool's
        // init task converts the worker error into an init-handle failure,
        // and `bootstrap::readiness::run_readiness_probe` then triggers a
        // hard process exit. ECS retries the task, which is the correct
        // response to "every worker hit TRT autotuner OOM mid-build" rather
        // than serving HTTP 500 traffic from a known-broken pool.
        if should_fail_readiness(
            stats.fresh_compiles,
            stats.engine_count_after,
            config.prewarm_strict,
        ) {
            tracing::error!(
                target: "bge_m3_embedding_server::prewarm",
                worker_id = id,
                gpu_device = config.device_id,
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                cache_path = %trt_cache::engine_cache_path(cache_dir).display(),
                prewarm_strict = config.prewarm_strict,
                "Prewarm postcondition failed and prewarm_strict=1: \
                 refusing to signal ready"
            );
            let err = anyhow::anyhow!(
                "Worker {id} prewarm postcondition failed \
                 (fresh_compiles={}, engine_count_after={}); \
                 prewarm_strict=1: refusing to signal ready",
                stats.fresh_compiles,
                stats.engine_count_after,
            );
            return Err(err);
        }
    }

    // Report the RSS delta so EmbedPool can derive the true per-worker
    // model footprint for workspace-budget calculations.
    let post_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(pre_load_rss);
    let rss_delta = post_load_rss.saturating_sub(pre_load_rss);
    info!(
        "Worker {id} models loaded — signaling ready (rss_delta_mb={})",
        rss_delta / (1024 * 1024)
    );
    let _ = rt.block_on(ready_tx.send(Ok(rss_delta)));

    Ok(StartupOutcome {
        initial_models,
        detected_sm,
    })
}
