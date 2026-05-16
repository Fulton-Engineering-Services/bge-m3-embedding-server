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

//! Library crate for the bge-m3 embedding server.
//!
//! `main.rs` is a 20–30 line entry point that calls [`run`]; all real
//! orchestration logic lives here so it can be unit-tested and reused from
//! integration tests without spawning the binary.

// Rustdoc lints — enforce documentation quality
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![warn(rustdoc::unescaped_backticks)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::bare_urls)]
#![warn(rustdoc::redundant_explicit_links)]
#![warn(rustdoc::private_doc_tests)]

pub mod binpack;
pub mod bootstrap;
pub mod config;
pub mod embedder;
pub mod error;
pub mod gpu_stats;
pub mod handler;
pub mod logging;
pub mod models;
pub mod probe;
pub mod state;
pub mod sysinfo;
pub mod weights;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use tokio::sync::Semaphore;
use tracing::info;

use crate::binpack::CostModel;
use crate::bootstrap::{build_router, run_readiness_probe};
use crate::config::{Config, EpSelection};
use crate::embedder::adaptive_warmup::{AdaptiveWarmupConfig, JitSuspectSender};
use crate::embedder::{EmbedPool, WorkerConfig};
use crate::gpu_stats::GpuStatsCollector;
use crate::state::{AppState, ProbeStatus};

/// Process-exit code emitted by the warmup-only path when the postcondition
/// fails (compile-success events present but no `.engine` files on disk).
///
/// Distinct from `1` (general failure) so operators can tell the warmup
/// container "compiled successfully but did not persist anything" apart
/// from "compile errored mid-run". Surfaced in the wrapping ECS task /
/// EC2 userdata log group as the container's `exitCode`.
pub const WARMUP_POSTCONDITION_FAILED_EXIT_CODE: i32 = 2;

/// Decides whether the warmup-only postcondition is violated.
///
/// The postcondition is: when `BGE_M3_EP=tensorrt` is set and the warmup-only
/// path has run to completion, at least one `.engine` file must exist in the
/// engine cache directory. A `true` return value should produce an `ERROR`
/// log and a non-zero exit so deployments fail loudly instead of silently
/// looking healthy with a perpetually-cold cache.
///
/// Non-TensorRT EPs are exempt — they do not produce engine plan files at
/// all, so an empty engine cache directory is the expected steady state.
#[must_use]
pub fn warmup_postcondition_failed(ep: EpSelection, engine_count: usize) -> bool {
    ep == EpSelection::TensorRt && engine_count == 0
}

/// Polls `live_workers` until it reaches zero or `timeout` elapses.
///
/// Used by the warmup-only path after the [`EmbedPool`] handle has been
/// dropped: closing the request channel asks each worker to break out of
/// its receive loop, drop its `ort::session::Session`, and return — but
/// dropping the pool only signals intent. The actual worker exit happens
/// asynchronously on a `spawn_blocking` thread, and we want the ORT/TRT
/// destructor to run BEFORE the process exits so any session-shutdown
/// flushes (e.g. the TRT timing cache) make it to disk.
///
/// Returns silently on timeout — callers proceed to fsync + exit regardless.
/// Engine plan files are durable independent of this wait (each shape
/// fsyncs before returning from `run_warmup_shape`), so a slow worker
/// shutdown only loses ancillary state.
async fn wait_for_workers_to_exit(live_workers: &Arc<AtomicUsize>, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while live_workers.load(Ordering::Acquire) > 0 {
        if std::time::Instant::now() >= deadline {
            tracing::warn!(
                live_workers = live_workers.load(Ordering::Acquire),
                timeout_secs = timeout.as_secs(),
                "warmup-only mode: timed out waiting for workers to exit; \
                 proceeding to fsync and process exit"
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Runs the embedding server end-to-end: load config, spawn the worker pool,
/// install the readiness probe, start the heartbeat, and serve HTTP traffic.
///
/// Background tasks log and call `process::exit(1)` on their own unrecoverable
/// failures so the container is restarted by the orchestrator.
///
/// # Errors
///
/// Returns `Err` if the TCP listener cannot bind to the configured address.
#[allow(clippy::too_many_lines)]
pub async fn run() -> anyhow::Result<()> {
    info!(
        version = env!("CARGO_PKG_VERSION"),
        git_sha = env!("BGE_M3_GIT_SHA"),
        target_arch = std::env::consts::ARCH,
        target_os = std::env::consts::OS,
        profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        "bge-m3-embedding-server build info"
    );

    let cfg = Config::from_env();

    let disable_probe_cache = std::env::var("BGE_M3_DISABLE_PROBE_CACHE")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "yes"));

    info!(
        bind = %cfg.bind_addr,
        workers = cfg.workers,
        max_batch = cfg.max_batch,
        max_seq_length = cfg.max_seq_length,
        cache_dir = %cfg.cache_dir,
        idle_timeout_secs = cfg.idle_timeout.map(|d| d.as_secs()),
        model_variant = ?cfg.model_variant,
        memory_safety_factor = cfg.memory_safety_factor,
        auto_budget = cfg.cost_model_override.is_none(),
        disable_probe_cache,
        "Starting bge-m3-embedding-server"
    );

    // Allocate one shared cost-model handle.  Conservative defaults are used
    // until the background probe (or cache hit) updates the handle via ArcSwap.
    // All workers share the same Arc<ArcSwap<CostModel>> so a single store()
    // call in the probe task is immediately visible to every worker.
    let initial_cost_model = cfg
        .cost_model_override
        .unwrap_or_else(|| CostModel::conservative(CostModel::DEFAULT_MAX_WORKSPACE));
    let cost_model_handle = Arc::new(ArcSwap::from_pointee(initial_cost_model));

    // Request concurrency limiter.  Start with cfg_workers - 1 permits so the
    // background probe always has a worker slot free.  The probe (or any terminal
    // probe bypass) calls add_permits(1) to raise to cfg_workers once the probe
    // lifecycle ends.  Minimum is 1 so a single-worker deployment always accepts
    // at least one concurrent request (at the cost of a shared probe slot).
    let initial_permits = cfg.workers.saturating_sub(1).max(1);
    let request_permits = Arc::new(Semaphore::new(initial_permits));

    // Pre-create the JIT-suspect channel so the sender half can be placed in
    // WorkerConfig before the pool is spawned.  The receiver half is passed to
    // `spawn_adaptive_warmup` after the pool is created.  When adaptive warmup
    // is disabled the sender is dropped immediately (workers hold `None`).
    let (jit_suspect_tx, jit_suspect_rx): (
        Option<JitSuspectSender>,
        Option<tokio::sync::mpsc::Receiver<(usize, usize)>>,
    ) = if cfg.adaptive_warmup_enabled && cfg.ep == EpSelection::TensorRt {
        let (tx, rx) = tokio::sync::mpsc::channel::<(usize, usize)>(64);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let (pool, init_handle) = EmbedPool::spawn(
        cfg.workers,
        PathBuf::from(&cfg.cache_dir),
        WorkerConfig {
            cost_model: Arc::clone(&cost_model_handle),
            idle_timeout: cfg.idle_timeout,
            model_variant: cfg.model_variant,
            max_seq_length: cfg.max_seq_length,
            intra_threads: cfg.intra_threads,
            ep: cfg.ep,
            trt_warmup_shapes: cfg.trt_warmup_shapes,
            // device_id is overridden per-worker by EmbedPool::spawn;
            // the initial value here is a harmless placeholder.
            device_id: 0,
            gpu_count: cfg.gpu_count,
            trt_max_workspace_bytes: cfg.trt_max_workspace_bytes,
            gpu_mem_limit_bytes: cfg.gpu_mem_limit_bytes,
            jit_suspect_tx,
        },
    );

    // Warmup-only path: wait for all workers (and TRT engine compilation) to
    // finish, log the engine count, verify the on-disk postcondition, and
    // exit cleanly through `main.rs`.  No HTTP listener is bound — intended
    // for use as an ECS init container that pre-populates the shared EFS
    // engine cache before the main container starts.
    if cfg.warmup_only {
        // Emit GPU heartbeats while engines are compiling so operators have
        // VRAM and temperature visibility in CloudWatch during the warmup
        // window.  Uses the same GpuStatsCollector and interval as the
        // normal-mode heartbeat; no-op on CPU builds.
        let warmup_hb_handle = if cfg.heartbeat_secs > 0 {
            let gpu_stats = GpuStatsCollector::init(cfg.gpu_count);
            let heartbeat_secs = cfg.heartbeat_secs;
            Some(tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_secs(heartbeat_secs));
                tick.tick().await; // skip the immediate t=0 tick
                loop {
                    tick.tick().await;
                    gpu_stats.emit_heartbeat();
                }
            }))
        } else {
            None
        };

        init_handle
            .await
            .map_err(|e| anyhow::anyhow!("Worker pool task panicked: {e}"))?
            .map_err(|e| anyhow::anyhow!("Worker pool initialization failed: {e}"))?;

        if let Some(h) = warmup_hb_handle {
            h.abort();
        }

        // Explicit teardown BEFORE the postcondition check and exit.
        //
        // Dropping `pool` releases the only `mpsc::Sender<EmbedRequest>`
        // owned outside the worker threads, which closes the channel.
        // Each worker's recv() call returns `Ok(None)`, the worker breaks
        // out of its loop, drops its `ort::session::Session`, and exits.
        //
        // ORT TRT EP writes engine plan files synchronously inside
        // `session.run()`, but it can buffer auxiliary state (e.g. the
        // timing cache) that is only flushed on session destruction.
        // Calling `process::exit(0)` while sessions are still alive would
        // skip those destructors, which is the failure mode we are
        // engineering away from.  After dropping the pool we wait briefly
        // for `live_workers` to drain so worker drop paths can run.
        let cache_dir_path = PathBuf::from(&cfg.cache_dir);
        let live_workers = pool.live_worker_count();
        let live_workers_arc = pool.live_workers_for_shutdown();
        drop(pool);

        wait_for_workers_to_exit(&live_workers_arc, Duration::from_secs(10)).await;
        let live_workers_after = live_workers_arc.load(Ordering::Acquire);
        info!(
            live_workers_before = live_workers,
            live_workers_after, "warmup-only mode: pool dropped, waited for worker teardown"
        );

        // Final fsync sweep covers any sidecar files (timing cache,
        // `.profile`) that may have been written during the session-drop
        // path. Engine plan files were already fsynced inside
        // `run_warmup_shape`, so this second pass is belt-and-braces.
        let engine_cache_dir = crate::embedder::trt_cache::engine_cache_path(&cache_dir_path);
        crate::embedder::trt_cache::fsync_cache_dir(&engine_cache_dir);

        let trt_info = crate::embedder::trt_cache::ensure_and_inspect(&cache_dir_path);

        if warmup_postcondition_failed(cfg.ep, trt_info.engine_count) {
            tracing::error!(
                engine_count = trt_info.engine_count,
                cache_path = %trt_info.path.display(),
                ep = %cfg.ep,
                "warmup-only postcondition failed: compile-success events \
                 present but no .engine files on disk; check TRT EP \
                 construction, EFS mount, and engine cache path resolution"
            );
            std::process::exit(WARMUP_POSTCONDITION_FAILED_EXIT_CODE);
        }

        info!(
            engine_count = trt_info.engine_count,
            profile_count = trt_info.profile_count,
            cache_path = %trt_info.path.display(),
            ep = %cfg.ep,
            "warmup-only mode: all TRT engines compiled and cached, exiting"
        );
        return Ok(());
    }

    // Spawn the adaptive warmup background task if enabled.  A clone of the
    // pool is sufficient — `EmbedPool` is `Clone` (all fields are
    // reference-counted).  The warmup-only mode does not use adaptive warmup
    // (TRT engines are compiled synchronously during startup).
    if let Some(rx) = jit_suspect_rx {
        let adaptive_cfg = AdaptiveWarmupConfig {
            enabled: cfg.adaptive_warmup_enabled,
            quiet_secs: cfg.adaptive_warmup_quiet_secs,
            max_shapes_per_hour: cfg.adaptive_warmup_max_shapes_per_hour,
        };
        crate::embedder::adaptive_warmup::spawn_adaptive_warmup(adaptive_cfg, pool.clone(), rx);
        info!(
            quiet_secs = cfg.adaptive_warmup_quiet_secs,
            max_shapes_per_hour = cfg.adaptive_warmup_max_shapes_per_hour,
            "adaptive warmup task spawned"
        );
    }

    let state = Arc::new(AppState {
        pool,
        ready: AtomicBool::new(false),
        max_batch: cfg.max_batch,
        total_workers: cfg.workers,
        max_seq_length: cfg.max_seq_length,
        tuning: std::sync::OnceLock::new(),
        cost_model: cost_model_handle,
        probe_status: AtomicU8::new(ProbeStatus::Running as u8),
        request_permits,
    });

    let app = build_router(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    info!(bind = %cfg.bind_addr, "Listening");

    let state_for_readiness = Arc::clone(&state);
    let cfg_max_seq = cfg.max_seq_length;
    let cfg_workers = cfg.workers;
    let cfg_safety = cfg.memory_safety_factor;
    let cost_model_override = cfg.cost_model_override;
    let cache_dir = PathBuf::from(&cfg.cache_dir);
    let model_variant_str = cfg.model_variant.to_string();

    tokio::spawn(async move {
        if let Err(e) = run_readiness_probe(
            init_handle,
            state_for_readiness,
            cfg_max_seq,
            cfg_workers,
            cfg_safety,
            cost_model_override,
            cache_dir,
            model_variant_str,
            disable_probe_cache,
        )
        .await
        {
            tracing::error!("{e}");
            std::process::exit(1);
        }
    });

    // Periodic heartbeat — logs RSS, worker counts, queue depth, and permits
    // at a fixed interval so dashboards can detect slow leaks or saturation.
    // On GPU builds, also emits per-device VRAM and utilization stats.
    let heartbeat_secs = cfg.heartbeat_secs;
    if heartbeat_secs > 0 {
        let gpu_stats = GpuStatsCollector::init(cfg.gpu_count);
        let state_hb = Arc::clone(&state);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(heartbeat_secs));
            // Skip the first (immediate) tick so we don't log at t=0 before
            // the server has finished starting up.
            tick.tick().await;
            loop {
                tick.tick().await;
                let rss_mb = sysinfo::read_process_rss_bytes().unwrap_or(0) / (1024 * 1024);
                info!(
                    rss_mb,
                    live_workers = state_hb.pool.live_worker_count(),
                    loaded_workers = state_hb.pool.loaded_worker_count(),
                    queue_depth = state_hb.pool.queue_depth(),
                    available_permits = state_hb.request_permits.available_permits(),
                    probe_status =
                        ProbeStatus::from_u8(state_hb.probe_status.load(Ordering::Acquire))
                            .as_str(),
                    "heartbeat"
                );
                gpu_stats.emit_heartbeat();
            }
        });
    }

    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── warmup_postcondition_failed ──────────────────────────────────────

    #[test]
    fn postcondition_passes_for_tensorrt_with_engines_present() {
        assert!(!warmup_postcondition_failed(EpSelection::TensorRt, 1));
        assert!(!warmup_postcondition_failed(EpSelection::TensorRt, 16));
    }

    #[test]
    fn postcondition_fails_for_tensorrt_with_zero_engines() {
        assert!(
            warmup_postcondition_failed(EpSelection::TensorRt, 0),
            "TRT EP + 0 engines must be flagged as a postcondition failure"
        );
    }

    #[test]
    fn postcondition_passes_for_cpu_ep_regardless_of_engine_count() {
        assert!(!warmup_postcondition_failed(EpSelection::Cpu, 0));
        assert!(!warmup_postcondition_failed(EpSelection::Cpu, 16));
    }

    #[test]
    fn postcondition_passes_for_cuda_ep_regardless_of_engine_count() {
        assert!(!warmup_postcondition_failed(EpSelection::Cuda, 0));
        assert!(!warmup_postcondition_failed(EpSelection::Cuda, 16));
    }

    #[test]
    fn warmup_postcondition_exit_code_is_distinct_from_general_failure() {
        assert_eq!(
            WARMUP_POSTCONDITION_FAILED_EXIT_CODE, 2,
            "operators rely on exit_code=2 being specific to the warmup-only \
             postcondition; do not collapse it back into 1"
        );
    }

    // ─── wait_for_workers_to_exit ─────────────────────────────────────────
    //
    // These tests use real time deliberately: the helper polls a
    // `live_workers` atomic on a 100 ms tokio::time::sleep cadence, which
    // we want to exercise end-to-end. Adding `tokio/test-util` purely to
    // virtualise time would buy little here.

    /// When `live_workers` is already 0, the helper returns immediately
    /// without sleeping for the entire timeout window.
    #[tokio::test(flavor = "current_thread")]
    async fn wait_for_workers_returns_immediately_when_already_zero() {
        let live = Arc::new(AtomicUsize::new(0));
        let start = std::time::Instant::now();
        wait_for_workers_to_exit(&live, Duration::from_secs(60)).await;
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "wait should return immediately when live_workers is already zero; \
             elapsed={:?}",
            start.elapsed()
        );
    }

    /// Workers transitioning to zero mid-wait causes the helper to return
    /// well before the timeout.
    #[tokio::test(flavor = "current_thread")]
    async fn wait_for_workers_returns_when_count_reaches_zero() {
        let live = Arc::new(AtomicUsize::new(2));
        let live_for_task = Arc::clone(&live);
        let drainer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            live_for_task.store(0, Ordering::Release);
        });

        let start = std::time::Instant::now();
        wait_for_workers_to_exit(&live, Duration::from_secs(60)).await;
        let elapsed = start.elapsed();
        drainer.await.unwrap();

        assert_eq!(live.load(Ordering::Acquire), 0);
        assert!(
            elapsed < Duration::from_secs(5),
            "wait should have returned soon after counter reached zero; \
             elapsed={elapsed:?}"
        );
    }

    /// When `live_workers` never reaches zero, the helper returns at the
    /// timeout without blocking forever.
    #[tokio::test(flavor = "current_thread")]
    async fn wait_for_workers_times_out_when_count_stays_positive() {
        let live = Arc::new(AtomicUsize::new(3));
        let start = std::time::Instant::now();
        wait_for_workers_to_exit(&live, Duration::from_millis(300)).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(280),
            "wait should have honored the timeout; elapsed={elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "wait should not block past the timeout by more than one poll; \
             elapsed={elapsed:?}"
        );
        assert_eq!(
            live.load(Ordering::Acquire),
            3,
            "live counter must be untouched after timeout"
        );
    }
}
