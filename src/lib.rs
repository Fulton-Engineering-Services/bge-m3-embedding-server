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
//!
//! ## Lints
//!
//! These pedantic clippy lints are allowed crate-wide:
//! - `missing_errors_doc` — internal-purpose lib; per-fn `# Errors` sections
//!   would duplicate the names of the wrapped underlying anyhow errors with
//!   no added value.
//! - `missing_panics_doc` — same rationale.
//! - `must_use_candidate` — many of the type's accessors return primitive
//!   `usize`/`bool` values that are routinely used at call sites without a
//!   formal `#[must_use]` reminder.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

pub mod binpack;
pub mod bootstrap;
pub mod config;
pub mod embedder;
pub mod error;
pub mod handler;
pub mod models;
pub mod probe;
pub mod state;
pub mod sysinfo;
pub mod weights;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use tokio::sync::Semaphore;
use tracing::info;

use crate::binpack::CostModel;
use crate::bootstrap::{build_router, run_readiness_probe};
use crate::config::Config;
use crate::embedder::{EmbedPool, WorkerConfig};
use crate::state::{AppState, ProbeStatus};

/// Runs the embedding server end-to-end: load config, spawn the worker pool,
/// install the readiness probe, start the heartbeat, and serve HTTP traffic.
///
/// Returns `Err` only on an unrecoverable startup failure (bind error, etc.);
/// background tasks log and `process::exit(1)` if their own setup fails so the
/// container is restarted by the orchestrator.
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

    let (pool, init_handle) = EmbedPool::spawn(
        cfg.workers,
        PathBuf::from(&cfg.cache_dir),
        WorkerConfig {
            cost_model: Arc::clone(&cost_model_handle),
            idle_timeout: cfg.idle_timeout,
            model_variant: cfg.model_variant,
            max_seq_length: cfg.max_seq_length,
            intra_threads: cfg.intra_threads,
        },
    );

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
    let heartbeat_secs = cfg.heartbeat_secs;
    if heartbeat_secs > 0 {
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
            }
        });
    }

    axum::serve(listener, app).await?;
    Ok(())
}
