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

use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::sysinfo::{MemoryReading, MemorySource};
use arc_swap::ArcSwap;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;

/// Status of the background memory probe.
///
/// Stored in `AppState.probe_status` as an `AtomicU8` so it can be updated
/// from the background probe task without holding a lock.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    /// Probe not run — a cost-model override (`BGE_M3_DISABLE_AUTO_BUDGET`,
    /// `BGE_M3_TOKEN_BUDGET`, or explicit A/B env vars) was in effect.
    Disabled = 0,
    /// Probe is running in the background; workers are using conservative defaults.
    Running = 1,
    /// Probe completed successfully; fitted `(a, b)` are now active.
    Complete = 2,
    /// Probe failed or produced invalid coefficients; conservative defaults remain.
    Failed = 3,
    /// Probe was skipped — valid coefficients loaded from the EFS cache file.
    CacheHit = 4,
}

impl ProbeStatus {
    /// Returns the JSON-serializable string representation used in `/health`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Running => "running",
            Self::Complete => "complete",
            Self::Failed => "failed",
            Self::CacheHit => "cache_hit",
        }
    }

    /// Converts a raw `u8` (from `AtomicU8::load`) back to `ProbeStatus`.
    #[must_use]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Disabled,
            1 => Self::Running,
            2 => Self::Complete,
            4 => Self::CacheHit,
            _ => Self::Failed,
        }
    }
}

/// Shared application state injected into every request handler via [`axum::extract::State`].
pub struct AppState {
    /// The embedding worker pool. Handles dense and sparse embedding requests.
    pub pool: EmbedPool,
    /// Atomic flag set to `true` once model warm-up and readiness probes complete.
    ///
    /// Handlers check this before dispatching to the pool to return `503`
    /// while models are still loading.
    pub ready: AtomicBool,
    /// Maximum batch size enforced by the handler layer.
    pub max_batch: usize,
    /// Total number of workers configured at startup.
    ///
    /// Used by the `/health` endpoint to report degraded state when
    /// `live_workers < total_workers`.
    pub total_workers: usize,
    /// Maximum tokenized sequence length in use.
    pub max_seq_length: usize,
    /// Static memory-detection info written once before the probe starts.
    ///
    /// Written to `OnceLock` as soon as memory detection completes (before the
    /// background probe finishes), so `/health` can show `memory_source`,
    /// `available_bytes`, and `model_rss_bytes_per_worker` even while the probe
    /// is still running.
    pub tuning: OnceLock<TuningInfo>,
    /// Live cost-model coefficients.
    ///
    /// Initialized to conservative defaults at startup. Updated atomically by
    /// the background probe (or cache-hit path) once fitted coefficients are
    /// available. All workers share this same handle and observe the update
    /// lock-free on their next `session.run()` call.
    pub cost_model: Arc<ArcSwap<CostModel>>,
    /// Current state of the background memory probe.
    ///
    /// Updated atomically from the background probe task. Read by `/health`
    /// to expose `probe_status` in the `tuning` block.
    pub probe_status: AtomicU8,
    /// Concurrency gate for in-flight embedding requests.
    ///
    /// Initialized to `max(cfg_workers - 1, 1)` permits, reserving one worker
    /// slot for the background auto-budget probe.  Raised to `cfg_workers`
    /// atomically on every terminal probe-status transition (`Disabled`,
    /// `CacheHit`, `Complete`, `Failed`) so full concurrency is available once
    /// the probe no longer needs a reserved worker.
    ///
    /// Test helpers set this to `usize::MAX` (effectively uncapped) so that
    /// existing tests do not need to acquire a permit.
    pub request_permits: Arc<Semaphore>,
}

/// Static workspace memory info surfaced by the `/health` endpoint.
///
/// Written once to [`AppState::tuning`] immediately after memory detection.
/// The cost-model fields (`a`, `b`, `max_workspace_bytes`) are served
/// dynamically from [`AppState::cost_model`] so they reflect the live
/// probe result rather than the initial conservative defaults.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TuningInfo {
    /// Where the available-memory reading came from.
    pub memory_source: String,
    /// Total available bytes detected at startup.
    pub available_bytes: usize,
    /// Measured model session RSS delta (bytes) — max across all workers.
    ///
    /// Accurate on Linux via `/proc/self/status`; `0` on other platforms.
    pub model_rss_bytes_per_worker: usize,
    /// Worst-case total peak memory (bytes) when all workers run simultaneously
    /// at their per-worker workspace ceiling.
    ///
    /// Formula: `cfg_workers × per_worker_workspace + cfg_workers × model_rss + OS_HEADROOM`.
    pub worst_case_peak_bytes: usize,
    /// Worst-case peak as a percentage of detected available memory.
    ///
    /// A value above 90% triggers a startup `WARN` log.
    pub utilization_pct: f64,
}

impl TuningInfo {
    /// Creates a [`TuningInfo`] from a memory reading and probe measurements.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mem: &MemoryReading,
        model_rss_per_worker: usize,
        worst_case_peak_bytes: usize,
        utilization_pct: f64,
    ) -> Self {
        Self {
            memory_source: mem.source.to_string(),
            available_bytes: mem.available_bytes,
            model_rss_bytes_per_worker: model_rss_per_worker,
            worst_case_peak_bytes,
            utilization_pct,
        }
    }

    /// Convenience builder for the case where memory detection was not possible
    /// (macOS without cgroup support, or probe disabled).
    #[must_use]
    #[allow(dead_code)]
    pub fn unknown(model_rss_per_worker: usize) -> Self {
        Self {
            memory_source: MemorySource::HostRam.to_string(),
            available_bytes: 0,
            model_rss_bytes_per_worker: model_rss_per_worker,
            worst_case_peak_bytes: 0,
            utilization_pct: 0.0,
        }
    }
}
