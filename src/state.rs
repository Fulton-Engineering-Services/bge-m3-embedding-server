use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::sysinfo::{MemoryReading, MemorySource};
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

/// Shared application state injected into every request handler via [`axum::extract::State`].
pub struct AppState {
    /// The embedding worker pool. Handles dense and sparse embedding requests.
    pub pool: EmbedPool,
    /// Atomic flag set to `true` once model warm-up completes.
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
    /// Derived or configured workspace cost model.
    ///
    /// Exposed by `/health` so operators can verify what the server derived.
    /// Written exactly once during init (before `ready` is set) via `OnceLock`.
    pub tuning: OnceLock<TuningInfo>,
}

/// Workspace tuning data surfaced by the `/health` endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TuningInfo {
    /// Bytes per token-position (linear FFN / projection term).
    pub a_bytes_per_token: f64,
    /// Bytes per token-position-squared (quadratic attention term).
    pub b_bytes_per_token_sq: f64,
    /// Maximum workspace bytes per worker per `session.run()` call.
    pub max_workspace_bytes: usize,
    /// Where the available-memory reading came from.
    pub memory_source: String,
    /// Total available bytes detected at startup.
    pub available_bytes: usize,
    /// Estimated model session RSS delta (bytes loaded by one worker).
    pub model_rss_bytes_per_worker: usize,
}

impl TuningInfo {
    pub fn new(
        cost_model: &CostModel,
        mem: &MemoryReading,
        model_rss_per_worker: usize,
    ) -> Self {
        Self {
            a_bytes_per_token: cost_model.a,
            b_bytes_per_token_sq: cost_model.b,
            max_workspace_bytes: cost_model.max_workspace_bytes,
            memory_source: mem.source.to_string(),
            available_bytes: mem.available_bytes,
            model_rss_bytes_per_worker: model_rss_per_worker,
        }
    }

    /// Convenience builder for the case where memory detection was not possible
    /// (macOS without cgroup support, or probe disabled).
    #[allow(dead_code)]
    pub fn unknown(cost_model: &CostModel) -> Self {
        Self {
            a_bytes_per_token: cost_model.a,
            b_bytes_per_token_sq: cost_model.b,
            max_workspace_bytes: cost_model.max_workspace_bytes,
            memory_source: MemorySource::HostRam.to_string(),
            available_bytes: 0,
            model_rss_bytes_per_worker: 0,
        }
    }
}
