use crate::embedder::EmbedPool;
use std::sync::atomic::AtomicBool;

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
}
