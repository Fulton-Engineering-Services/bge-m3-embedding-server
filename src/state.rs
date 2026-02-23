use crate::embedder::EmbedPool;
use std::sync::atomic::AtomicBool;

/// Shared application state passed to all Axum handlers via `State<Arc<AppState>>`.
/// Fields are intentionally `pub` — this is a plain data carrier constructed once
/// in `main()` and read by handlers (ARC-3).
pub struct AppState {
    pub pool: EmbedPool,
    pub ready: AtomicBool,
    pub max_batch: usize,
    pub total_workers: usize,
}
