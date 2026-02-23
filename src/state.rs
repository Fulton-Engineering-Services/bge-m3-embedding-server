use crate::embedder::EmbedPool;
use std::sync::atomic::AtomicBool;

pub struct AppState {
    pub pool: EmbedPool,
    pub ready: AtomicBool,
    pub max_batch: usize,
}
