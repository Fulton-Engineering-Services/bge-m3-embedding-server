use crate::embedder::Embedder;
use std::sync::atomic::AtomicBool;
use tokio::sync::Mutex;

pub struct AppState {
    pub embedder: Mutex<Option<Embedder>>,
    pub ready: AtomicBool,
}
