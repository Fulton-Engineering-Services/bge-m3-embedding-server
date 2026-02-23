mod config;
mod embedder;
mod handler;
mod models;
mod state;

use axum::{routing::get, routing::post, Router};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use config::Config;
use embedder::Embedder;
use handler::{health, sparse_embeddings};
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cfg = Config::from_env();
    let cache_dir = cfg.cache_dir.clone();
    let bind_addr = cfg.bind_addr.clone();

    info!("Starting bge-m3-axum-fastembed-rs, cache dir: {cache_dir}, bind: {bind_addr}");

    let state = Arc::new(AppState {
        embedder: Mutex::new(None),
        ready: AtomicBool::new(false),
    });

    let app = Router::new()
        .route("/v1/sparse-embeddings", post(sparse_embeddings))
        .route("/health", get(health))
        .with_state(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    info!("Listening on {bind_addr}");

    let state_for_loader = Arc::clone(&state);
    tokio::spawn(async move {
        info!("Loading BGE-M3 sparse model...");
        let result =
            tokio::task::spawn_blocking(move || Embedder::new(Path::new(&cache_dir))).await;

        match result {
            Ok(Ok(embedder)) => {
                *state_for_loader.embedder.lock().await = Some(embedder);
                state_for_loader
                    .ready
                    .store(true, std::sync::atomic::Ordering::Release);
                info!("BGE-M3 sparse model ready");
            }
            Ok(Err(e)) => {
                tracing::error!("Failed to load model: {e}");
                std::process::exit(1);
            }
            Err(e) => {
                tracing::error!("Model load task panicked: {e}");
                std::process::exit(1);
            }
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}
