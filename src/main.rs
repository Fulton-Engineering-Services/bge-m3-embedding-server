mod config;
mod embedder;
mod error;
mod handler;
mod models;
mod state;

use axum::{routing::get, routing::post, Router};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

use config::Config;
use embedder::EmbedPool;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cfg = Config::from_env();

    info!(
        bind = %cfg.bind_addr,
        workers = cfg.workers,
        max_batch = cfg.max_batch,
        cache_dir = %cfg.cache_dir,
        "Starting bge-m3-axum-fastembed-rs"
    );

    let (pool, init_handle) = EmbedPool::spawn(cfg.workers, PathBuf::from(&cfg.cache_dir));

    let state = Arc::new(AppState {
        pool,
        ready: AtomicBool::new(false),
        max_batch: cfg.max_batch,
    });

    let app = Router::new()
        .route("/v1/embeddings", post(handler::dense_embeddings))
        .route("/v1/sparse-embeddings", post(handler::sparse_embeddings))
        .route("/health", get(handler::health))
        .layer(TraceLayer::new_for_http())
        .with_state(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    info!(bind = %cfg.bind_addr, "Listening");

    let state_for_readiness = Arc::clone(&state);
    tokio::spawn(async move {
        // Wait for all worker threads to finish loading their models.
        match init_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                error!("Worker pool initialization failed: {e}");
                std::process::exit(1);
            }
            Err(e) => {
                error!("Worker pool task panicked: {e}");
                std::process::exit(1);
            }
        }

        // Warm-up probe: verify the pool can actually serve requests.
        match state_for_readiness.pool.dense(vec!["ready".into()]).await {
            Ok(_) => {
                state_for_readiness.ready.store(true, Ordering::Release);
                info!("Model ready — accepting requests");
            }
            Err(e) => {
                error!("Readiness probe failed: {e}");
                std::process::exit(1);
            }
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}
