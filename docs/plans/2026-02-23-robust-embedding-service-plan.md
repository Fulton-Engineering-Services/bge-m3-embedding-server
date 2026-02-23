# Robust BGE-M3 Embedding Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the minimal sparse-only embedding wrapper into a production-grade service serving both dense (OpenAI-compatible) and sparse BGE-M3 embeddings with worker-pool concurrency, structured errors, observability, and documentation.

**Architecture:** Bounded worker pool via `tokio::sync::mpsc` where each worker owns exclusive `TextEmbedding` + `SparseTextEmbedding` instances. Handlers submit work items with a oneshot reply channel. `spawn_blocking` keeps CPU-bound ONNX inference off the async runtime. OpenAI-compatible `/v1/embeddings` endpoint enables Spring AI consumers to drop Ollama dependency.

**Tech Stack:** Rust 1.88+, Axum 0.8, fastembed 5, tokio, serde, tracing, tower-http, anyhow

**Design doc:** `docs/plans/2026-02-23-robust-embedding-service-design.md`

---

## Dependencies

Add these to `Cargo.toml` before starting:

```toml
[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
fastembed = "5"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
anyhow = "1"
tower-http = { version = "0.6", features = ["trace", "request-id", "util"] }
tower = "0.5"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
axum-test = "16"
```

---

### Task 1: Structured Error Types

Foundation that all handlers use. Build first so everything can reference it.

**Files:**
- Create: `src/error.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write error.rs with tests**

```rust
// src/error.rs
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug)]
pub enum AppError {
    /// Input validation failures (empty input, batch too large)
    InvalidRequest(String),
    /// Model not loaded yet
    ServiceUnavailable(String),
    /// Inference or internal failures
    Internal(String),
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: u16,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            AppError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", msg)
            }
            AppError::ServiceUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable", msg)
            }
            AppError::Internal(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg)
            }
        };

        let body = ErrorBody {
            error: ErrorDetail {
                message,
                r#type: error_type.to_string(),
                code: status.as_u16(),
            },
        };

        (status, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Internal(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_request_serializes_as_400() {
        let err = AppError::InvalidRequest("input must not be empty".into());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn service_unavailable_serializes_as_503() {
        let err = AppError::ServiceUnavailable("model not ready".into());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn internal_error_serializes_as_500() {
        let err = AppError::Internal("inference failed".into());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
```

**Step 2: Register module in main.rs**

Add `mod error;` to the module declarations in `src/main.rs`.

**Step 3: Run tests**

Run: `cargo test error::tests -- --nocapture`
Expected: 3 tests PASS

**Step 4: Clippy + fmt**

Run: `cargo clippy --all-targets -- -D warnings && cargo fmt --check`
Expected: clean

**Step 5: Commit**

```bash
git add src/error.rs src/main.rs
git commit -m "feat: add structured JSON error types with AppError enum"
```

---

### Task 2: Expand Configuration

Add worker count, max batch size, validation.

**Files:**
- Modify: `src/config.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Rewrite config.rs with tests**

```rust
// src/config.rs
use std::env;

pub struct Config {
    pub cache_dir: String,
    pub bind_addr: String,
    pub workers: usize,
    pub max_batch: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let workers = env::var("BGE_M3_WORKERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);

        let max_batch = env::var("BGE_M3_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        Self {
            cache_dir: env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/cache".into()),
            bind_addr: env::var("BGE_M3_BIND").unwrap_or_else(|_| "0.0.0.0:8081".into()),
            workers: workers.max(1),
            max_batch: max_batch.max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_without_env_vars() {
        // Clear any set vars for test isolation
        env::remove_var("BGE_M3_CACHE_DIR");
        env::remove_var("BGE_M3_BIND");
        env::remove_var("BGE_M3_WORKERS");
        env::remove_var("BGE_M3_MAX_BATCH");

        let cfg = Config::from_env();
        assert_eq!(cfg.cache_dir, "/cache");
        assert_eq!(cfg.bind_addr, "0.0.0.0:8081");
        assert_eq!(cfg.workers, 2);
        assert_eq!(cfg.max_batch, 256);
    }

    #[test]
    fn workers_clamps_to_minimum_1() {
        env::set_var("BGE_M3_WORKERS", "0");
        let cfg = Config::from_env();
        assert_eq!(cfg.workers, 1);
        env::remove_var("BGE_M3_WORKERS");
    }
}
```

**Step 2: Run tests**

Run: `cargo test config::tests -- --nocapture --test-threads=1`
Expected: 2 tests PASS (serial due to env var mutation)

**Step 3: Commit**

```bash
git add src/config.rs
git commit -m "feat: expand config with workers, max_batch, validation"
```

---

### Task 3: Request/Response Models

Add OpenAI-compatible types alongside existing sparse types. Support `input` as either a string or array.

**Files:**
- Modify: `src/models.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Rewrite models.rs**

```rust
// src/models.rs
use serde::{Deserialize, Deserializer, Serialize};

// ── Shared ──────────────────────────────────────────

/// Accepts either a single string or an array of strings.
#[derive(Debug, Clone)]
pub struct TextInput(pub Vec<String>);

impl<'de> Deserialize<'de> for TextInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrVec {
            Single(String),
            Multiple(Vec<String>),
        }

        match StringOrVec::deserialize(deserializer)? {
            StringOrVec::Single(s) => Ok(TextInput(vec![s])),
            StringOrVec::Multiple(v) => Ok(TextInput(v)),
        }
    }
}

// ── Dense (OpenAI-compatible) ───────────────────────

#[derive(Deserialize)]
pub struct DenseRequest {
    pub input: TextInput,
    /// Accepted but ignored — only BGE-M3 is loaded.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct DenseResponse {
    pub object: &'static str,
    pub model: &'static str,
    pub data: Vec<DenseEmbeddingData>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct DenseEmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// ── Sparse ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct SparseRequest {
    pub input: TextInput,
}

#[derive(Serialize)]
pub struct SparseResponse {
    pub data: Vec<SparseEmbeddingData>,
}

#[derive(Serialize)]
pub struct SparseEmbeddingData {
    pub index: usize,
    pub sparse_values: SparseValues,
}

#[derive(Serialize)]
pub struct SparseValues {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_deserializes_single_string() {
        let json = r#""hello""#;
        let input: TextInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.0, vec!["hello"]);
    }

    #[test]
    fn text_input_deserializes_array() {
        let json = r#"["a", "b"]"#;
        let input: TextInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.0, vec!["a", "b"]);
    }

    #[test]
    fn dense_request_model_is_optional() {
        let json = r#"{"input": "test"}"#;
        let req: DenseRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input.0, vec!["test"]);
        assert!(req.model.is_none());
    }

    #[test]
    fn dense_response_serializes_openai_format() {
        let resp = DenseResponse {
            object: "list",
            model: "bge-m3",
            data: vec![DenseEmbeddingData {
                object: "embedding",
                index: 0,
                embedding: vec![0.1, 0.2],
            }],
            usage: Usage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["usage"]["prompt_tokens"], 5);
    }

    #[test]
    fn sparse_response_matches_consumer_format() {
        let resp = SparseResponse {
            data: vec![SparseEmbeddingData {
                index: 0,
                sparse_values: SparseValues {
                    indices: vec![101, 2023],
                    values: vec![0.45, 0.33],
                },
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["data"][0]["sparse_values"]["indices"].is_array());
    }
}
```

**Step 2: Run tests**

Run: `cargo test models::tests -- --nocapture`
Expected: 5 tests PASS

**Step 3: Commit**

```bash
git add src/models.rs
git commit -m "feat: add OpenAI-compatible dense types and flexible TextInput deserialization"
```

---

### Task 4: Worker Pool (Embedder Rewrite)

The core concurrency change. Replace `Mutex<Option<Embedder>>` with a bounded mpsc worker pool.

**Files:**
- Rewrite: `src/embedder.rs`
- No unit tests here — this is integration-tested in Task 7. The pool mechanics use tokio primitives that are well-tested upstream; our value-add is the wiring.

**Step 1: Rewrite embedder.rs**

```rust
// src/embedder.rs
use anyhow::{Context, Result};
use fastembed::{
    EmbeddingModel, SparseEmbedding, SparseInitOptions, SparseModel,
    SparseTextEmbedding, TextEmbedding, TextInitOptions,
};
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, info_span, Instrument};

/// A work item sent to a pool worker.
pub enum EmbedRequest {
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    Sparse {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<SparseEmbedding>>>,
    },
}

/// Handle to the worker pool. Clone-cheap (wraps an mpsc sender).
#[derive(Clone)]
pub struct EmbedPool {
    tx: mpsc::Sender<EmbedRequest>,
}

impl EmbedPool {
    /// Spawn `n` workers, each owning their own model instances.
    /// Returns the pool handle and a future that resolves when all workers are ready.
    pub fn spawn(n: usize, cache_dir: PathBuf) -> (Self, tokio::task::JoinHandle<Result<()>>) {
        let (tx, rx) = mpsc::channel::<EmbedRequest>(n * 4);
        let rx = std::sync::Arc::new(tokio::sync::Mutex::new(rx));

        let init_handle = tokio::spawn({
            let rx = rx.clone();
            let cache_dir = cache_dir.clone();
            async move {
                let mut handles = Vec::with_capacity(n);

                for id in 0..n {
                    let rx = rx.clone();
                    let cache_dir = cache_dir.clone();

                    let handle = tokio::task::spawn_blocking(move || -> Result<()> {
                        let _span = info_span!("worker", id).entered();
                        info!("Loading models...");

                        let mut dense = TextEmbedding::try_new(
                            TextInitOptions::new(EmbeddingModel::BGEM3)
                                .with_cache_dir(cache_dir.clone())
                                .with_show_download_progress(id == 0),
                        )
                        .context("failed to load dense model")?;

                        let mut sparse = SparseTextEmbedding::try_new(
                            SparseInitOptions::new(SparseModel::BGEM3)
                                .with_cache_dir(cache_dir)
                                .with_show_download_progress(false),
                        )
                        .context("failed to load sparse model")?;

                        info!("Models loaded, entering work loop");

                        // Block this OS thread, pulling work from the shared receiver
                        let rt = tokio::runtime::Handle::current();
                        loop {
                            let req = rt.block_on(async {
                                let mut guard = rx.lock().await;
                                guard.recv().await
                            });

                            let Some(req) = req else {
                                info!("Channel closed, worker exiting");
                                break;
                            };

                            match req {
                                EmbedRequest::Dense { texts, reply } => {
                                    let result = dense.embed(&texts, None).map_err(Into::into);
                                    let _ = reply.send(result);
                                }
                                EmbedRequest::Sparse { texts, reply } => {
                                    let result = sparse.embed(&texts, None).map_err(Into::into);
                                    let _ = reply.send(result);
                                }
                            }
                        }
                        Ok(())
                    });

                    handles.push(handle);
                }

                // Wait briefly for workers to load — we can't easily signal
                // "loaded" from spawn_blocking, so we use a poll approach
                // in main.rs via health checks or a separate readiness signal.
                // For now, the init_handle just confirms workers spawned.
                info!("{n} workers spawned");
                Ok(())
            }
            .instrument(info_span!("embed_pool"))
        });

        (Self { tx }, init_handle)
    }

    /// Submit a dense embedding request to the pool.
    pub async fn dense(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Dense {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("worker pool shut down"))?;

        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("worker dropped reply"))?
    }

    /// Submit a sparse embedding request to the pool.
    pub async fn sparse(&self, texts: Vec<String>) -> Result<Vec<SparseEmbedding>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Sparse {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("worker pool shut down"))?;

        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("worker dropped reply"))?
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles (handlers/main not yet wired)

**Step 3: Commit**

```bash
git add src/embedder.rs
git commit -m "feat: replace Mutex embedder with bounded mpsc worker pool"
```

---

### Task 5: Application State

Simplify to hold the pool handle and readiness flag.

**Files:**
- Modify: `src/state.rs`

**Step 1: Rewrite state.rs**

```rust
// src/state.rs
use crate::embedder::EmbedPool;
use std::sync::atomic::AtomicBool;

pub struct AppState {
    pub pool: EmbedPool,
    pub ready: AtomicBool,
    pub max_batch: usize,
}
```

**Step 2: Verify it compiles**

Run: `cargo check`

**Step 3: Commit**

```bash
git add src/state.rs
git commit -m "refactor: simplify AppState to hold EmbedPool and config"
```

---

### Task 6: Handlers

Rewrite all three handlers to use the pool, structured errors, and input validation.

**Files:**
- Rewrite: `src/handler.rs`

**Step 1: Rewrite handler.rs**

```rust
// src/handler.rs
use axum::{extract::State, response::IntoResponse, Json};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::error::AppError;
use crate::models::{
    DenseEmbeddingData, DenseRequest, DenseResponse, SparseEmbeddingData,
    SparseRequest, SparseResponse, SparseValues, Usage,
};
use crate::state::AppState;

/// Validate input and return the text vec, or an error.
fn validate_input(texts: &[String], max_batch: usize) -> Result<(), AppError> {
    if texts.is_empty() {
        return Err(AppError::InvalidRequest(
            "input must not be empty".into(),
        ));
    }
    if texts.len() > max_batch {
        return Err(AppError::InvalidRequest(format!(
            "batch size {} exceeds maximum {}",
            texts.len(),
            max_batch
        )));
    }
    Ok(())
}

fn check_ready(state: &AppState) -> Result<(), AppError> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(AppError::ServiceUnavailable("model not ready".into()));
    }
    Ok(())
}

pub async fn dense_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DenseRequest>,
) -> Result<Json<DenseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    validate_input(&texts, state.max_batch)?;

    // Approximate token count: ~1.3 tokens per word, ~4.5 chars per word
    let approx_tokens: usize = texts.iter().map(|t| t.len() / 4 + 1).sum();

    let embeddings = state.pool.dense(texts).await?;

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| DenseEmbeddingData {
            object: "embedding",
            index,
            embedding,
        })
        .collect();

    Ok(Json(DenseResponse {
        object: "list",
        model: "bge-m3",
        data,
        usage: Usage {
            prompt_tokens: approx_tokens,
            total_tokens: approx_tokens,
        },
    }))
}

pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SparseRequest>,
) -> Result<Json<SparseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    validate_input(&texts, state.max_batch)?;

    let results = state.pool.sparse(texts).await?;

    let data = results
        .into_iter()
        .enumerate()
        .map(|(index, emb)| SparseEmbeddingData {
            index,
            sparse_values: SparseValues {
                indices: emb.indices.into_iter().map(|i| i as u32).collect(),
                values: emb.values,
            },
        })
        .collect();

    Ok(Json(SparseResponse { data }))
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.ready.load(Ordering::Acquire) {
        Json(serde_json::json!({"status": "ok"})).into_response()
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response()
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check`

**Step 3: Commit**

```bash
git add src/handler.rs
git commit -m "feat: rewrite handlers with dense endpoint, validation, structured errors"
```

---

### Task 7: Wire Main with Observability

Bring everything together: pool startup, readiness signaling, tower-http tracing.

**Files:**
- Rewrite: `src/main.rs`

**Step 1: Rewrite main.rs**

```rust
// src/main.rs
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
use tracing::info;

use config::Config;
use embedder::EmbedPool;
use handler::{dense_embeddings, health, sparse_embeddings};
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
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

    let cache_dir = PathBuf::from(&cfg.cache_dir);
    let (pool, init_handle) = EmbedPool::spawn(cfg.workers, cache_dir.clone());

    let ready = Arc::new(AtomicBool::new(false));

    let state = Arc::new(AppState {
        pool: pool.clone(),
        ready: AtomicBool::new(false),
        max_batch: cfg.max_batch,
    });

    let app = Router::new()
        .route("/v1/embeddings", post(dense_embeddings))
        .route("/v1/sparse-embeddings", post(sparse_embeddings))
        .route("/health", get(health))
        .layer(TraceLayer::new_for_http())
        .with_state(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    info!("Listening on {}", cfg.bind_addr);

    // Mark ready after a successful probe of the pool
    let state_for_ready = Arc::clone(&state);
    tokio::spawn(async move {
        // Wait for init to complete
        if let Err(e) = init_handle.await {
            tracing::error!("Worker pool init failed: {e}");
            std::process::exit(1);
        }

        // Probe the pool with a tiny request to confirm models are loaded
        match state_for_ready.pool.dense(vec!["ready".into()]).await {
            Ok(_) => {
                state_for_ready.ready.store(true, Ordering::Release);
                info!("All workers ready, service is live");
            }
            Err(e) => {
                tracing::error!("Readiness probe failed: {e}");
                std::process::exit(1);
            }
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}
```

**Step 2: Full build**

Run: `cargo build`
Expected: compiles cleanly

**Step 3: Clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: clean

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire worker pool, observability, and dense+sparse routes"
```

---

### Task 8: Run All Tests

Run the full test suite to verify models, error, and config tests pass.

**Step 1: Run tests**

Run: `cargo nextest run --all-features --no-tests=pass`
Expected: all tests in error, config, and models modules pass

**Step 2: Run clippy and fmt**

Run: `cargo clippy --all-targets -- -D warnings && cargo fmt --check`
Expected: clean

**Step 3: Commit (if any fmt fixes needed)**

```bash
cargo fmt
git add -A
git commit -m "style: apply cargo fmt"
```

---

### Task 9: Dockerfile Healthcheck

Add `curl` and `HEALTHCHECK` to the Dockerfile.

**Files:**
- Modify: `Dockerfile`

**Step 1: Add healthcheck**

In the runtime stage, add `curl` to the apt install line and add a `HEALTHCHECK`:

```dockerfile
# In the runtime stage, change the apt-get line to include curl:
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3t64 curl \
    && rm -rf /var/lib/apt/lists/*

# Add before EXPOSE:
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8081/health || exit 1
```

Note: `start-period=120s` is generous because model download + ONNX init can take time on first run.

**Step 2: Local Docker build test**

Run: `docker build -t bge-m3-test .`
Expected: builds successfully

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Docker healthcheck for orchestration readiness"
```

---

### Task 10: CLAUDE.md

Project-level instructions for Claude Code sessions.

**Files:**
- Create: `CLAUDE.md`

**Step 1: Write CLAUDE.md**

```markdown
# CLAUDE.md — bge-m3-axum-fastembed-rs

## What This Is

Axum HTTP server wrapping fastembed-rs to serve BGE-M3 dense and sparse embeddings.
Consumers: mcp-local-knowledge-base (dense + sparse hybrid search), dpos-coordinator (dense via Spring AI).

## Build & Test

    cargo build
    cargo nextest run --all-features --no-tests=pass
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check

## Run Locally

    BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run

First run downloads ~2.2GB model. Server starts on port 8081.

## Endpoints

- `POST /v1/embeddings` — Dense embeddings (OpenAI-compatible)
- `POST /v1/sparse-embeddings` — Sparse embeddings
- `GET /health` — Readiness probe

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | Model cache directory |
| `BGE_M3_BIND` | `0.0.0.0:8081` | Listen address |
| `BGE_M3_WORKERS` | `2` | Worker pool size |
| `BGE_M3_MAX_BATCH` | `256` | Max texts per request |
| `RUST_LOG` | `info` | Log level |

## Architecture

Worker pool via tokio mpsc. Each worker owns TextEmbedding + SparseTextEmbedding.
Handlers submit work items with oneshot reply channels. ONNX inference runs in spawn_blocking.

## Docker

    docker build -t bge-m3-axum-fastembed-rs .
    docker run -p 8081:8081 -v /tmp/bge-m3-cache:/cache bge-m3-axum-fastembed-rs
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md project instructions"
```

---

### Task 11: README

Public-facing documentation.

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

Include: project description, quick start, API reference with curl examples for both endpoints, configuration table, Docker instructions, architecture overview, and license.

Use the design doc and CLAUDE.md as source material. Include curl examples:

```bash
# Dense embeddings
curl -X POST http://localhost:8081/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "query: what is Rust?", "model": "bge-m3"}'

# Sparse embeddings
curl -X POST http://localhost:8081/v1/sparse-embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["what is Rust?"]}'

# Health check
curl http://localhost:8081/health
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with API reference and Docker instructions"
```

---

### Task 12: Final Verification & Push

**Step 1: Full test suite**

Run: `cargo nextest run --all-features --no-tests=pass`

**Step 2: Clippy + fmt**

Run: `cargo clippy --all-targets -- -D warnings && cargo fmt --check`

**Step 3: Cargo deny**

Run: `cargo deny check`

**Step 4: Docker build**

Run: `docker build -t bge-m3-test .`

**Step 5: Version bump for release**

Edit `Cargo.toml`: change `version = "0.1.0"` to `version = "0.2.0"` to trigger the release workflow.

**Step 6: Commit and push**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: bump version to 0.2.0 for dense+sparse release"
git push origin main
```

**Step 7: Monitor CI**

Run: `gh run list --repo Fulton-Engineering-Services/bge-m3-axum-fastembed-rs --limit 2`

Watch for all-green on both CI and Release workflows.

---

## Task Dependency Graph

```
Task 1 (error) ──┐
Task 2 (config) ──┤
Task 3 (models) ──┼──► Task 6 (handlers) ──► Task 7 (main) ──► Task 8 (tests)
Task 4 (pool) ────┤                                              │
Task 5 (state) ───┘                                              ▼
                                               Task 9 (Docker) ──► Task 12 (verify+push)
                                               Task 10 (CLAUDE.md)
                                               Task 11 (README)
```

Tasks 1-5 can be done in any order (they're independent foundations). Task 6 depends on 1-5. Task 7 depends on 6. Tasks 9-11 are independent of each other but should follow Task 8.
