use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::{atomic::Ordering, Arc};
use std::time::Instant;

use crate::error::AppError;
use crate::models::{
    DenseEmbeddingData, DenseRequest, DenseResponse, DualEmbeddingData, DualRequest, DualResponse,
    ModelEntry, ModelsResponse, SparseEmbeddingData, SparseRequest, SparseResponse, SparseValues,
    Usage,
};
use crate::state::{AppState, ProbeStatus};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Maximum characters allowed per individual input string (SEC-3).
const MAX_STRING_CHARS: usize = 32_768;

/// Validates a batch of input texts against size and length constraints.
///
/// Returns [`AppError::InvalidRequest`] if:
/// - `texts` is empty
/// - `texts.len() > max_batch`
/// - any individual text exceeds [`MAX_STRING_CHARS`] characters
fn validate_input(texts: &[String], max_batch: usize) -> Result<(), AppError> {
    if texts.is_empty() {
        return Err(AppError::InvalidRequest(
            "input must not be empty".to_string(),
        ));
    }
    if texts.len() > max_batch {
        return Err(AppError::InvalidRequest(format!(
            "batch size {} exceeds maximum {}",
            texts.len(),
            max_batch
        )));
    }
    for (i, text) in texts.iter().enumerate() {
        let char_count = text.chars().count();
        if char_count > MAX_STRING_CHARS {
            return Err(AppError::InvalidRequest(format!(
                "input[{i}] length {char_count} exceeds maximum {MAX_STRING_CHARS} characters"
            )));
        }
    }
    Ok(())
}

/// Checks whether the service is ready to handle embedding requests.
fn check_ready(state: &AppState) -> Result<(), AppError> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(AppError::ServiceUnavailable("model not ready".to_string()));
    }
    if state.pool.live_worker_count() == 0 {
        return Err(AppError::ServiceUnavailable(
            "no workers available".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public handlers
// ---------------------------------------------------------------------------

#[tracing::instrument(skip(state, req), fields(batch_size, prompt_tokens, chunks, max_chunk_seq, tokenize_ms, inference_ms, total_ms))]
pub async fn dense_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DenseRequest>,
) -> Result<Json<DenseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    drop(req.model);
    validate_input(&texts, state.max_batch)?;
    let batch_size = texts.len();
    tracing::Span::current().record("batch_size", batch_size);

    let prompt_tokens: usize = texts.iter().map(|t| t.chars().count() / 4 + 1).sum();
    tracing::Span::current().record("prompt_tokens", prompt_tokens);

    let t0 = Instant::now();

    // Acquire a concurrency permit before dispatching to the worker pool.
    // This is released on drop when the handler returns (success or error).
    let _permit = Arc::clone(&state.request_permits)
        .acquire_owned()
        .await
        .expect("request semaphore is never closed");

    let (embeddings, embed_stats) = state.pool.dense(texts).await?;

    let total_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
    tracing::Span::current()
        .record("chunks", embed_stats.chunks)
        .record("max_chunk_seq", embed_stats.max_chunk_seq)
        .record("tokenize_ms", embed_stats.tokenize_ms)
        .record("inference_ms", embed_stats.inference_ms)
        .record("total_ms", total_ms);
    tracing::info!(
        route = "dense",
        batch_size,
        prompt_tokens,
        chunks = embed_stats.chunks,
        max_chunk_seq = embed_stats.max_chunk_seq,
        total_token_positions = embed_stats.total_token_positions,
        tokenize_ms = embed_stats.tokenize_ms,
        inference_ms = embed_stats.inference_ms,
        total_ms,
        "embedding request complete"
    );

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
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

#[allow(clippy::cast_possible_truncation)]
#[tracing::instrument(skip(state, req), fields(batch_size, chunks, max_chunk_seq, tokenize_ms, inference_ms, total_ms))]
pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SparseRequest>,
) -> Result<Json<SparseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    validate_input(&texts, state.max_batch)?;
    let batch_size = texts.len();
    tracing::Span::current().record("batch_size", batch_size);

    let t0 = Instant::now();

    let _permit = Arc::clone(&state.request_permits)
        .acquire_owned()
        .await
        .expect("request semaphore is never closed");

    let (embeddings, embed_stats) = state.pool.sparse(texts).await?;

    let total_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
    tracing::Span::current()
        .record("chunks", embed_stats.chunks)
        .record("max_chunk_seq", embed_stats.max_chunk_seq)
        .record("tokenize_ms", embed_stats.tokenize_ms)
        .record("inference_ms", embed_stats.inference_ms)
        .record("total_ms", total_ms);
    tracing::info!(
        route = "sparse",
        batch_size,
        chunks = embed_stats.chunks,
        max_chunk_seq = embed_stats.max_chunk_seq,
        total_token_positions = embed_stats.total_token_positions,
        tokenize_ms = embed_stats.tokenize_ms,
        inference_ms = embed_stats.inference_ms,
        total_ms,
        "embedding request complete"
    );

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, emb)| SparseEmbeddingData {
            index,
            sparse_values: SparseValues {
                indices: emb.indices.iter().map(|i| *i as u32).collect(),
                values: emb.values,
            },
        })
        .collect();

    Ok(Json(SparseResponse { data }))
}

#[allow(clippy::cast_possible_truncation)]
#[tracing::instrument(skip(state, req), fields(batch_size, prompt_tokens, chunks, max_chunk_seq, tokenize_ms, inference_ms, total_ms))]
pub async fn both_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DualRequest>,
) -> Result<Json<DualResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    drop(req.model);
    validate_input(&texts, state.max_batch)?;
    let batch_size = texts.len();
    tracing::Span::current().record("batch_size", batch_size);

    let prompt_tokens: usize = texts.iter().map(|t| t.chars().count() / 4 + 1).sum();
    tracing::Span::current().record("prompt_tokens", prompt_tokens);

    let t0 = Instant::now();

    let _permit = Arc::clone(&state.request_permits)
        .acquire_owned()
        .await
        .expect("request semaphore is never closed");

    let (pairs, embed_stats) = state.pool.both(texts).await?;

    let total_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
    tracing::Span::current()
        .record("chunks", embed_stats.chunks)
        .record("max_chunk_seq", embed_stats.max_chunk_seq)
        .record("tokenize_ms", embed_stats.tokenize_ms)
        .record("inference_ms", embed_stats.inference_ms)
        .record("total_ms", total_ms);
    tracing::info!(
        route = "both",
        batch_size,
        prompt_tokens,
        chunks = embed_stats.chunks,
        max_chunk_seq = embed_stats.max_chunk_seq,
        total_token_positions = embed_stats.total_token_positions,
        tokenize_ms = embed_stats.tokenize_ms,
        inference_ms = embed_stats.inference_ms,
        total_ms,
        "embedding request complete"
    );

    let data = pairs
        .into_iter()
        .enumerate()
        .map(|(index, pair)| DualEmbeddingData {
            index,
            embedding: pair.dense,
            sparse_values: SparseValues {
                indices: pair.sparse.indices.iter().map(|i| *i as u32).collect(),
                values: pair.sparse.values,
            },
        })
        .collect();

    Ok(Json(DualResponse {
        object: "list",
        model: "bge-m3",
        data,
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ready = state.ready.load(Ordering::Acquire);
    let live = state.pool.live_worker_count();
    let loaded = state.pool.loaded_worker_count();
    let total = state.total_workers;

    if !ready {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response();
    }

    if live == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "fail",
                "workers": { "live": live, "total": total }
            })),
        )
            .into_response();
    }

    if loaded == 0 {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "idle",
                "workers": { "live": live, "total": total }
            })),
        )
            .into_response();
    }

    let status = if live < total { "warn" } else { "ok" };

    // Read the live cost model and probe status atomically.
    let cm = state.cost_model.load();
    let probe_status = ProbeStatus::from_u8(state.probe_status.load(Ordering::Acquire)).as_str();

    let mut tuning = serde_json::json!({
        "a_bytes_per_token": cm.a,
        "b_bytes_per_token_sq": cm.b,
        "max_workspace_bytes": cm.max_workspace_bytes,
        "probe_status": probe_status,
    });

    // Add static memory fields when available (written before probe starts).
    if let Some(ti) = state.tuning.get() {
        tuning["memory_source"] = serde_json::Value::String(ti.memory_source.clone());
        tuning["available_bytes"] =
            serde_json::Value::Number(serde_json::Number::from(ti.available_bytes));
        tuning["model_rss_bytes_per_worker"] =
            serde_json::Value::Number(serde_json::Number::from(ti.model_rss_bytes_per_worker));
    }

    let body = serde_json::json!({
        "status": status,
        "workers": { "live": live, "total": total },
        "max_seq_length": state.max_seq_length,
        "tuning": tuning,
    });

    (StatusCode::OK, Json(body)).into_response()
}

/// Returns an OpenAI-compatible models list confirming BGE-M3 is resident.
pub async fn models(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelEntry {
            id: "bge-m3",
            object: "model",
        }],
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binpack::CostModel;
    use crate::embedder::EmbedPool;
    use arc_swap::ArcSwap;
    use axum::body::to_bytes;
    use std::sync::atomic::{AtomicBool, AtomicU8};
    use tokio::sync::Semaphore;

    fn make_state(ready: bool, max_batch: usize) -> Arc<AppState> {
        Arc::new(AppState {
            pool: EmbedPool::closed_for_test(),
            ready: AtomicBool::new(ready),
            max_batch,
            total_workers: 2,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        })
    }

    // --- validate_input ---

    #[test]
    fn validate_input_rejects_empty() {
        let result = validate_input(&[], 10);
        assert!(
            matches!(result, Err(AppError::InvalidRequest(msg)) if msg == "input must not be empty")
        );
    }

    #[test]
    fn validate_input_rejects_over_batch() {
        let texts: Vec<String> = (0..5).map(|i| format!("text {i}")).collect();
        let result = validate_input(&texts, 3);
        assert!(
            matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains('5') && msg.contains('3'))
        );
    }

    #[test]
    fn validate_input_accepts_at_limit() {
        let texts: Vec<String> = (0..3).map(|i| format!("text {i}")).collect();
        assert!(validate_input(&texts, 3).is_ok());
    }

    #[test]
    fn validate_input_accepts_single() {
        let texts = vec!["hello".to_string()];
        assert!(validate_input(&texts, 256).is_ok());
    }

    // --- check_ready ---

    #[test]
    fn check_ready_returns_err_when_not_ready() {
        let state = make_state(false, 10);
        let result = check_ready(&state);
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "model not ready")
        );
    }

    #[test]
    fn check_ready_returns_err_when_pool_dead() {
        let state = make_state(true, 10);
        let result = check_ready(&state);
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
        );
    }

    // --- health handler ---

    #[tokio::test]
    async fn health_returns_fail_when_ready_but_pool_dead() {
        let state = make_state(true, 256);
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response status should be parseable");
        assert_eq!(body["status"], "fail");
        assert_eq!(body["workers"]["live"], 0);
        assert_eq!(body["workers"]["total"], 2);
    }

    #[tokio::test]
    async fn health_returns_idle_when_models_unloaded() {
        let state = Arc::new(AppState {
            pool: EmbedPool::idle_for_test(),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response status should be parseable");
        assert_eq!(body["status"], "idle");
        assert_eq!(body["workers"]["live"], 1);
        assert_eq!(body["workers"]["total"], 1);
    }

    #[tokio::test]
    async fn health_returns_503_when_not_ready() {
        let state = make_state(false, 10);
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response status should be parseable");
        assert_eq!(body["status"], "loading");
    }

    #[tokio::test]
    async fn health_ok_includes_max_seq_length() {
        use crate::state::TuningInfo;
        use crate::sysinfo::{MemoryReading, MemorySource};

        let mem = MemoryReading {
            available_bytes: 8_000_000_000,
            source: MemorySource::CgroupV2,
        };
        let tuning = TuningInfo::new(&mem, 500_000_000, 22_000_000_000, 78.5);

        let tuning_lock = std::sync::OnceLock::new();
        let _ = tuning_lock.set(tuning);
        let fitted_cm = CostModel {
            a: 18_432.0,
            b: 6.2,
            max_workspace_bytes: 1_073_741_824,
        };
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: tuning_lock,
            cost_model: Arc::new(ArcSwap::from_pointee(fitted_cm)),
            probe_status: AtomicU8::new(ProbeStatus::Complete as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["max_seq_length"], 8192);
        assert!(
            body["tuning"].is_object(),
            "tuning should always be present"
        );
        assert_eq!(body["tuning"]["probe_status"], "complete");
        assert_eq!(body["tuning"]["a_bytes_per_token"], 18_432.0);
        assert_eq!(body["tuning"]["b_bytes_per_token_sq"], 6.2);
    }

    // --- dense_embeddings handler (validation paths only) ---

    #[tokio::test]
    async fn dense_embeddings_rejects_when_not_ready() {
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = DenseRequest {
            input: TextInput(vec!["hello".to_string()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    #[tokio::test]
    async fn dense_embeddings_rejects_when_pool_dead() {
        use crate::models::TextInput;
        let state = make_state(true, 256);
        let req = DenseRequest {
            input: TextInput(vec!["hello".to_string()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
        );
    }

    #[tokio::test]
    async fn dense_embeddings_rejects_empty_input() {
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = DenseRequest {
            input: TextInput(vec![]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    #[tokio::test]
    async fn dense_embeddings_rejects_over_batch() {
        use crate::models::TextInput;
        let state = make_state(false, 2);
        let req = DenseRequest {
            input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    // --- sparse_embeddings handler (validation paths only) ---

    #[tokio::test]
    async fn sparse_embeddings_rejects_when_not_ready() {
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = SparseRequest {
            input: TextInput(vec!["hello".to_string()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    #[tokio::test]
    async fn sparse_embeddings_rejects_when_pool_dead() {
        use crate::models::TextInput;
        let state = make_state(true, 256);
        let req = SparseRequest {
            input: TextInput(vec!["hello".to_string()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
        );
    }

    #[tokio::test]
    async fn sparse_embeddings_rejects_empty_input() {
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = SparseRequest {
            input: TextInput(vec![]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    #[tokio::test]
    async fn models_handler_returns_bge_m3_entry() {
        let state = make_state(true, 256);
        let response = models(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body readable");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("valid json");
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "bge-m3");
        assert_eq!(body["data"][0]["object"], "model");
    }

    #[tokio::test]
    async fn sparse_embeddings_rejects_over_batch() {
        use crate::models::TextInput;
        let state = make_state(false, 2);
        let req = SparseRequest {
            input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    // --- handler validation with ready pool (TST-5) ---

    #[tokio::test]
    async fn dense_embeddings_returns_invalid_request_for_empty_input_when_ready() {
        use crate::models::TextInput;
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = DenseRequest {
            input: TextInput(vec![]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("empty")),
            "expected InvalidRequest for empty input, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn dense_embeddings_returns_invalid_request_for_over_batch_when_ready() {
        use crate::models::TextInput;
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 2,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = DenseRequest {
            input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
            "expected InvalidRequest for over-batch, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn sparse_embeddings_returns_invalid_request_for_empty_input_when_ready() {
        use crate::models::TextInput;
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = SparseRequest {
            input: TextInput(vec![]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("empty")),
            "expected InvalidRequest for empty input, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn sparse_embeddings_returns_invalid_request_for_over_batch_when_ready() {
        use crate::models::TextInput;
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 2,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = SparseRequest {
            input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
            "expected InvalidRequest for over-batch, got: {result:?}"
        );
    }

    // --- per-string length validation ---

    #[test]
    fn validate_input_rejects_oversized_string() {
        let long = "x".repeat(super::MAX_STRING_CHARS + 1);
        let texts = vec![long];
        let result = validate_input(&texts, 256);
        assert!(
            matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains("exceeds maximum"))
        );
    }

    #[test]
    fn validate_input_accepts_at_char_limit() {
        let at_limit = "x".repeat(super::MAX_STRING_CHARS);
        let texts = vec![at_limit];
        assert!(
            validate_input(&texts, 256).is_ok(),
            "string exactly at MAX_STRING_CHARS should be accepted"
        );
    }

    // --- happy-path tests (using fixture pool) ---

    #[tokio::test]
    async fn dense_embeddings_returns_correct_shape() {
        use crate::models::TextInput;
        let fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(fixture, vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = DenseRequest {
            input: TextInput(vec!["hello".into(), "world".into()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
        let Json(resp) = result.expect("dense_embeddings should succeed");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2, 0.3]);
        assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.model, "bge-m3");
    }

    // --- both_embeddings handler ---

    #[tokio::test]
    async fn both_embeddings_rejects_when_not_ready() {
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = DualRequest {
            input: TextInput(vec!["hello".to_string()]),
            model: None,
        };
        let result = both_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    #[tokio::test]
    async fn both_embeddings_rejects_when_pool_dead() {
        use crate::models::TextInput;
        let state = make_state(true, 256);
        let req = DualRequest {
            input: TextInput(vec!["hello".to_string()]),
            model: None,
        };
        let result = both_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
        );
    }

    #[tokio::test]
    async fn both_embeddings_returns_invalid_request_for_empty_input_when_ready() {
        use crate::models::TextInput;
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = DualRequest {
            input: TextInput(vec![]),
            model: None,
        };
        let result = both_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("empty")),
            "expected InvalidRequest for empty input, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn both_embeddings_returns_invalid_request_for_over_batch_when_ready() {
        use crate::models::TextInput;
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 2,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = DualRequest {
            input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
            model: None,
        };
        let result = both_embeddings(State(state), Json(req)).await;
        assert!(
            matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
            "expected InvalidRequest for over-batch, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn both_embeddings_returns_correct_shape() {
        use crate::models::TextInput;
        let dense_fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let sparse_fixture = vec![
            crate::embedder::SparseEmbedding {
                indices: vec![42usize],
                values: vec![0.5f32],
            },
            crate::embedder::SparseEmbedding {
                indices: vec![100usize, 200usize],
                values: vec![0.7f32, 0.9f32],
            },
        ];
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(dense_fixture, sparse_fixture),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = DualRequest {
            input: TextInput(vec!["hello".into(), "world".into()]),
            model: None,
        };
        let result = both_embeddings(State(state), Json(req)).await;
        assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
        let Json(resp) = result.expect("both_embeddings should succeed");
        assert_eq!(resp.object, "list");
        assert_eq!(resp.model, "bge-m3");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2, 0.3]);
        assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
        assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
        assert_eq!(resp.data[1].sparse_values.indices, vec![100u32, 200u32]);
        assert_eq!(resp.data[1].sparse_values.values, vec![0.7f32, 0.9f32]);
    }

    #[tokio::test]
    async fn sparse_embeddings_returns_correct_shape() {
        use crate::models::TextInput;
        let sparse_fixture = vec![crate::embedder::SparseEmbedding {
            indices: vec![42usize],
            values: vec![0.5f32],
        }];
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], sparse_fixture),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
        });
        let req = SparseRequest {
            input: TextInput(vec!["hello".into()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(
            result.is_ok(),
            "expected Ok but got error from sparse handler"
        );
        let Json(resp) = result.expect("sparse_embeddings should succeed");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
        assert_eq!(resp.data[0].sparse_values.values, vec![0.5f32]);
    }

    // -----------------------------------------------------------------------
    // Permit-gating tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn dense_embeddings_blocks_when_no_permits_available() {
        use crate::models::TextInput;
        use std::time::Duration;
        // Semaphore with 0 permits — all requests must queue.
        let permits = Arc::new(Semaphore::new(0));
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![vec![0.1f32]], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
            cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
                CostModel::DEFAULT_MAX_WORKSPACE,
            ))),
            probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
            request_permits: Arc::clone(&permits),
        });
        let req = DenseRequest {
            input: TextInput(vec!["hello".into()]),
            model: None,
        };
        // Fire the request in a background task.
        let state_clone = Arc::clone(&state);
        let handle =
            tokio::spawn(async move { dense_embeddings(State(state_clone), Json(req)).await });
        // Give the task time to start and attempt permit acquisition.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !handle.is_finished(),
            "request should still be blocked on 0 permits"
        );
        // Release a permit — request should now complete.
        permits.add_permits(1);
        let result = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("request should complete after permit is released")
            .expect("task should not panic");
        assert!(
            result.is_ok(),
            "dense_embeddings should succeed once permitted"
        );
    }

    #[tokio::test]
    async fn request_permits_rises_after_probe_complete() {
        // Verify the semaphore protocol: starting at N-1 permits, add_permits(1)
        // after probe completion brings it to N.
        let n: usize = 7;
        let initial = n.saturating_sub(1).max(1);
        let permits = Arc::new(Semaphore::new(initial));
        assert_eq!(
            permits.available_permits(),
            initial,
            "initial permits should be cfg_workers - 1"
        );
        permits.add_permits(1);
        assert_eq!(
            permits.available_permits(),
            n,
            "after probe completes, permits should equal cfg_workers"
        );
    }

    // -----------------------------------------------------------------------
    // Worst-case budget invariant
    // -----------------------------------------------------------------------

    #[test]
    fn worst_case_peak_below_available_when_correctly_budgeted() {
        // Production config: 28 GB Fargate task, 7 fp16 workers, 0.7 safety.
        // With accurate model_rss_per_worker the formula must stay under the limit.
        use crate::embedder::OS_HEADROOM_BYTES;

        let available_bytes: usize = 28 * 1024 * 1024 * 1024; // 28 GB
        let cfg_workers: usize = 7;
        let model_rss_per_worker: usize = 1_080_000_000; // ~1.08 GB fp16
        let memory_safety_factor: f64 = 0.7;

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let total_workspace = available_bytes
            .saturating_sub(cfg_workers.saturating_mul(model_rss_per_worker))
            .saturating_sub(OS_HEADROOM_BYTES);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let per_worker_workspace =
            ((total_workspace as f64) * memory_safety_factor / (cfg_workers as f64)) as usize;

        let worst_case_peak = cfg_workers
            .saturating_mul(per_worker_workspace)
            .saturating_add(cfg_workers.saturating_mul(model_rss_per_worker))
            .saturating_add(OS_HEADROOM_BYTES);

        assert!(
            worst_case_peak < available_bytes,
            "worst_case_peak ({} MB) must be below available ({} MB)",
            worst_case_peak / (1024 * 1024),
            available_bytes / (1024 * 1024),
        );

        #[allow(clippy::cast_precision_loss)]
        let utilization_pct = (worst_case_peak as f64 / available_bytes as f64) * 100.0;
        assert!(
            utilization_pct < 90.0,
            "utilization_pct ({utilization_pct:.1}%) must be below the 90% WARN threshold"
        );
    }
}
