use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::{atomic::Ordering, Arc};

use crate::error::AppError;
use crate::models::{
    DenseEmbeddingData, DenseRequest, DenseResponse, ModelEntry, ModelsResponse,
    SparseEmbeddingData, SparseRequest, SparseResponse, SparseValues, Usage,
};
use crate::state::AppState;

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

#[tracing::instrument(skip(state, req), fields(batch_size))]
pub async fn dense_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DenseRequest>,
) -> Result<Json<DenseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    drop(req.model);
    validate_input(&texts, state.max_batch)?;
    tracing::Span::current().record("batch_size", texts.len());

    let prompt_tokens: usize = texts.iter().map(|t| t.chars().count() / 4 + 1).sum();

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
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

#[allow(clippy::cast_possible_truncation)]
#[tracing::instrument(skip(state, req), fields(batch_size))]
pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SparseRequest>,
) -> Result<Json<SparseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    validate_input(&texts, state.max_batch)?;
    tracing::Span::current().record("batch_size", texts.len());

    let embeddings = state.pool.sparse(texts).await?;

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

    let mut body = serde_json::json!({
        "status": status,
        "workers": { "live": live, "total": total },
        "max_seq_length": state.max_seq_length,
    });

    if let Some(tuning) = state.tuning.get() {
        body["tuning"] = serde_json::to_value(tuning)
            .unwrap_or(serde_json::Value::Null);
    }

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
    use crate::embedder::EmbedPool;
    use axum::body::to_bytes;
    use std::sync::atomic::AtomicBool;

    fn make_state(ready: bool, max_batch: usize) -> Arc<AppState> {
        Arc::new(AppState {
            pool: EmbedPool::closed_for_test(),
            ready: AtomicBool::new(ready),
            max_batch,
            total_workers: 2,
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
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
        use crate::binpack::CostModel;
        use crate::sysinfo::{MemoryReading, MemorySource};

        let cm = CostModel::conservative(1024 * 1024 * 1024);
        let mem = MemoryReading { available_bytes: 8_000_000_000, source: MemorySource::CgroupV2 };
        let tuning = TuningInfo::new(&cm, &mem, 500_000_000);

        let tuning_lock = std::sync::OnceLock::new();
        let _ = tuning_lock.set(tuning);
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], vec![]),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
            max_seq_length: 8192,
            tuning: tuning_lock,
        });
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["max_seq_length"], 8192);
        assert!(body["tuning"].is_object(), "tuning should be present");
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
        let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("body readable");
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
        });
        let req = SparseRequest {
            input: TextInput(vec!["hello".into()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(result.is_ok(), "expected Ok but got error from sparse handler");
        let Json(resp) = result.expect("sparse_embeddings should succeed");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
        assert_eq!(resp.data[0].sparse_values.values, vec![0.5f32]);
    }
}
