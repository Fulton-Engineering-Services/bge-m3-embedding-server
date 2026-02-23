use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::{atomic::Ordering, Arc};

use crate::error::AppError;
use crate::models::{
    DenseEmbeddingData, DenseRequest, DenseResponse, SparseEmbeddingData, SparseRequest,
    SparseResponse, SparseValues, Usage,
};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Maximum characters allowed per individual input string (SEC-3).
const MAX_STRING_CHARS: usize = 32_768;

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
    // model field is accepted for OpenAI API compatibility but ignored; BGE-M3 is always used.
    drop(req.model);
    validate_input(&texts, state.max_batch)?;
    tracing::Span::current().record("batch_size", texts.len());

    // Approximate token count using char length (COR-4). Char-based is more
    // accurate than byte-based for multi-byte UTF-8 inputs.
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

// SPLADE vocabulary indices from BGE-M3 tokenizer are bounded by vocab size (~30K tokens),
// well within u32::MAX. The cast is safe for this model.
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

    let status = if live < total { "warn" } else { "ok" };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": status,
            "workers": { "live": live, "total": total }
        })),
    )
        .into_response()
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
        // Use a closed channel — we only test validation/readiness paths
        // that return before reaching the pool (TST-5, COR-7).
        Arc::new(AppState {
            pool: EmbedPool::closed_for_test(),
            ready: AtomicBool::new(ready),
            max_batch,
            total_workers: 2, // default value for tests
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
        // make_state(true, ...) uses EmbedPool::closed_for_test() which has live_workers = 0
        let state = make_state(true, 10);
        let result = check_ready(&state);
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
        );
    }

    // --- health handler ---

    // Note: health "ok" and "warn" states are tested at the router level in src/main.rs tests (pkg-005)

    #[tokio::test]
    async fn health_returns_fail_when_ready_but_pool_dead() {
        // make_state(true, ...) uses EmbedPool::closed_for_test() which has live_workers = 0
        // ready=true + live=0 → 503 "fail"
        let state = make_state(true, 256);
        let response = health(State(state)).await.into_response();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "expected 503 when pool is dead"
        );
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
        // make_state(true, ...) has live_workers = 0 — check_ready returns ServiceUnavailable
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
        // validate_input is tested directly in validate_input_* tests above.
        // At the handler level, make_state uses closed_for_test() (live_workers=0),
        // so check_ready fires before validate_input. We verify the InvalidRequest
        // error via direct validate_input calls instead.
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = DenseRequest {
            input: TextInput(vec![]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        // not-ready fires before empty-input validation
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
        // not-ready fires before batch-size validation
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
        // make_state(true, ...) has live_workers = 0 — check_ready returns ServiceUnavailable
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
        // validate_input is tested directly in validate_input_* tests above.
        // At the handler level, make_state uses closed_for_test() (live_workers=0),
        // so check_ready fires before validate_input.
        use crate::models::TextInput;
        let state = make_state(false, 256);
        let req = SparseRequest {
            input: TextInput(vec![]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        // not-ready fires before empty-input validation
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
    }

    #[tokio::test]
    async fn sparse_embeddings_rejects_over_batch() {
        use crate::models::TextInput;
        let state = make_state(false, 2);
        let req = SparseRequest {
            input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        // not-ready fires before batch-size validation
        assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
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
        });
        let req = DenseRequest {
            input: TextInput(vec!["hello".into(), "world".into()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
        let Json(resp) = result.unwrap();
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
        // Construct SparseEmbedding using struct literal syntax.
        // fastembed::SparseEmbedding has public fields: indices: Vec<usize>, values: Vec<f32>.
        // It does not implement Clone or Debug.
        let sparse_fixture = vec![fastembed::SparseEmbedding {
            indices: vec![42usize],
            values: vec![0.5f32],
        }];
        let state = Arc::new(AppState {
            pool: EmbedPool::with_fixed_responses(vec![], sparse_fixture),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
        });
        let req = SparseRequest {
            input: TextInput(vec!["hello".into()]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(
            result.is_ok(),
            "expected Ok but got error from sparse handler"
        );
        let Json(resp) = result.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
        assert_eq!(resp.data[0].sparse_values.values, vec![0.5f32]);
    }
}
