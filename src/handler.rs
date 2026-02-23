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
    Ok(())
}

fn check_ready(state: &AppState) -> Result<(), AppError> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(AppError::ServiceUnavailable("model not ready".to_string()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public handlers
// ---------------------------------------------------------------------------

pub async fn dense_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DenseRequest>,
) -> Result<Json<DenseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    // model field is accepted for OpenAI API compatibility but ignored; BGE-M3 is always used.
    let _model = req.model;
    validate_input(&texts, state.max_batch)?;

    let prompt_tokens: usize = texts.iter().map(|t| t.len() / 4 + 1).sum();

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

pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SparseRequest>,
) -> Result<Json<SparseResponse>, AppError> {
    check_ready(&state)?;
    let texts = req.input.0;
    validate_input(&texts, state.max_batch)?;

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
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response()
    }
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
        // We need a real EmbedPool channel to construct AppState, but we
        // won't actually send requests in these unit tests — we only test
        // the validation and readiness logic paths that return early.
        let (pool, _handle) = EmbedPool::spawn(1, std::path::PathBuf::from("/nonexistent"));
        Arc::new(AppState {
            pool,
            ready: AtomicBool::new(ready),
            max_batch,
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
            matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains("5") && msg.contains("3"))
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

    #[tokio::test]
    async fn check_ready_returns_ok_when_ready() {
        let state = make_state(true, 10);
        assert!(check_ready(&state).is_ok());
    }

    #[tokio::test]
    async fn check_ready_returns_err_when_not_ready() {
        let state = make_state(false, 10);
        let result = check_ready(&state);
        assert!(
            matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "model not ready")
        );
    }

    // --- health handler ---

    #[tokio::test]
    async fn health_returns_200_when_ready() {
        let state = make_state(true, 10);
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn health_returns_503_when_not_ready() {
        let state = make_state(false, 10);
        let response = health(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
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
    async fn dense_embeddings_rejects_empty_input() {
        use crate::models::TextInput;
        let state = make_state(true, 256);
        let req = DenseRequest {
            input: TextInput(vec![]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn dense_embeddings_rejects_over_batch() {
        use crate::models::TextInput;
        let state = make_state(true, 2);
        let req = DenseRequest {
            input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            model: None,
        };
        let result = dense_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::InvalidRequest(_))));
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
    async fn sparse_embeddings_rejects_empty_input() {
        use crate::models::TextInput;
        let state = make_state(true, 256);
        let req = SparseRequest {
            input: TextInput(vec![]),
        };
        let result = sparse_embeddings(State(state), Json(req)).await;
        assert!(matches!(result, Err(AppError::InvalidRequest(_))));
    }
}
