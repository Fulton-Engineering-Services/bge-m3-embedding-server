use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
};
use std::sync::Arc;

use crate::models::{EmbeddingData, EmbeddingRequest, EmbeddingResponse, SparseValues};
use crate::state::AppState;

pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    if !state.ready.load(std::sync::atomic::Ordering::Acquire) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "model not ready".to_string(),
        ));
    }

    let mut embedder_guard = state.embedder.lock().await;
    let embedder = embedder_guard.as_mut().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "embedder not initialized".to_string(),
        )
    })?;

    let results = embedder
        .embed(req.input)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data = results
        .into_iter()
        .enumerate()
        .map(|(index, emb)| EmbeddingData {
            index,
            sparse_values: SparseValues {
                indices: emb.indices.into_iter().map(|i| i as u32).collect(),
                values: emb.values,
            },
        })
        .collect();

    Ok(Json(EmbeddingResponse { data }))
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.ready.load(std::sync::atomic::Ordering::Acquire) {
        Json(serde_json::json!({"status": "ok"})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response()
    }
}
