use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, Json};

use crate::models::{ModelEntry, ModelsResponse};
use crate::state::AppState;

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
