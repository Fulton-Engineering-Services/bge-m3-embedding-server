// Copyright (c) 2026 J. Patrick Fulton
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `GET /v1/models` handler — OpenAI-compatible fleet discovery endpoint.

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
