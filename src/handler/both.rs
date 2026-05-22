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

//! `POST /v1/embeddings:both` handler — dense + sparse embeddings in one pass.

use std::sync::Arc;
use std::time::Instant;

use axum::{extract::State, http::HeaderMap, Json};

use super::common::{check_ready, collect_x_headers, validate_input};
use crate::error::AppError;
use crate::models::{DualEmbeddingData, DualRequest, DualResponse, SparseValues, Usage};
use crate::state::AppState;

/// Handles `POST /v1/embeddings:both` — returns dense and sparse embeddings in one pass.
///
/// # Errors
///
/// - [`AppError::ServiceUnavailable`] if the model is not ready or no workers are live.
/// - [`AppError::InvalidRequest`] if the batch is empty, exceeds `max_batch`, or any
///   text exceeds the per-string character limit.
/// - [`AppError::Internal`] if the embedding pool returns an inference error.
///
/// # Panics
///
/// Panics if the request semaphore has been closed — should not occur in normal operation.
#[allow(clippy::cast_possible_truncation)]
#[tracing::instrument(
    skip(state, req, headers),
    fields(
        batch_size,
        prompt_tokens,
        chunks,
        max_chunk_seq,
        tokenize_ms,
        inference_ms,
        queue_wait_ms,
        total_ms,
    )
)]
pub async fn both_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<DualRequest>,
) -> Result<Json<DualResponse>, AppError> {
    check_ready(&state)?;
    let x_headers = collect_x_headers(&headers);
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

    let queue_wait_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);

    let (pairs, embed_stats) = state.pool.both(texts).await?;

    let total_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
    tracing::Span::current()
        .record("chunks", embed_stats.chunks)
        .record("max_chunk_seq", embed_stats.max_chunk_seq)
        .record("tokenize_ms", embed_stats.tokenize_ms)
        .record("inference_ms", embed_stats.inference_ms)
        .record("queue_wait_ms", queue_wait_ms)
        .record("total_ms", total_ms);
    let x_headers_val =
        (!x_headers.is_empty()).then(|| serde_json::to_string(&x_headers).unwrap_or_default());
    tracing::info!(
        route = "both",
        batch_size,
        prompt_tokens,
        chunks = embed_stats.chunks,
        max_chunk_seq = embed_stats.max_chunk_seq,
        total_token_positions = embed_stats.total_token_positions,
        tokenize_ms = embed_stats.tokenize_ms,
        inference_ms = embed_stats.inference_ms,
        queue_wait_ms,
        total_ms,
        x_headers = x_headers_val,
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
