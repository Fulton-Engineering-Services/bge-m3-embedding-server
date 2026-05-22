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

//! `POST /v1/sparse-embeddings` handler — BGE-M3 SPLADE-style sparse embeddings.

use std::sync::Arc;
use std::time::Instant;

use axum::{Json, extract::State, http::HeaderMap};

use super::common::{check_ready, collect_x_headers, validate_input};
use crate::error::AppError;
use crate::models::{SparseEmbeddingData, SparseRequest, SparseResponse, SparseValues};
use crate::state::AppState;

/// Handles `POST /v1/sparse-embeddings` — returns sparse (SPLADE-style) embeddings.
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
        chunks,
        max_chunk_seq,
        tokenize_ms,
        inference_ms,
        queue_wait_ms,
        total_ms,
    )
)]
pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<SparseRequest>,
) -> Result<Json<SparseResponse>, AppError> {
    check_ready(&state)?;
    let x_headers = collect_x_headers(&headers);
    let texts = req.input.0;
    validate_input(&texts, state.max_batch)?;
    let batch_size = texts.len();
    tracing::Span::current().record("batch_size", batch_size);

    let t0 = Instant::now();

    let _permit = Arc::clone(&state.request_permits)
        .acquire_owned()
        .await
        .expect("request semaphore is never closed");

    let queue_wait_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);

    let (embeddings, embed_stats) = state.pool.sparse(texts).await?;

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
        route = "sparse",
        batch_size,
        chunks = embed_stats.chunks,
        max_chunk_seq = embed_stats.max_chunk_seq,
        total_token_positions = embed_stats.total_token_positions,
        seq_len_min = embed_stats.seq_len_min,
        seq_len_max = embed_stats.seq_len_max,
        seq_len_mean = embed_stats.seq_len_mean,
        seq_len_p95 = embed_stats.seq_len_p95,
        tokenize_ms = embed_stats.tokenize_ms,
        inference_ms = embed_stats.inference_ms,
        queue_wait_ms,
        total_ms,
        x_headers = x_headers_val,
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
