use std::sync::Arc;
use std::time::Instant;

use axum::{extract::State, Json};

use super::common::{check_ready, validate_input};
use crate::error::AppError;
use crate::models::{DualEmbeddingData, DualRequest, DualResponse, SparseValues, Usage};
use crate::state::AppState;

#[allow(clippy::cast_possible_truncation)]
#[tracing::instrument(
    skip(state, req),
    fields(
        batch_size,
        prompt_tokens,
        chunks,
        max_chunk_seq,
        tokenize_ms,
        inference_ms,
        total_ms
    )
)]
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
