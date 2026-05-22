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

//! Paired dense + sparse embedding pipeline (one forward pass per chunk).

use anyhow::Result;
use ort::value::TensorRef;

use super::error::ort_err;
use super::math::{normalize_l2, seq_len_distribution, sparse_maxpool, sparse_project};
use super::tokenize::{build_chunk_arrays, tokenize_no_pad};
use super::types::{DualEmbedding, EmbedStats, SparseEmbedding};
use crate::binpack::{CostModel, bin_pack};
use crate::config::ModelVariant;

/// Produces paired dense + sparse embeddings using **one** `session.run()` per chunk.
///
/// Both projections are derived from the same forward pass:
/// - **FP32**: extracts both `sentence_embedding` (dense) and `token_embeddings`
///   (sparse base) from the model's dual outputs.
/// - **FP16/INT8**: extracts dense from the CLS token (position 0) of
///   `last_hidden_state`, and sparse from the full hidden states of the same
///   tensor. This avoids a second forward pass.
///
/// Numerically equivalent to calling [`super::dense::embed_dense`] and
/// [`super::sparse::embed_sparse`] separately, within FP rounding tolerance.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(super) fn embed_both(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    cost_model: &CostModel,
    model_variant: ModelVariant,
) -> Result<(Vec<DualEmbedding>, EmbedStats)> {
    let (weight, bias) = crate::weights::sparse_linear();
    let weight_view = weight.view();

    let tokenize_start = std::time::Instant::now();
    let encodings = tokenize_no_pad(tokenizer, texts)?;
    let seq_lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
    let tokenize_ms = u64::try_from(tokenize_start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let seq_dist = seq_len_distribution(&seq_lens);
    let total_token_positions: usize = seq_lens.iter().sum();
    let chunks = bin_pack(&seq_lens, cost_model);

    let mut all_dual: Vec<Option<DualEmbedding>> = (0..texts.len()).map(|_| None).collect();

    let mut max_chunk_seq: usize = 0;
    let mut inference_ms: u64 = 0;

    for (chunk_idx, chunk_indices) in chunks.iter().enumerate() {
        let chunk_max = chunk_indices
            .iter()
            .map(|&i| seq_lens[i])
            .max()
            .unwrap_or(1)
            .max(1);

        max_chunk_seq = max_chunk_seq.max(chunk_max);

        let (ids_array, mask_array) = build_chunk_arrays(&encodings, chunk_indices, chunk_max)?;

        let ids_tensor = TensorRef::from_array_view(ids_array.view()).map_err(ort_err)?;
        let mask_tensor = TensorRef::from_array_view(mask_array.view()).map_err(ort_err)?;

        let chunk_start = std::time::Instant::now();
        let outputs = {
            let _span = tracing::debug_span!(
                "chunk",
                chunk_idx,
                batch = chunk_indices.len(),
                max_seq = chunk_max
            )
            .entered();
            session
                .run(ort::inputs! {
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                })
                .map_err(ort_err)?
        };
        let chunk_ms = u64::try_from(chunk_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        inference_ms = inference_ms.saturating_add(chunk_ms);
        tracing::debug!(
            chunk_idx,
            batch = chunk_indices.len(),
            max_seq = chunk_max,
            elapsed_ms = chunk_ms,
            "both chunk inference complete"
        );

        // Extract dense + token-level hidden states from the same outputs.
        // FP32: separate sentence_embedding + token_embeddings outputs.
        // FP16/INT8: derive dense (CLS) and sparse-base from last_hidden_state.
        let (dense_emb, token_emb) = match model_variant {
            ModelVariant::Fp32 => {
                let dense = outputs["sentence_embedding"]
                    .try_extract_array::<f32>()
                    .map_err(ort_err)?
                    .to_owned();
                let tokens = outputs["token_embeddings"]
                    .try_extract_array::<f32>()
                    .map_err(ort_err)?
                    .to_owned();
                (dense, tokens)
            }
            ModelVariant::Fp16 | ModelVariant::Int8 => {
                let lhs = outputs["last_hidden_state"]
                    .try_extract_array::<f32>()
                    .map_err(ort_err)?;
                let dense = lhs.index_axis(ndarray::Axis(1), 0).to_owned();
                let tokens = lhs.to_owned();
                (dense, tokens)
            }
        };

        for (chunk_pos, &orig_idx) in chunk_indices.iter().enumerate() {
            // Dense: CLS row, L2-normalized.
            let dense_row = dense_emb.index_axis(ndarray::Axis(0), chunk_pos);
            let mut dense_vec = dense_row
                .as_slice()
                .expect("dense embedding should be contiguous")
                .to_vec();
            normalize_l2(&mut dense_vec);

            // Sparse: project each token's hidden state, then max-pool.
            let enc = &encodings[orig_idx];
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let batch_hidden = token_emb.index_axis(ndarray::Axis(0), chunk_pos);

            let scores: Vec<f32> = (0..ids.len())
                .map(|j| {
                    let hidden = batch_hidden.index_axis(ndarray::Axis(0), j);
                    let hidden_slice = hidden
                        .as_slice()
                        .expect("hidden state should be contiguous");
                    sparse_project(hidden_slice, &weight_view, *bias)
                })
                .collect();

            let (indices, values) = sparse_maxpool(ids, mask, &scores);

            all_dual[orig_idx] = Some(DualEmbedding {
                dense: dense_vec,
                sparse: SparseEmbedding { indices, values },
            });
        }
    }

    let stats = EmbedStats {
        chunks: chunks.len(),
        max_chunk_seq,
        total_token_positions,
        tokenize_ms,
        inference_ms,
        seq_len_min: seq_dist.min,
        seq_len_max: seq_dist.max,
        seq_len_mean: seq_dist.mean,
        seq_len_p95: seq_dist.p95,
    };

    Ok((
        all_dual
            .into_iter()
            .map(|d| d.expect("every slot must be filled"))
            .collect(),
        stats,
    ))
}
