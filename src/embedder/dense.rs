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

//! Dense embedding pipeline.

use anyhow::Result;
use ort::value::TensorRef;

use super::error::ort_err;
use super::math::{normalize_l2, seq_len_distribution};
use super::tokenize::{build_chunk_arrays, tokenize_no_pad};
use super::types::EmbedStats;
use crate::binpack::{bin_pack, CostModel};
use crate::config::ModelVariant;

/// Produces L2-normalized dense embeddings.
///
/// Tokenizes once, then uses the cost model to bin-pack into chunks that fit
/// within the workspace budget. Results are scattered back to the original
/// input order.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn embed_dense(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    cost_model: &CostModel,
    model_variant: ModelVariant,
) -> Result<(Vec<Vec<f32>>, EmbedStats)> {
    let tokenize_start = std::time::Instant::now();
    let encodings = tokenize_no_pad(tokenizer, texts)?;
    let seq_lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
    let tokenize_ms = u64::try_from(tokenize_start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let seq_dist = seq_len_distribution(&seq_lens);
    let total_token_positions: usize = seq_lens.iter().sum();
    let chunks = bin_pack(&seq_lens, cost_model);

    // Pre-allocate output slots (one per input text, filled below).
    let mut all_embeddings: Vec<Vec<f32>> = (0..texts.len()).map(|_| Vec::new()).collect();

    let mut max_chunk_seq: usize = 0;
    let mut inference_ms: u64 = 0;

    for (chunk_idx, chunk_indices) in chunks.iter().enumerate() {
        let chunk_max = chunk_indices
            .iter()
            .map(|&i| seq_lens[i])
            .max()
            .unwrap_or(1)
            .max(1); // guard: at least 1 to avoid 0-dim tensors

        max_chunk_seq = max_chunk_seq.max(chunk_max);

        let (ids_array, mask_array) = build_chunk_arrays(&encodings, chunk_indices, chunk_max)?;
        let batch_len = ids_array.nrows();

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
            "dense chunk inference complete"
        );

        // FP32: sentence_embedding [batch, 1024] — pre-pooled CLS output.
        // FP16/INT8: last_hidden_state [batch, seq, 1024] — CLS token at position 0.
        let emb: ndarray::ArrayD<f32> = match model_variant {
            ModelVariant::Fp32 => outputs["sentence_embedding"]
                .try_extract_array::<f32>()
                .map_err(ort_err)?
                .to_owned(),
            ModelVariant::Fp16 | ModelVariant::Int8 => {
                let lhs = outputs["last_hidden_state"]
                    .try_extract_array::<f32>()
                    .map_err(ort_err)?;
                lhs.index_axis(ndarray::Axis(1), 0).to_owned()
            }
        };

        for (chunk_pos, &orig_idx) in chunk_indices.iter().enumerate() {
            debug_assert!(chunk_pos < batch_len, "chunk_pos must be within batch");
            let row = emb.index_axis(ndarray::Axis(0), chunk_pos);
            let mut vec = row
                .as_slice()
                .expect("embedding should be contiguous")
                .to_vec();
            normalize_l2(&mut vec);
            all_embeddings[orig_idx] = vec;
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

    Ok((all_embeddings, stats))
}
