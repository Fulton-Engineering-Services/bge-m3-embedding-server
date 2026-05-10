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

//! Tokenizer load + no-pad tokenization + chunk-array build helpers.

use std::path::Path;

use anyhow::Result;

/// Loads and configures the BGE-M3 tokenizer with truncation at `max_seq_length`
/// but **no** padding. Padding is applied per-chunk in [`build_chunk_arrays`].
pub(super) fn load_tokenizer(
    tokenizer_path: &Path,
    max_seq_length: usize,
) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_seq_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("Failed to set truncation: {e}"))?;

    // No BatchLongest padding here — we pad manually in build_chunk_arrays
    // so each chunk only pads to its own longest sequence.
    tokenizer.with_padding(None);

    Ok(tokenizer)
}

/// Tokenizes `texts` without applying any padding. Returns one `Encoding` per text,
/// each truncated to the tokenizer's configured `max_length`.
pub(super) fn tokenize_no_pad(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<Vec<tokenizers::Encoding>> {
    let str_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch_fast(str_refs, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    Ok(encodings)
}

/// Builds `input_ids` and `attention_mask` arrays for a single chunk.
///
/// `indices` selects which encodings from `all_encodings` belong to this chunk.
/// `pad_to` is the chunk-local maximum sequence length; all sequences are
/// right-padded with `pad_id = 1` (XLM-RoBERTa `<pad>` token).
#[allow(clippy::cast_possible_truncation)]
pub(super) fn build_chunk_arrays(
    all_encodings: &[tokenizers::Encoding],
    indices: &[usize],
    pad_to: usize,
) -> Result<(ndarray::Array2<i64>, ndarray::Array2<i64>)> {
    let batch = indices.len();
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * pad_to);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * pad_to);

    for &idx in indices {
        let enc = &all_encodings[idx];
        let token_ids = enc.get_ids();
        let attn_mask = enc.get_attention_mask();
        let seq_len = token_ids.len();

        // Copy token ids and mask
        ids_flat.extend(token_ids.iter().map(|&id| i64::from(id)));
        mask_flat.extend(attn_mask.iter().map(|&m| i64::from(m)));

        // Right-pad with pad_id=1 / mask=0
        let pad = pad_to.saturating_sub(seq_len);
        ids_flat.extend(std::iter::repeat_n(1i64, pad));
        mask_flat.extend(std::iter::repeat_n(0i64, pad));
    }

    let ids_array = ndarray::Array2::from_shape_vec((batch, pad_to), ids_flat)?;
    let mask_array = ndarray::Array2::from_shape_vec((batch, pad_to), mask_flat)?;

    Ok((ids_array, mask_array))
}
