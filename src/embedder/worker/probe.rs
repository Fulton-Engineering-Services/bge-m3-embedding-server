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

//! Probe inference helpers for startup measurement and arena priming.

use anyhow::Result;
use ort::value::TensorRef;

use crate::embedder::error::ort_err;
use crate::embedder::tokenize::{build_chunk_arrays, tokenize_no_pad};
use crate::embedder::types::ProbeResult;
use crate::sysinfo;

/// Runs a single `session.run()` for the probe, measuring RSS before and after.
///
/// The probe texts are already tokenized and padded to `pad_to` externally.
/// This function just runs inference and returns RSS deltas so `probe.rs` can
/// fit the cost model.
pub(crate) fn probe_run_dense(
    session: &mut ort::session::Session,
    ids_array: &ndarray::Array2<i64>,
    mask_array: &ndarray::Array2<i64>,
) -> Result<ProbeResult> {
    let rss_before = sysinfo::read_process_rss_bytes().unwrap_or(0);

    let ids_tensor = TensorRef::from_array_view(ids_array.view()).map_err(ort_err)?;
    let mask_tensor = TensorRef::from_array_view(mask_array.view()).map_err(ort_err)?;

    // Run inference (output discarded — we only care about RSS).
    let _outputs = session
        .run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })
        .map_err(ort_err)?;

    let rss_after = sysinfo::read_process_rss_bytes().unwrap_or(rss_before);

    Ok(ProbeResult {
        rss_before,
        rss_after,
    })
}
/// Runs one probe batch: tokenize texts, build padded arrays, call `session.run()`,
/// and return RSS deltas. Uses `embed_dense`'s no-pad tokenizer path.
pub(super) fn run_probe_batch(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<ProbeResult> {
    let encodings = tokenize_no_pad(tokenizer, texts)?;
    let pad_to = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(1)
        .max(1);
    let indices: Vec<usize> = (0..texts.len()).collect();
    let (ids_array, mask_array) = build_chunk_arrays(&encodings, &indices, pad_to)?;
    probe_run_dense(session, &ids_array, &mask_array)
}
