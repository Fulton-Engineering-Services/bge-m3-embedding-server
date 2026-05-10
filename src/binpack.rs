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

//! Bin-packing algorithm that groups tokenized sequences into `session.run()`
//! calls that each fit within the per-worker workspace budget.
//!
//! The central type is [`CostModel`], which captures the quadratic memory
//! scaling of BGE-M3 attention and is used by `bin_pack` to partition an
//! incoming batch into chunks that are safe to run in a single ORT call.

/// Quadratic-aware workspace cost model for ONNX attention inference.
///
/// BGE-M3 uses multi-head attention whose intermediate tensor footprint scales
/// as `O(batch * seq^2)` (attention score matrix) plus `O(batch * seq)`
/// (FFN intermediates, projection matrices). The total peak workspace is
/// approximately:
///
/// ```text
/// peak ≈ a * (batch * seq) + b * (batch * seq^2)
/// ```
///
/// where `a` (bytes/token-position) captures the FFN / projection contribution
/// and `b` (bytes/token-position^2) captures the attention contribution.
///
/// At sequence length 512 attention is small relative to FFN, so a linear
/// approximation works. At 8192, `b * N^2` dominates by ~16×, so using only
/// `a` would under-budget by that same factor.
///
/// Coefficients are derived at startup by [`crate::probe`] or set
/// conservatively from compile-time defaults when measurement is unavailable.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct CostModel {
    /// Bytes per token-position (linear term: FFN intermediates, projections).
    pub a: f64,
    /// Bytes per token-position-squared (quadratic term: attention scores).
    pub b: f64,
    /// Maximum workspace bytes available per worker for a single `session.run()` call.
    pub max_workspace_bytes: usize,
}

impl CostModel {
    /// Conservative static defaults calibrated so a `(16, 512)` chunk lands at
    /// ~140 MB workspace — matching the old static budget at the previous default
    /// `BGE_M3_ONNX_BATCH_SIZE = 16`, `MAX_SEQ_LENGTH = 512`.
    ///
    /// These are used when the probe cannot run (no ORT, no model, macOS without
    /// cgroup support) or when `BGE_M3_DISABLE_AUTO_BUDGET` is set.
    ///
    /// Formula check: 16 KiB/token × 16 × 512 + 8 B/token² × 16 × 512²
    ///   = 16384 × 8192 + 8 × 16 × 262144
    ///   = 134 217 728 + 33 554 432
    ///   = 167 772 160 ≈ 160 MB per chunk (workers run sequentially inside one worker).
    pub const CONSERVATIVE_A: f64 = 16_384.0; // 16 KiB per token-position
    /// Conservative quadratic coefficient (bytes per token-position squared).
    pub const CONSERVATIVE_B: f64 = 8.0; // 8 bytes per token-position^2

    /// Default maximum workspace per worker when memory cannot be detected.
    ///
    /// 2 GiB is conservatively safe for the Fargate 28 GiB task with 7 workers
    /// (`28 GB * 0.7 safety / 7 workers ≈ 2.8 GB`); we round down for headroom.
    pub const DEFAULT_MAX_WORKSPACE: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

    /// Constructs a `CostModel` with conservative defaults and the given workspace ceiling.
    #[must_use]
    pub fn conservative(max_workspace_bytes: usize) -> Self {
        Self {
            a: Self::CONSERVATIVE_A,
            b: Self::CONSERVATIVE_B,
            max_workspace_bytes,
        }
    }

    /// Estimated peak workspace (bytes) for a single `session.run()` call with
    /// `count` texts and `max_seq` as the padded sequence length.
    ///
    /// Uses saturating arithmetic on `u128` to avoid overflow at large inputs.
    //
    // cast_precision_loss: n is u128, but realistic values (batch ≤ 256, seq ≤ 8192)
    //   keep n ≤ 2_097_152 — well within f64's 2^52 mantissa — so no bits are lost.
    // cast_possible_truncation: f64 → u128 intentionally floors fractional bytes;
    //   this is a memory *budget estimate*, not an exact byte count.
    // cast_sign_loss: a and b are validated positive at construction, so the
    //   products are always ≥ 0 before the cast.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn chunk_cost(&self, count: usize, max_seq: usize) -> u128 {
        let n = count as u128 * max_seq as u128;
        let linear = (self.a * n as f64) as u128;
        let quad = (self.b * n as f64 * max_seq as f64) as u128;
        linear.saturating_add(quad)
    }

    /// Returns `true` if the chunk fits within the workspace budget.
    #[must_use]
    pub fn fits(&self, count: usize, max_seq: usize) -> bool {
        self.chunk_cost(count, max_seq) <= self.max_workspace_bytes as u128
    }
}

/// Length-sorted greedy bin-packer.
///
/// Partitions `seq_lengths` (indexed 0..n) into contiguous groups (chunks)
/// where each chunk satisfies `cost_model.fits(chunk.len(), max_seq_in_chunk)`.
///
/// If a single text exceeds the budget on its own — which can happen when
/// `max_workspace_bytes` is very small or the text is at `MAX_SEQ_LENGTH` and
/// the budget is tighter than one text — it gets its own single-element chunk.
/// The caller (ORT session) will either succeed or fail; we never silently
/// truncate or discard inputs.
///
/// # Returns
///
/// `Vec<Vec<usize>>` where each inner `Vec` contains the **original indices**
/// of texts in that chunk, sorted ascending by sequence length. The outer vec
/// preserves the order chunks should be processed in. Callers scatter results
/// back to the original positions using these indices.
///
/// # Complexity
///
/// `O(n log n)` sort + `O(n)` scan.
pub(crate) fn bin_pack(seq_lengths: &[usize], cost_model: &CostModel) -> Vec<Vec<usize>> {
    if seq_lengths.is_empty() {
        return Vec::new();
    }

    // Sort indices by ascending sequence length so we can greedily pack
    // short texts together. Long texts naturally form their own small chunks.
    let mut sorted: Vec<usize> = (0..seq_lengths.len()).collect();
    sorted.sort_unstable_by_key(|&i| seq_lengths[i]);

    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_max_seq: usize = 0;

    for idx in sorted {
        let seq = seq_lengths[idx];
        let new_max = current_max_seq.max(seq);
        let new_count = current.len() + 1;

        if current.is_empty() || cost_model.fits(new_count, new_max) {
            // Adding this text keeps the chunk within budget.
            current.push(idx);
            current_max_seq = new_max;
        } else {
            // Flush the current chunk and start a new one.
            tracing::debug!(
                chunk_idx = chunks.len(),
                batch = current.len(),
                max_seq = current_max_seq,
                estimated_workspace_mb =
                    cost_model.chunk_cost(current.len(), current_max_seq) / (1024 * 1024),
                "bin_pack chunk decided"
            );
            chunks.push(std::mem::take(&mut current));
            current.push(idx);
            current_max_seq = seq;
        }
    }

    if !current.is_empty() {
        tracing::debug!(
            chunk_idx = chunks.len(),
            batch = current.len(),
            max_seq = current_max_seq,
            estimated_workspace_mb =
                cost_model.chunk_cost(current.len(), current_max_seq) / (1024 * 1024),
            "bin_pack chunk decided"
        );
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests;
