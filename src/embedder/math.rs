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

//! Pure dense/sparse math helpers (testable without ORT).

use std::collections::HashMap;

use ndarray::ArrayView1;

/// CLS, PAD, SEP/EOS, UNK — excluded from sparse output.
pub(super) const SPECIAL_TOKENS: [u32; 4] = [0, 1, 2, 3];

/// L2-normalizes `vec` in place. If the norm is zero, leaves the vector unchanged.
pub(super) fn normalize_l2(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Projects a single token's hidden state through the sparse-linear layer.
///
/// Returns `max(0, dot(hidden, weight) + bias)` (ReLU-gated score).
pub(super) fn sparse_project(hidden: &[f32], weight: &ArrayView1<f32>, bias: f32) -> f32 {
    let hidden_view = ArrayView1::from(hidden);
    (hidden_view.dot(weight) + bias).max(0.0)
}

/// Max-pools sparse scores by vocabulary token ID, excluding special tokens
/// and tokens masked by the attention mask.
///
/// Returns sorted `(indices, values)` vectors suitable for `SparseEmbedding`.
pub(super) fn sparse_maxpool(ids: &[u32], mask: &[u32], scores: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let mut token_weights: HashMap<usize, f32> = HashMap::new();

    for (j, &token_id) in ids.iter().enumerate() {
        if mask[j] == 0 {
            continue;
        }
        if SPECIAL_TOKENS.contains(&token_id) {
            continue;
        }
        let score = scores[j];
        if score > 0.0 {
            token_weights
                .entry(token_id as usize)
                .and_modify(|w| *w = w.max(score))
                .or_insert(score);
        }
    }

    let mut indices: Vec<usize> = token_weights.keys().copied().collect();
    indices.sort_unstable();
    let values: Vec<f32> = indices.iter().map(|k| token_weights[k]).collect();
    (indices, values)
}

/// Computes the median of a `Vec<usize>` in-place (sorts the slice).
///
/// Returns `0` for empty input. For even-length inputs returns the lower
/// of the two middle elements (no floating-point required).
pub(super) fn median_usize(values: &mut [usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

/// Per-batch token-length distribution statistics.
///
/// All lengths are measured in tokens (post-tokenization id counts), before
/// any padding. Carried in [`super::types::EmbedStats`] and logged on every
/// completed embed request so operators can correlate latency spikes with new
/// `(batch_size, seq_len)` shapes hitting TRT engine compile paths.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SeqLenDistribution {
    /// Minimum token sequence length in the batch.
    pub min: usize,
    /// Maximum token sequence length in the batch.
    pub max: usize,
    /// Mean token sequence length (integer, truncated toward zero).
    pub mean: usize,
    /// 95th-percentile token sequence length.
    ///
    /// Index is `(n * 95) / 100` on a sorted copy — a conservative floor
    /// that never exceeds `n - 1`.  For a batch of 64 texts this maps to
    /// index 60, meaning the 61st-shortest sequence.
    pub p95: usize,
}

/// Computes min, max, mean, and p95 token-length statistics for an embed batch.
///
/// Returns a zeroed [`SeqLenDistribution`] for an empty slice.  Allocates a
/// temporary sorted copy of `lens`; batch sizes are ≤ 256 so this is cheap.
pub(super) fn seq_len_distribution(lens: &[usize]) -> SeqLenDistribution {
    if lens.is_empty() {
        return SeqLenDistribution::default();
    }
    let min = *lens.iter().min().expect("non-empty");
    let max = *lens.iter().max().expect("non-empty");
    let mean = lens.iter().sum::<usize>() / lens.len();
    let mut sorted = lens.to_vec();
    sorted.sort_unstable();
    let p95_idx = (sorted.len() * 95) / 100;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];
    SeqLenDistribution {
        min,
        max,
        mean,
        p95,
    }
}

#[cfg(test)]
mod tests {
    use super::seq_len_distribution;

    #[test]
    fn single_element() {
        let d = seq_len_distribution(&[42]);
        assert_eq!(d.min, 42);
        assert_eq!(d.max, 42);
        assert_eq!(d.mean, 42);
        assert_eq!(d.p95, 42);
    }

    #[test]
    fn empty_returns_zeros() {
        let d = seq_len_distribution(&[]);
        assert_eq!(d.min, 0);
        assert_eq!(d.max, 0);
        assert_eq!(d.mean, 0);
        assert_eq!(d.p95, 0);
    }

    #[test]
    fn uniform_batch() {
        let lens: Vec<usize> = vec![100; 64];
        let d = seq_len_distribution(&lens);
        assert_eq!(d.min, 100);
        assert_eq!(d.max, 100);
        assert_eq!(d.mean, 100);
        assert_eq!(d.p95, 100);
    }

    #[test]
    fn ascending_sequence_p95() {
        // lens = [1, 2, ..., 100]. Sorted same order.
        // p95_idx = (100 * 95) / 100 = 95 → sorted[95] = 96.
        let lens: Vec<usize> = (1..=100).collect();
        let d = seq_len_distribution(&lens);
        assert_eq!(d.min, 1);
        assert_eq!(d.max, 100);
        assert_eq!(d.mean, 50); // sum=5050, /100 = 50 (integer)
        assert_eq!(d.p95, 96);
    }

    #[test]
    fn two_elements() {
        // p95_idx = (2 * 95) / 100 = 1, so sorted[1] = max.
        let d = seq_len_distribution(&[10, 200]);
        assert_eq!(d.min, 10);
        assert_eq!(d.max, 200);
        assert_eq!(d.mean, 105);
        assert_eq!(d.p95, 200);
    }

    #[test]
    fn p95_does_not_panic_on_small_batches() {
        // For n in 1..20 verify p95_idx stays within bounds.
        for n in 1usize..=20 {
            let lens: Vec<usize> = (1..=n).collect();
            let d = seq_len_distribution(&lens);
            // p95 must be between min and max inclusive.
            assert!(d.p95 >= d.min, "p95 < min for n={n}");
            assert!(d.p95 <= d.max, "p95 > max for n={n}");
        }
    }

    #[test]
    fn mean_truncated() {
        // sum=5, len=2 → 5/2=2 (integer truncation).
        let d = seq_len_distribution(&[2, 3]);
        assert_eq!(d.mean, 2);
    }
}
