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
