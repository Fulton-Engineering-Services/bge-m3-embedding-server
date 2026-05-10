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

// -----------------------------------------------------------------------
// Pure-math helper tests (no ORT session needed)
// -----------------------------------------------------------------------

use super::super::super::math::{normalize_l2, sparse_maxpool, sparse_project};

// ── normalize_l2 ──────────────────────────────────────────────────────────

#[test]
fn normalize_l2_unit_vector() {
    let mut v = vec![3.0, 4.0];
    normalize_l2(&mut v);
    let expected_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((expected_norm - 1.0).abs() < 1e-6, "should be unit length");
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn normalize_l2_zero_vector_unchanged() {
    let mut v = vec![0.0, 0.0, 0.0];
    normalize_l2(&mut v);
    assert!(v.iter().all(|&x| x == 0.0), "zero vector should stay zero");
}

#[test]
fn normalize_l2_already_unit() {
    let mut v = vec![1.0, 0.0, 0.0];
    normalize_l2(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!(v[1].abs() < 1e-6);
    assert!(v[2].abs() < 1e-6);
}

#[test]
fn normalize_l2_sign_preservation() {
    let mut v = vec![-3.0, 4.0];
    normalize_l2(&mut v);
    assert!(
        (v[0] - (-0.6)).abs() < 1e-6,
        "negative sign must be preserved"
    );
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn normalize_l2_single_element() {
    let mut v = vec![5.0];
    normalize_l2(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-6);

    let mut v2 = vec![-7.0];
    normalize_l2(&mut v2);
    assert!((v2[0] - (-1.0)).abs() < 1e-6);
}

#[test]
fn normalize_l2_output_norm_is_one() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    normalize_l2(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "output norm must equal 1.0, got {norm}"
    );
}

// ── sparse_project ────────────────────────────────────────────────────────

#[test]
fn sparse_project_positive_score() {
    let weight = ndarray::array![1.0, 2.0, 3.0];
    let hidden = [1.0, 1.0, 1.0];
    let score = sparse_project(&hidden, &weight.view(), 0.5);
    assert!((score - 6.5).abs() < 1e-6);
}

#[test]
fn sparse_project_relu_clamps_negative() {
    let weight = ndarray::array![1.0, 1.0];
    let hidden = [-5.0, -5.0];
    let score = sparse_project(&hidden, &weight.view(), 0.0);
    assert!(
        score.abs() < 1e-6,
        "negative scores should be clamped to zero"
    );
}

#[test]
fn sparse_project_zero_weight() {
    let weight = ndarray::array![0.0, 0.0, 0.0];
    let hidden = [1.0, 2.0, 3.0];
    let score = sparse_project(&hidden, &weight.view(), 1.0);
    assert!((score - 1.0).abs() < 1e-6);
}

#[test]
fn sparse_project_negative_bias() {
    let weight = ndarray::array![1.0, 1.0];
    let hidden = [1.0, 1.0];
    let score = sparse_project(&hidden, &weight.view(), -3.0);
    assert!(score.abs() < 1e-6, "negative bias should clamp via ReLU");
}

// ── sparse_maxpool ────────────────────────────────────────────────────────

#[test]
fn sparse_maxpool_all_masked_out() {
    let ids = [100, 200, 300];
    let mask = [0, 0, 0];
    let scores = [0.5, 0.8, 0.3];
    let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn sparse_maxpool_basic() {
    let ids = [10, 20, 10];
    let mask = [1, 1, 1];
    let scores = [0.3, 0.5, 0.7];
    let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
    assert_eq!(indices, vec![10, 20]);
    assert!((values[0] - 0.7).abs() < 1e-6);
    assert!((values[1] - 0.5).abs() < 1e-6);
}

#[test]
fn sparse_maxpool_filters_special_tokens() {
    let ids = [0, 1, 2, 3, 100];
    let mask = [1, 1, 1, 1, 1];
    let scores = [0.9, 0.9, 0.9, 0.9, 0.5];
    let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
    assert_eq!(indices, vec![100]);
    assert!((values[0] - 0.5).abs() < 1e-6);
}

#[test]
fn sparse_maxpool_respects_attention_mask() {
    let ids = [100, 200];
    let mask = [1, 0];
    let scores = [0.5, 0.9];
    let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
    assert_eq!(indices, vec![100]);
    assert!((values[0] - 0.5).abs() < 1e-6);
}

#[test]
fn sparse_maxpool_skips_zero_scores() {
    let ids = [100, 200];
    let mask = [1, 1];
    let scores = [0.0, 0.5];
    let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
    assert_eq!(indices, vec![200]);
    assert!((values[0] - 0.5).abs() < 1e-6);
}

#[test]
fn sparse_maxpool_empty_input() {
    let ids: [u32; 0] = [];
    let mask: [u32; 0] = [];
    let scores: [f32; 0] = [];
    let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn sparse_maxpool_returns_sorted_indices() {
    let ids = [300, 100, 200];
    let mask = [1, 1, 1];
    let scores = [0.1, 0.2, 0.3];
    let (indices, _) = sparse_maxpool(&ids, &mask, &scores);
    assert_eq!(indices, vec![100, 200, 300]);
}
