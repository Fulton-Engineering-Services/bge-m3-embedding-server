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

//! Tests for post-inference cache-miss signaling (`log_inference_complete`).

use super::super::propagation::log_inference_complete;
use super::super::trt_retry::CHUNK_CACHE_HIT_THRESHOLD_MS;
use super::helpers::sample_embed_stats;

#[test]
fn log_inference_complete_cache_hit_does_not_broadcast() {
    let stats = sample_embed_stats(CHUNK_CACHE_HIT_THRESHOLD_MS - 1, 128);
    let (prop_tx, _prop_rx) = tokio::sync::broadcast::channel(4);
    let shape = log_inference_complete(&stats, 0, "dense", None, Some(&prop_tx), 2);
    assert!(shape.is_none());
}

#[test]
fn log_inference_complete_cache_miss_broadcasts_shape() {
    let stats = sample_embed_stats(CHUNK_CACHE_HIT_THRESHOLD_MS, 512);
    let (prop_tx, _prop_rx) = tokio::sync::broadcast::channel(4);
    let shape = log_inference_complete(&stats, 1, "both", None, Some(&prop_tx), 4);
    assert_eq!(shape, Some((4, 512)));
}

#[test]
fn log_inference_complete_cache_miss_without_propagation_tx() {
    let stats = sample_embed_stats(9_000, 8192);
    let (jit_tx, mut jit_rx) = tokio::sync::mpsc::channel(2);
    let shape = log_inference_complete(&stats, 2, "sparse", Some(&jit_tx), None, 1);
    assert!(shape.is_none());
    assert_eq!(jit_rx.try_recv().unwrap(), (1, 8192));
}

#[test]
fn log_inference_complete_boundary_at_threshold_is_miss() {
    let stats = sample_embed_stats(CHUNK_CACHE_HIT_THRESHOLD_MS, 256);
    let (jit_tx, mut jit_rx) = tokio::sync::mpsc::channel(2);
    let shape = log_inference_complete(&stats, 0, "dense", Some(&jit_tx), None, 3);
    assert!(shape.is_none());
    assert_eq!(jit_rx.try_recv().unwrap(), (3, 256));
}
