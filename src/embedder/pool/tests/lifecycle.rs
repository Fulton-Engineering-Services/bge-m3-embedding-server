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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::super::math::median_usize;
use super::super::super::types::EmbedStats;
use super::super::EmbedPool;

// ── worker count accessors ──────────────────────────────────────────────

#[test]
fn live_worker_count_returns_zero_for_closed_pool() {
    let pool = EmbedPool::closed_for_test();
    assert_eq!(pool.live_worker_count(), 0);
}

#[test]
fn loaded_worker_count_returns_zero_for_closed_pool() {
    let pool = EmbedPool::closed_for_test();
    assert_eq!(pool.loaded_worker_count(), 0);
}

#[test]
fn idle_for_test_has_live_workers_but_no_loaded_workers() {
    let pool = EmbedPool::idle_for_test();
    assert_eq!(pool.live_worker_count(), 1);
    assert_eq!(pool.loaded_worker_count(), 0);
}

// ── model RSS accessor ──────────────────────────────────────────────────

#[tokio::test]
async fn model_rss_per_worker_bytes_returns_zero_for_test_pool() {
    // Test helpers initialize model_rss_per_worker_bytes to 0 since no real
    // load_models() runs; the getter must reflect that.
    let pool = EmbedPool::closed_for_test();
    assert_eq!(pool.model_rss_per_worker_bytes(), 0);
    let pool2 = EmbedPool::with_fixed_responses(vec![], vec![]);
    assert_eq!(pool2.model_rss_per_worker_bytes(), 0);
}

#[test]
fn model_rss_per_worker_bytes_defaults_to_zero() {
    let pool = EmbedPool::closed_for_test();
    assert_eq!(pool.model_rss_per_worker_bytes(), 0);
}

#[test]
fn model_rss_per_worker_bytes_reflects_stored_value() {
    let pool = EmbedPool::closed_for_test();
    let atomic: Arc<AtomicUsize> = pool.model_rss_per_worker_bytes_atomic();
    atomic.store(1_234_567, Ordering::Release);
    assert_eq!(pool.model_rss_per_worker_bytes(), 1_234_567);
}

// ── EmbedStats ────────────────────────────────────────────────────────────

#[test]
fn embed_stats_default_is_all_zero() {
    let stats = EmbedStats::default();
    assert_eq!(stats.chunks, 0);
    assert_eq!(stats.max_chunk_seq, 0);
    assert_eq!(stats.total_token_positions, 0);
    assert_eq!(stats.tokenize_ms, 0);
    assert_eq!(stats.inference_ms, 0);
}

#[tokio::test]
async fn fixture_pool_returns_default_embed_stats() {
    let pool = EmbedPool::with_fixed_responses(vec![vec![0.1f32]], vec![]);
    let (_embeddings, stats) = pool
        .dense(vec!["hello".into()])
        .await
        .expect("fixture pool should succeed");
    // Fixture always returns EmbedStats::default() — all zero.
    assert_eq!(
        stats.chunks, 0,
        "fixture pool stats should be default zeros"
    );
    assert_eq!(stats.inference_ms, 0);
}

// ── queue depth ───────────────────────────────────────────────────────────

#[tokio::test]
async fn queue_depth_is_zero_when_idle() {
    let pool = EmbedPool::with_fixed_responses(vec![], vec![]);
    // No requests in flight — queue should be empty.
    assert_eq!(pool.queue_depth(), 0);
}

#[test]
fn queue_depth_is_zero_for_closed_pool() {
    let pool = EmbedPool::closed_for_test();
    // Closed pool has a dropped receiver; capacity reports max - 0 pending = 0.
    assert_eq!(pool.queue_depth(), 0);
}

// ── median_usize ──────────────────────────────────────────────────────────

#[test]
fn median_usize_empty_returns_zero() {
    let mut v: Vec<usize> = vec![];
    assert_eq!(median_usize(&mut v), 0);
}

#[test]
fn median_usize_single_element() {
    let mut v = vec![42usize];
    assert_eq!(median_usize(&mut v), 42);
}

#[test]
fn median_usize_odd_count() {
    let mut v = vec![3usize, 1, 2];
    assert_eq!(median_usize(&mut v), 2);
}

#[test]
fn median_usize_even_count_returns_lower_middle() {
    // For even-length: returns the lower of the two middle elements.
    let mut v = vec![1usize, 3, 5, 7];
    // sorted: [1, 3, 5, 7]; len/2 = 2 → v[2] = 5
    assert_eq!(median_usize(&mut v), 5);
}

#[test]
fn median_usize_outlier_does_not_inflate_result() {
    // Simulates the production scenario: 5 clean readings ~1100 MB
    // and one contaminated reading at 8459 MB (parallel-load artifact).
    // Before the fix, fetch_max returned 8459; median returns 1100.
    let mut v = vec![
        1_100usize * 1024 * 1024,
        1_080usize * 1024 * 1024,
        1_100usize * 1024 * 1024,
        8_459usize * 1024 * 1024, // outlier
        1_090usize * 1024 * 1024,
    ];
    let median = median_usize(&mut v);
    // sorted: [1080, 1090, 1100, 1100, 8459]; len/2 = 2 → v[2] = 1100 MiB
    assert_eq!(median, 1_100 * 1024 * 1024);
    assert!(
        median < 2_000 * 1024 * 1024,
        "median ({} MiB) should be nowhere near the outlier (8459 MiB)",
        median / (1024 * 1024)
    );
}
