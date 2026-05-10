use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;

use super::super::math::{median_usize, normalize_l2, sparse_maxpool, sparse_project};
use super::super::types::{EmbedStats, SparseEmbedding};
use super::super::worker::WorkerConfig;
use super::EmbedPool;
use crate::binpack::CostModel;

fn bad_cache_dir() -> PathBuf {
    PathBuf::from("/dev/null/impossible")
}

fn test_cost_model_handle() -> Arc<ArcSwap<CostModel>> {
    Arc::new(ArcSwap::from_pointee(CostModel::conservative(
        CostModel::DEFAULT_MAX_WORKSPACE,
    )))
}

#[tokio::test]
async fn spawn_propagates_leader_load_failure() {
    let (pool, init_handle) = EmbedPool::spawn(
        1,
        bad_cache_dir(),
        WorkerConfig {
            cost_model: test_cost_model_handle(),
            idle_timeout: None,
            model_variant: crate::config::ModelVariant::Fp32,
            max_seq_length: 512,
            intra_threads: 1,
        },
    );

    let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
        .await
        .expect("init_handle should resolve quickly, not hang")
        .expect("JoinHandle should not panic");

    assert!(
        result.is_err(),
        "init should return Err on leader load failure"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("failed to load"),
        "error should mention load failure, got: {msg}"
    );

    assert_eq!(pool.loaded_worker_count(), 0);
}

#[tokio::test]
async fn spawn_multi_worker_fails_fast_on_leader_failure() {
    let (pool, init_handle) = EmbedPool::spawn(
        3,
        bad_cache_dir(),
        WorkerConfig {
            cost_model: test_cost_model_handle(),
            idle_timeout: None,
            model_variant: crate::config::ModelVariant::Fp32,
            max_seq_length: 512,
            intra_threads: 1,
        },
    );

    let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
        .await
        .expect("init_handle should resolve quickly, not hang")
        .expect("JoinHandle should not panic");

    assert!(
        result.is_err(),
        "init should fail without spawning followers"
    );
    assert_eq!(pool.loaded_worker_count(), 0);
}

#[tokio::test]
async fn dense_returns_error_when_channel_closed() {
    let pool = EmbedPool::closed_for_test();
    let result = pool.dense(vec!["hello".into()]).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("channel closed"));
}

#[tokio::test]
async fn sparse_returns_error_when_channel_closed() {
    let pool = EmbedPool::closed_for_test();
    let result = pool.sparse(vec!["hello".into()]).await;
    let err = result.expect_err("expected an error");
    assert!(err.to_string().contains("channel closed"));
}

#[tokio::test]
async fn both_returns_error_when_channel_closed() {
    let pool = EmbedPool::closed_for_test();
    let result = pool.both(vec!["hello".into()]).await;
    let err = result.expect_err("expected an error");
    assert!(err.to_string().contains("channel closed"));
}

#[tokio::test]
async fn both_returns_paired_dense_and_sparse_from_fixture() {
    let dense_fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let sparse_fixture = vec![
        SparseEmbedding {
            indices: vec![10],
            values: vec![0.7],
        },
        SparseEmbedding {
            indices: vec![20, 30],
            values: vec![0.8, 0.9],
        },
    ];
    let pool = EmbedPool::with_fixed_responses(dense_fixture.clone(), sparse_fixture.clone());
    let (result, _stats) = pool
        .both(vec!["a".into(), "b".into()])
        .await
        .expect("both should succeed against fixture pool");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].dense, dense_fixture[0]);
    assert_eq!(result[1].dense, dense_fixture[1]);
    assert_eq!(result[0].sparse.indices, sparse_fixture[0].indices);
    assert_eq!(result[1].sparse.indices, sparse_fixture[1].indices);
}

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

#[tokio::test]
async fn model_rss_per_worker_bytes_returns_zero_for_test_pool() {
    // Test helpers initialize model_rss_per_worker_bytes to 0 since no real
    // load_models() runs; the getter must reflect that.
    let pool = EmbedPool::closed_for_test();
    assert_eq!(pool.model_rss_per_worker_bytes(), 0);
    let pool2 = EmbedPool::with_fixed_responses(vec![], vec![]);
    assert_eq!(pool2.model_rss_per_worker_bytes(), 0);
}

// -----------------------------------------------------------------------
// Pure-math helper tests (no ORT session needed)
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// REPO_REVISION drift detection (ARC-3)
// -----------------------------------------------------------------------

fn extract_const_str(path: &str, const_name: &str) -> String {
    let prefix = format!("const {const_name}");
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            let start = trimmed.find('"').expect("missing opening quote");
            let end = trimmed[start + 1..]
                .find('"')
                .expect("missing closing quote");
            return trimmed[start + 1..start + 1 + end].to_string();
        }
    }
    panic!("{const_name} not found in {path}");
}

#[test]
fn repo_revision_consistent_across_all_copies() {
    let embedder = extract_const_str("src/embedder/model_files.rs", "REPO_REVISION");
    let bench = extract_const_str("benches/coreml/setup.rs", "REPO_REVISION");
    let example = extract_const_str("examples/fp16_eval.rs", "REPO_REVISION");

    assert_eq!(
        embedder, bench,
        "REPO_REVISION mismatch: src/embedder/model_files.rs ({embedder}) != benches/coreml/setup.rs ({bench})"
    );
    assert_eq!(
        embedder, example,
        "REPO_REVISION mismatch: src/embedder/model_files.rs ({embedder}) != examples/fp16_eval.rs ({example})"
    );
    assert_eq!(embedder.len(), 40, "REPO_REVISION should be a 40-char SHA");
    assert!(
        embedder.chars().all(|c| c.is_ascii_hexdigit()),
        "REPO_REVISION should be hexadecimal"
    );
}

#[test]
fn xenova_repo_revision_consistent_across_all_copies() {
    let embedder = extract_const_str("src/embedder/model_files.rs", "XENOVA_REPO_REVISION");
    let bench = extract_const_str("benches/coreml/setup.rs", "XENOVA_REPO_REVISION");

    assert_eq!(
        embedder, bench,
        "XENOVA_REPO_REVISION mismatch: \
         src/embedder/model_files.rs ({embedder}) != benches/coreml/setup.rs ({bench})"
    );
    assert_eq!(embedder.len(), 40);
    assert!(embedder.chars().all(|c| c.is_ascii_hexdigit()));
}

// -----------------------------------------------------------------------
// Benchmark corpus shape validation (TST-5)
// -----------------------------------------------------------------------

#[test]
fn benchmark_corpus_has_expected_shape() {
    let content = std::fs::read_to_string("benches/fixtures/corpus.json")
        .expect("corpus.json must be readable from project root");
    let corpus: serde_json::Value =
        serde_json::from_str(&content).expect("corpus.json must be valid JSON");

    assert!(corpus.get("metadata").is_some(), "must have 'metadata' key");
    assert!(
        corpus.get("scenarios").is_some(),
        "must have 'scenarios' key"
    );

    let sources = &corpus["metadata"]["sources"];
    assert_eq!(sources["knowledgebase_chunks"]["count"], 50);
    assert_eq!(sources["coordinator_vector_store"]["count"], 75);
    assert_eq!(sources["codekeeper_symbols"]["count"], 50);
    assert_eq!(sources["boundary_cases"]["count"], 9);

    let scenarios = corpus["scenarios"]
        .as_object()
        .expect("scenarios must be object");
    let expected: &[(&str, usize)] = &[
        ("document_chunks", 50),
        ("tool_descriptions", 75),
        ("code_symbols", 50),
        ("boundary_cases", 9),
    ];
    for &(name, count) in expected {
        let texts = scenarios
            .get(name)
            .and_then(|s| s.get("texts"))
            .and_then(|t| t.as_array())
            .unwrap_or_else(|| panic!("scenarios.{name}.texts must be an array"));
        assert_eq!(texts.len(), count);
    }

    let total: usize = scenarios
        .values()
        .filter_map(|s| s.get("texts").and_then(|t| t.as_array()).map(Vec::len))
        .sum();
    assert_eq!(total, 184, "total corpus texts should be 184");
}

// -----------------------------------------------------------------------
// EmbedStats and queue_depth
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// median_usize
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// model_rss_per_worker_bytes accessor
// -----------------------------------------------------------------------

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
