use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::body::to_bytes;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{extract::State, Json};
use tokio::sync::Semaphore;

use super::common::{check_ready, validate_input, MAX_STRING_CHARS};
use super::{both_embeddings, dense_embeddings, health, models, sparse_embeddings};
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::error::AppError;
use crate::models::{DenseRequest, DualRequest, SparseRequest};
use crate::state::{AppState, ProbeStatus};

fn make_state(ready: bool, max_batch: usize) -> Arc<AppState> {
    Arc::new(AppState {
        pool: EmbedPool::closed_for_test(),
        ready: AtomicBool::new(ready),
        max_batch,
        total_workers: 2,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    })
}

// --- validate_input ---

#[test]
fn validate_input_rejects_empty() {
    let result = validate_input(&[], 10);
    assert!(
        matches!(result, Err(AppError::InvalidRequest(msg)) if msg == "input must not be empty")
    );
}

#[test]
fn validate_input_rejects_over_batch() {
    let texts: Vec<String> = (0..5).map(|i| format!("text {i}")).collect();
    let result = validate_input(&texts, 3);
    assert!(
        matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains('5') && msg.contains('3'))
    );
}

#[test]
fn validate_input_accepts_at_limit() {
    let texts: Vec<String> = (0..3).map(|i| format!("text {i}")).collect();
    assert!(validate_input(&texts, 3).is_ok());
}

#[test]
fn validate_input_accepts_single() {
    let texts = vec!["hello".to_string()];
    assert!(validate_input(&texts, 256).is_ok());
}

// --- check_ready ---

#[test]
fn check_ready_returns_err_when_not_ready() {
    let state = make_state(false, 10);
    let result = check_ready(&state);
    assert!(matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "model not ready"));
}

#[test]
fn check_ready_returns_err_when_pool_dead() {
    let state = make_state(true, 10);
    let result = check_ready(&state);
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}

// --- health handler ---

#[tokio::test]
async fn health_returns_fail_when_ready_but_pool_dead() {
    let state = make_state(true, 256);
    let response = health(State(state)).await.into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should be readable");
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response status should be parseable");
    assert_eq!(body["status"], "fail");
    assert_eq!(body["workers"]["live"], 0);
    assert_eq!(body["workers"]["total"], 2);
}

#[tokio::test]
async fn health_returns_idle_when_models_unloaded() {
    let state = Arc::new(AppState {
        pool: EmbedPool::idle_for_test(),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let response = health(State(state)).await.into_response();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should be readable");
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response status should be parseable");
    assert_eq!(body["status"], "idle");
    assert_eq!(body["workers"]["live"], 1);
    assert_eq!(body["workers"]["total"], 1);
}

#[tokio::test]
async fn health_returns_503_when_not_ready() {
    let state = make_state(false, 10);
    let response = health(State(state)).await.into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should be readable");
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response status should be parseable");
    assert_eq!(body["status"], "loading");
}

#[tokio::test]
async fn health_ok_includes_max_seq_length() {
    use crate::state::TuningInfo;
    use crate::sysinfo::{MemoryReading, MemorySource};

    let mem = MemoryReading {
        available_bytes: 8_000_000_000,
        source: MemorySource::CgroupV2,
    };
    let tuning = TuningInfo::new(&mem, 500_000_000, 22_000_000_000, 78.5);

    let tuning_lock = std::sync::OnceLock::new();
    let _ = tuning_lock.set(tuning);
    let fitted_cm = CostModel {
        a: 18_432.0,
        b: 6.2,
        max_workspace_bytes: 1_073_741_824,
    };
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: tuning_lock,
        cost_model: Arc::new(ArcSwap::from_pointee(fitted_cm)),
        probe_status: AtomicU8::new(ProbeStatus::Complete as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let response = health(State(state)).await.into_response();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["max_seq_length"], 8192);
    assert!(
        body["tuning"].is_object(),
        "tuning should always be present"
    );
    assert_eq!(body["tuning"]["probe_status"], "complete");
    assert_eq!(body["tuning"]["a_bytes_per_token"], 18_432.0);
    assert_eq!(body["tuning"]["b_bytes_per_token_sq"], 6.2);
}

// --- dense_embeddings handler (validation paths only) ---

#[tokio::test]
async fn dense_embeddings_rejects_when_not_ready() {
    use crate::models::TextInput;
    let state = make_state(false, 256);
    let req = DenseRequest {
        input: TextInput(vec!["hello".to_string()]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn dense_embeddings_rejects_when_pool_dead() {
    use crate::models::TextInput;
    let state = make_state(true, 256);
    let req = DenseRequest {
        input: TextInput(vec!["hello".to_string()]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}

#[tokio::test]
async fn dense_embeddings_rejects_empty_input() {
    use crate::models::TextInput;
    let state = make_state(false, 256);
    let req = DenseRequest {
        input: TextInput(vec![]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn dense_embeddings_rejects_over_batch() {
    use crate::models::TextInput;
    let state = make_state(false, 2);
    let req = DenseRequest {
        input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

// --- sparse_embeddings handler (validation paths only) ---

#[tokio::test]
async fn sparse_embeddings_rejects_when_not_ready() {
    use crate::models::TextInput;
    let state = make_state(false, 256);
    let req = SparseRequest {
        input: TextInput(vec!["hello".to_string()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn sparse_embeddings_rejects_when_pool_dead() {
    use crate::models::TextInput;
    let state = make_state(true, 256);
    let req = SparseRequest {
        input: TextInput(vec!["hello".to_string()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}

#[tokio::test]
async fn sparse_embeddings_rejects_empty_input() {
    use crate::models::TextInput;
    let state = make_state(false, 256);
    let req = SparseRequest {
        input: TextInput(vec![]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn models_handler_returns_bge_m3_entry() {
    let state = make_state(true, 256);
    let response = models(State(state)).await.into_response();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body readable");
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("valid json");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "bge-m3");
    assert_eq!(body["data"][0]["object"], "model");
}

#[tokio::test]
async fn sparse_embeddings_rejects_over_batch() {
    use crate::models::TextInput;
    let state = make_state(false, 2);
    let req = SparseRequest {
        input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

// --- handler validation with ready pool (TST-5) ---

#[tokio::test]
async fn dense_embeddings_returns_invalid_request_for_empty_input_when_ready() {
    use crate::models::TextInput;
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = DenseRequest {
        input: TextInput(vec![]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("empty")),
        "expected InvalidRequest for empty input, got: {result:?}"
    );
}

#[tokio::test]
async fn dense_embeddings_returns_invalid_request_for_over_batch_when_ready() {
    use crate::models::TextInput;
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 2,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = DenseRequest {
        input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
        "expected InvalidRequest for over-batch, got: {result:?}"
    );
}

#[tokio::test]
async fn sparse_embeddings_returns_invalid_request_for_empty_input_when_ready() {
    use crate::models::TextInput;
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = SparseRequest {
        input: TextInput(vec![]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("empty")),
        "expected InvalidRequest for empty input, got: {result:?}"
    );
}

#[tokio::test]
async fn sparse_embeddings_returns_invalid_request_for_over_batch_when_ready() {
    use crate::models::TextInput;
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 2,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = SparseRequest {
        input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
        "expected InvalidRequest for over-batch, got: {result:?}"
    );
}

// --- per-string length validation ---

#[test]
fn validate_input_rejects_oversized_string() {
    let long = "x".repeat(MAX_STRING_CHARS + 1);
    let texts = vec![long];
    let result = validate_input(&texts, 256);
    assert!(
        matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains("exceeds maximum"))
    );
}

#[test]
fn validate_input_accepts_at_char_limit() {
    let at_limit = "x".repeat(MAX_STRING_CHARS);
    let texts = vec![at_limit];
    assert!(
        validate_input(&texts, 256).is_ok(),
        "string exactly at MAX_STRING_CHARS should be accepted"
    );
}

// --- happy-path tests (using fixture pool) ---

#[tokio::test]
async fn dense_embeddings_returns_correct_shape() {
    use crate::models::TextInput;
    let fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(fixture, vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = DenseRequest {
        input: TextInput(vec!["hello".into(), "world".into()]),
        model: None,
    };
    let result = dense_embeddings(State(state), Json(req)).await;
    assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
    let Json(resp) = result.expect("dense_embeddings should succeed");
    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2, 0.3]);
    assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
    assert_eq!(resp.data[0].index, 0);
    assert_eq!(resp.data[1].index, 1);
    assert_eq!(resp.object, "list");
    assert_eq!(resp.model, "bge-m3");
}

// --- both_embeddings handler ---

#[tokio::test]
async fn both_embeddings_rejects_when_not_ready() {
    use crate::models::TextInput;
    let state = make_state(false, 256);
    let req = DualRequest {
        input: TextInput(vec!["hello".to_string()]),
        model: None,
    };
    let result = both_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn both_embeddings_rejects_when_pool_dead() {
    use crate::models::TextInput;
    let state = make_state(true, 256);
    let req = DualRequest {
        input: TextInput(vec!["hello".to_string()]),
        model: None,
    };
    let result = both_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}

#[tokio::test]
async fn both_embeddings_returns_invalid_request_for_empty_input_when_ready() {
    use crate::models::TextInput;
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = DualRequest {
        input: TextInput(vec![]),
        model: None,
    };
    let result = both_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("empty")),
        "expected InvalidRequest for empty input, got: {result:?}"
    );
}

#[tokio::test]
async fn both_embeddings_returns_invalid_request_for_over_batch_when_ready() {
    use crate::models::TextInput;
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 2,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = DualRequest {
        input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
        model: None,
    };
    let result = both_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
        "expected InvalidRequest for over-batch, got: {result:?}"
    );
}

#[tokio::test]
async fn both_embeddings_returns_correct_shape() {
    use crate::models::TextInput;
    let dense_fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let sparse_fixture = vec![
        crate::embedder::SparseEmbedding {
            indices: vec![42usize],
            values: vec![0.5f32],
        },
        crate::embedder::SparseEmbedding {
            indices: vec![100usize, 200usize],
            values: vec![0.7f32, 0.9f32],
        },
    ];
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(dense_fixture, sparse_fixture),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = DualRequest {
        input: TextInput(vec!["hello".into(), "world".into()]),
        model: None,
    };
    let result = both_embeddings(State(state), Json(req)).await;
    assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
    let Json(resp) = result.expect("both_embeddings should succeed");
    assert_eq!(resp.object, "list");
    assert_eq!(resp.model, "bge-m3");
    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].index, 0);
    assert_eq!(resp.data[1].index, 1);
    assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2, 0.3]);
    assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
    assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
    assert_eq!(resp.data[1].sparse_values.indices, vec![100u32, 200u32]);
    assert_eq!(resp.data[1].sparse_values.values, vec![0.7f32, 0.9f32]);
}

#[tokio::test]
async fn sparse_embeddings_returns_correct_shape() {
    use crate::models::TextInput;
    let sparse_fixture = vec![crate::embedder::SparseEmbedding {
        indices: vec![42usize],
        values: vec![0.5f32],
    }];
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], sparse_fixture),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = SparseRequest {
        input: TextInput(vec!["hello".into()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        result.is_ok(),
        "expected Ok but got error from sparse handler"
    );
    let Json(resp) = result.expect("sparse_embeddings should succeed");
    assert_eq!(resp.data.len(), 1);
    assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
    assert_eq!(resp.data[0].sparse_values.values, vec![0.5f32]);
}

// -----------------------------------------------------------------------
// Permit-gating tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn dense_embeddings_blocks_when_no_permits_available() {
    use crate::models::TextInput;
    use std::time::Duration;
    // Semaphore with 0 permits — all requests must queue.
    let permits = Arc::new(Semaphore::new(0));
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![vec![0.1f32]], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::clone(&permits),
    });
    let req = DenseRequest {
        input: TextInput(vec!["hello".into()]),
        model: None,
    };
    // Fire the request in a background task.
    let state_clone = Arc::clone(&state);
    let handle = tokio::spawn(async move { dense_embeddings(State(state_clone), Json(req)).await });
    // Give the task time to start and attempt permit acquisition.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !handle.is_finished(),
        "request should still be blocked on 0 permits"
    );
    // Release a permit — request should now complete.
    permits.add_permits(1);
    let result = tokio::time::timeout(Duration::from_secs(1), handle)
        .await
        .expect("request should complete after permit is released")
        .expect("task should not panic");
    assert!(
        result.is_ok(),
        "dense_embeddings should succeed once permitted"
    );
}

#[tokio::test]
async fn request_permits_rises_after_probe_complete() {
    // Verify the semaphore protocol: starting at N-1 permits, add_permits(1)
    // after probe completion brings it to N.
    let n: usize = 7;
    let initial = n.saturating_sub(1).max(1);
    let permits = Arc::new(Semaphore::new(initial));
    assert_eq!(
        permits.available_permits(),
        initial,
        "initial permits should be cfg_workers - 1"
    );
    permits.add_permits(1);
    assert_eq!(
        permits.available_permits(),
        n,
        "after probe completes, permits should equal cfg_workers"
    );
}

// -----------------------------------------------------------------------
// Worst-case budget invariant
// -----------------------------------------------------------------------

#[test]
fn worst_case_peak_below_available_when_correctly_budgeted() {
    // Production config: 28 GB Fargate task, 7 fp16 workers, 0.7 safety.
    // With accurate model_rss_per_worker the formula must stay under the limit.
    use crate::embedder::OS_HEADROOM_BYTES;

    let available_bytes: usize = 28 * 1024 * 1024 * 1024; // 28 GB
    let cfg_workers: usize = 7;
    let model_rss_per_worker: usize = 1_080_000_000; // ~1.08 GB fp16
    let memory_safety_factor: f64 = 0.7;

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let total_workspace = available_bytes
        .saturating_sub(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_sub(OS_HEADROOM_BYTES);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let per_worker_workspace =
        ((total_workspace as f64) * memory_safety_factor / (cfg_workers as f64)) as usize;

    let worst_case_peak = cfg_workers
        .saturating_mul(per_worker_workspace)
        .saturating_add(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_add(OS_HEADROOM_BYTES);

    assert!(
        worst_case_peak < available_bytes,
        "worst_case_peak ({} MB) must be below available ({} MB)",
        worst_case_peak / (1024 * 1024),
        available_bytes / (1024 * 1024),
    );

    #[allow(clippy::cast_precision_loss)]
    let utilization_pct = (worst_case_peak as f64 / available_bytes as f64) * 100.0;
    assert!(
        utilization_pct < 90.0,
        "utilization_pct ({utilization_pct:.1}%) must be below the 90% WARN threshold"
    );
}
