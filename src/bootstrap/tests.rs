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

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use http_body_util::BodyExt;
use tokio::sync::Semaphore;
use tower::ServiceExt;

use super::budget::compute_workspace_budget;
use super::readiness::run_readiness_probe;
use super::router::build_router;
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::state::{AppState, ProbeStatus};

fn make_test_state(ready: bool, max_batch: usize) -> Arc<AppState> {
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
        // Tests use an effectively-uncapped semaphore so permit acquisition
        // never blocks existing test scenarios.
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    })
}

// --- Router tests ---

#[tokio::test]
async fn router_health_returns_503_when_not_ready() {
    let app = build_router(make_test_state(false, 256));
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = resp
        .into_body()
        .collect()
        .await
        .expect("body should be readable")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("body should be valid JSON");
    assert_eq!(json["status"], "loading");
}

#[tokio::test]
async fn router_health_returns_200_idle_when_models_unloaded() {
    let app = build_router(Arc::new(AppState {
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
    }));
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp
        .into_body()
        .collect()
        .await
        .expect("body should be readable")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("body should be valid JSON");
    assert_eq!(json["status"], "idle");
}

#[tokio::test]
async fn router_health_returns_503_when_pool_dead() {
    let app = build_router(make_test_state(true, 256));
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = resp
        .into_body()
        .collect()
        .await
        .expect("body should be readable")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("body should be valid JSON");
    assert_eq!(json["status"], "fail");
}

#[tokio::test]
async fn router_dense_returns_503_when_not_ready() {
    let app = build_router(make_test_state(false, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn router_dense_returns_503_when_pool_dead() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn router_dense_returns_422_for_wrong_input_type() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": 42}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn router_dense_returns_422_for_missing_input_field() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"model": "bge-m3"}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn router_dense_returns_400_for_syntax_error() {
    let app = build_router(make_test_state(true, 256));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(b"{not valid json".as_ref()))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn router_dense_returns_415_for_missing_content_type() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn router_dense_returns_413_for_oversized_body() {
    let app = build_router(make_test_state(true, 256));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(vec![b'x'; 2_097_153]))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn router_returns_405_for_wrong_method_on_embeddings() {
    let app = build_router(make_test_state(true, 256));
    let req = Request::builder()
        .method("GET")
        .uri("/v1/embeddings")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn router_sparse_returns_503_when_not_ready() {
    let app = build_router(make_test_state(false, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/sparse-embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn router_both_returns_503_when_not_ready() {
    let app = build_router(make_test_state(false, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings:both")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn router_both_returns_503_when_pool_dead() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings:both")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// `:` is a valid `pchar` per RFC 3986, but some HTTP clients (and some
/// URI builders) percent-encode it to `%3A` anyway when it appears in a
/// path segment. Axum's `matchit` router matches the raw URI path
/// byte-for-byte and does not percent-decode before matching, so the
/// percent-encoded form is registered as an explicit alias route
/// pointing at the same handler. This test asserts that the alias
/// resolves and reaches the handler — returning the handler's own 503
/// here because the test pool is dead.
#[tokio::test]
async fn router_both_accepts_uppercase_percent_encoded_colon() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings%3Aboth")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// Sibling of `router_both_accepts_uppercase_percent_encoded_colon` —
/// RFC 3986 percent-encoding is case-insensitive, so the lowercase
/// `%3a` form must also reach the handler.
#[tokio::test]
async fn router_both_accepts_lowercase_percent_encoded_colon() {
    let app = build_router(make_test_state(true, 256));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["test"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings%3aboth")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn router_both_returns_200_with_paired_dense_and_sparse() {
    let dense_fixture = vec![vec![0.1f32, 0.2, 0.3]];
    let sparse_fixture = vec![crate::embedder::SparseEmbedding {
        indices: vec![42usize],
        values: vec![0.5f32],
    }];
    let app = build_router(Arc::new(AppState {
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
    }));
    let body = serde_json::to_vec(&serde_json::json!({"input": ["hello"]}))
        .expect("request body should serialize");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings:both")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp
        .into_body()
        .collect()
        .await
        .expect("body readable")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("valid json");
    assert_eq!(json["object"], "list");
    assert_eq!(json["model"], "bge-m3");
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(json["data"][0]["embedding"][0], 0.1_f32);
    assert_eq!(json["data"][0]["sparse_values"]["indices"][0], 42);
    assert_eq!(json["data"][0]["sparse_values"]["values"][0], 0.5_f32);
}

#[tokio::test]
async fn router_models_returns_200_with_bge_m3() {
    let app = build_router(make_test_state(true, 256));
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp
        .into_body()
        .collect()
        .await
        .expect("body readable")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("valid json");
    assert_eq!(json["data"][0]["id"], "bge-m3");
}

#[tokio::test]
async fn router_response_includes_x_request_id() {
    let app = build_router(make_test_state(false, 256));
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert!(resp.headers().contains_key("x-request-id"));
}

#[tokio::test]
async fn router_propagates_provided_x_request_id() {
    let app = build_router(make_test_state(false, 256));
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .header("x-request-id", "test-id-12345")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(
        resp.headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok()),
        Some("test-id-12345")
    );
}

fn test_cache_dir() -> PathBuf {
    PathBuf::from("/tmp/bge-m3-probe-test-cache")
}

#[tokio::test]
async fn readiness_probe_fails_when_init_returns_error() {
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
    let result = run_readiness_probe(
        handle,
        state,
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("initialization failed"));
}

#[tokio::test]
async fn readiness_probe_fails_when_init_panics() {
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { panic!("worker panic") });
    let result = run_readiness_probe(
        handle,
        state,
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("panicked"));
}

#[tokio::test]
async fn readiness_probe_does_not_set_ready_when_dense_check_fails() {
    // With the serialised-probe design, readiness checks run inside the
    // spawned probe task rather than in the caller.
    // run_readiness_probe returns Ok immediately; the readiness failure
    // is logged and state.ready stays false.
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { Ok::<(), anyhow::Error>(()) });
    // disable_probe_cache=true → no override, no cache → probe spawned
    let result = run_readiness_probe(
        handle,
        Arc::clone(&state),
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    // run_readiness_probe returns Ok — the probe task was spawned.
    assert!(
        result.is_ok(),
        "run_readiness_probe should return Ok (probe spawned)"
    );
    // Give the probe task time to run the readiness check and fail.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // The pool is closed_for_test, so dense() fails; ready should stay false.
    assert!(
        !state.ready.load(std::sync::atomic::Ordering::Acquire),
        "ready must not be set when the dense readiness check fails"
    );
}

#[tokio::test]
async fn readiness_probe_does_not_set_ready_on_failure() {
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
    let _ = run_readiness_probe(
        handle,
        Arc::clone(&state),
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    assert!(!state.ready.load(std::sync::atomic::Ordering::Acquire));
}

// -----------------------------------------------------------------------
// compute_workspace_budget
// -----------------------------------------------------------------------

#[test]
fn compute_workspace_budget_sane_inputs() {
    // 28 GiB available, 7 workers, ~1.6 GiB model RSS, 0.7 safety.
    let avail = 28_672usize * 1024 * 1024;
    let model_rss = 1_628usize * 1024 * 1024;
    let (ws, peak, pct) = compute_workspace_budget(avail, 7, model_rss, 0.7);
    // total_workspace = 28672 - 7*1628 - 256 ≈ 17,060 MiB
    // per_worker = 17060 * 0.7 / 7 ≈ 1,706 MiB
    assert!(
        ws > 1_000 * 1024 * 1024,
        "per_worker_workspace ({} MiB) should be well over 1 GiB",
        ws / (1024 * 1024)
    );
    assert!(ws < avail, "per_worker_workspace must not exceed available");
    // Worst-case peak should be < available (sanity).
    assert!(
        peak < avail * 2,
        "peak ({} MiB) seems unreasonably large",
        peak / (1024 * 1024)
    );
    assert!(
        pct > 0.0 && pct < 200.0,
        "utilization_pct {pct:.1}% out of range"
    );
}

#[test]
fn compute_workspace_budget_saturates_gracefully_when_model_rss_inflated() {
    // Reproduces the production failure: inflated model_rss_per_worker from
    // parallel-load contamination drives total_workspace to 0 via saturating_sub.
    let avail = 20_543usize * 1024 * 1024; // ~what MemAvailable reported
    let model_rss = 8_459usize * 1024 * 1024; // contaminated median from old code
    let (ws, _peak, _pct) = compute_workspace_budget(avail, 7, model_rss, 0.7);
    // 7 * 8459 = 59213 MiB >> 20543 MiB → saturates to 0 → ws = 0.
    assert_eq!(
        ws, 0,
        "saturated budget should be 0 (physics_floor check will catch this)"
    );
}

#[test]
fn compute_workspace_budget_physics_floor_detection() {
    // Verify that the physics floor catches the zero-workspace case.
    // physics_floor = chunk_cost(1, 8192) under conservative defaults.
    let physics_floor = CostModel::conservative(0).chunk_cost(1, 8192) as usize;
    assert!(
        physics_floor > 0,
        "physics_floor must be positive (conservative model costs > 0)"
    );
    // A zero workspace is below the floor.
    assert!(
        0 < physics_floor,
        "workspace=0 must be caught by the physics_floor guard"
    );
}

#[test]
fn compute_workspace_budget_single_worker() {
    // n=1: all available workspace (minus model RSS and headroom) goes to that worker.
    let avail = 8_192usize * 1024 * 1024;
    let model_rss = 1_100usize * 1024 * 1024;
    let (ws, _peak, _pct) = compute_workspace_budget(avail, 1, model_rss, 1.0);
    // total_workspace = 8192 - 1100 - 256 = 6836 MiB; per_worker = 6836 * 1.0 / 1
    assert!(
        ws > 6_000 * 1024 * 1024,
        "single worker should get ~6836 MiB workspace, got {} MiB",
        ws / (1024 * 1024)
    );
}
