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

use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use http_body_util::BodyExt;
use tokio::sync::Semaphore;
use tower::ServiceExt;

use super::super::router::build_router;
use super::helpers::make_test_state;
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::state::{AppState, ProbeStatus};

// ── sparse / both endpoints ───────────────────────────────────────────────

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

/// RFC 3986 percent-encoding is case-insensitive, so the uppercase
/// `%3A` form must also reach the handler.
#[tokio::test]
async fn router_both_accepts_percent_encoded_colon() {
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

// ── models endpoint ───────────────────────────────────────────────────────

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

// ── request-id propagation ────────────────────────────────────────────────

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
