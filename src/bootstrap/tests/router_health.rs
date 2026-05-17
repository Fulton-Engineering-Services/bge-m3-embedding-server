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

#[tokio::test]
async fn router_health_returns_503_when_not_ready() {
    let app = build_router(make_test_state(false, 256), 33_554_432);
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
    let app = build_router(
        Arc::new(AppState {
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
        }),
        33_554_432,
    );
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
    let app = build_router(make_test_state(true, 256), 33_554_432);
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
