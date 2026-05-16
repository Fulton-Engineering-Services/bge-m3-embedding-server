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

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use tower::ServiceExt;

use super::super::router::build_router;
use super::helpers::make_test_state;

// ── dense endpoint ────────────────────────────────────────────────────────

#[tokio::test]
async fn router_dense_returns_503_when_not_ready() {
    let app = build_router(make_test_state(false, 256), 33_554_432);
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
    let app = build_router(make_test_state(true, 256), 33_554_432);
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
    let app = build_router(make_test_state(true, 256), 33_554_432);
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
    let app = build_router(make_test_state(true, 256), 33_554_432);
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
    let app = build_router(make_test_state(true, 256), 33_554_432);
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
    let app = build_router(make_test_state(true, 256), 33_554_432);
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
    // Use a 1 KiB limit so the test doesn't need to allocate 32+ MiB of body.
    let app = build_router(make_test_state(true, 256), 1024);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(vec![b'x'; 1025]))
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn router_returns_405_for_wrong_method_on_embeddings() {
    let app = build_router(make_test_state(true, 256), 33_554_432);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/embeddings")
        .body(Body::empty())
        .expect("request should build");
    let resp: Response = app.oneshot(req).await.expect("router should respond");
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}
