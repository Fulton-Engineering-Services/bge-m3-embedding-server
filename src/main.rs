mod config;
mod embedder;
mod error;
mod handler;
mod models;
mod state;

use axum::extract::DefaultBodyLimit;
use axum::http::HeaderValue;
use axum::{routing::get, routing::post, Router};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::TraceLayer;
use tracing::info;

use config::Config;
use embedder::EmbedPool;
use state::AppState;

#[derive(Clone, Default)]
struct UuidRequestId;

impl MakeRequestId for UuidRequestId {
    fn make_request_id<B>(&mut self, _request: &axum::http::Request<B>) -> Option<RequestId> {
        let id = uuid::Uuid::new_v4().to_string();
        HeaderValue::from_str(&id).ok().map(RequestId::new)
    }
}

pub(crate) fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(handler::dense_embeddings))
        .route("/v1/sparse-embeddings", post(handler::sparse_embeddings))
        .route("/v1/models", get(handler::models))
        .route("/health", get(handler::health))
        .layer(DefaultBodyLimit::max(2_097_152))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(TraceLayer::new_for_http())
        .layer(SetRequestIdLayer::x_request_id(UuidRequestId))
        .with_state(state)
}

async fn run_readiness_probe(
    init_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    state: Arc<AppState>,
) -> anyhow::Result<()> {
    init_handle
        .await
        .map_err(|e| anyhow::anyhow!("Worker pool task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("Worker pool initialization failed: {e}"))?;

    state
        .pool
        .dense(vec!["ready".into()])
        .await
        .map_err(|e| anyhow::anyhow!("Dense readiness probe failed: {e}"))?;

    state
        .pool
        .sparse(vec!["ready".into()])
        .await
        .map_err(|e| anyhow::anyhow!("Sparse readiness probe failed: {e}"))?;

    state.ready.store(true, Ordering::Release);
    tracing::info!("Models ready — accepting requests");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let log_format = std::env::var("BGE_M3_LOG_FORMAT").unwrap_or_default();
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    if log_format.eq_ignore_ascii_case("json") {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    }

    let cfg = Config::from_env();

    info!(
        bind = %cfg.bind_addr,
        workers = cfg.workers,
        max_batch = cfg.max_batch,
        cache_dir = %cfg.cache_dir,
        idle_timeout_secs = cfg.idle_timeout.map(|d| d.as_secs()),
        "Starting bge-m3-axum-fastembed-rs"
    );

    let (pool, init_handle) =
        EmbedPool::spawn(cfg.workers, PathBuf::from(&cfg.cache_dir), cfg.idle_timeout, cfg.onnx_batch_size);

    let state = Arc::new(AppState {
        pool,
        ready: AtomicBool::new(false),
        max_batch: cfg.max_batch,
        total_workers: cfg.workers,
    });

    let app = build_router(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    info!(bind = %cfg.bind_addr, "Listening");

    let state_for_readiness = Arc::clone(&state);
    tokio::spawn(async move {
        if let Err(e) = run_readiness_probe(init_handle, state_for_readiness).await {
            tracing::error!("{e}");
            std::process::exit(1);
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use http_body_util::BodyExt;
    use std::sync::atomic::AtomicBool;
    use tower::ServiceExt;

    fn make_test_state(ready: bool, max_batch: usize) -> Arc<AppState> {
        Arc::new(AppState {
            pool: EmbedPool::closed_for_test(),
            ready: AtomicBool::new(ready),
            max_batch,
            total_workers: 2,
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
        let json: serde_json::Value =
            serde_json::from_slice(&body).expect("body should be valid JSON");
        assert_eq!(json["status"], "loading");
    }

    #[tokio::test]
    async fn router_health_returns_200_idle_when_models_unloaded() {
        let app = build_router(Arc::new(AppState {
            pool: EmbedPool::idle_for_test(),
            ready: AtomicBool::new(true),
            max_batch: 256,
            total_workers: 1,
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
        let json: serde_json::Value =
            serde_json::from_slice(&body).expect("body should be valid JSON");
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
        let json: serde_json::Value =
            serde_json::from_slice(&body).expect("body should be valid JSON");
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

    // --- RequestId middleware tests ---

    #[tokio::test]
    async fn router_response_includes_x_request_id() {
        let app = build_router(make_test_state(false, 256));
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .expect("request should build");
        let resp: Response = app.oneshot(req).await.expect("router should respond");
        assert!(
            resp.headers().contains_key("x-request-id"),
            "response should include X-Request-ID header"
        );
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
            Some("test-id-12345"),
            "response should echo provided X-Request-ID"
        );
    }

    // --- run_readiness_probe tests ---

    #[tokio::test]
    async fn readiness_probe_fails_when_init_returns_error() {
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
        let result = run_readiness_probe(handle, state).await;
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
        let result = run_readiness_probe(handle, state).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("panicked"));
    }

    #[tokio::test]
    async fn readiness_probe_fails_when_dense_probe_fails() {
        // init succeeds but pool is closed → dense probe fails immediately
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Ok::<(), anyhow::Error>(()) });
        let result = run_readiness_probe(handle, state).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn readiness_probe_does_not_set_ready_on_failure() {
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
        let _ = run_readiness_probe(handle, Arc::clone(&state)).await;
        assert!(!state.ready.load(std::sync::atomic::Ordering::Acquire));
    }
}
