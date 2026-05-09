mod binpack;
mod config;
mod embedder;
mod error;
mod handler;
mod models;
mod probe;
mod state;
mod sysinfo;
mod weights;

use crate::binpack::CostModel;
use crate::embedder::{EmbedPool, WorkerConfig, OS_HEADROOM_BYTES};
use crate::state::{AppState, TuningInfo};
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

/// Runs after all workers are loaded:
/// 1. Detects available memory.
/// 2. Runs the startup probe on the leader worker (unless overridden).
/// 3. Derives the final cost model.
/// 4. Runs dense + sparse readiness probes.
/// 5. Sets `state.ready = true`.
// cast_precision_loss: available_bytes and total_workspace are ≤ ~28 GB (Fargate task
//   limit), well within f64's 2^52 mantissa (~4.5 PB); cfg_workers is ≤ 32.
// cast_possible_truncation: per_worker_workspace is a byte budget; truncating
//   sub-byte fractions is intentional and harmless.
// cast_sign_loss: total_workspace is derived from saturating_sub so it is always
//   ≥ 0 before the float multiplication.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn run_readiness_probe(
    init_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    state: Arc<AppState>,
    cfg_max_seq: usize,
    cfg_workers: usize,
    cfg_safety: f64,
    cost_model_override: Option<CostModel>,
) -> anyhow::Result<()> {
    init_handle
        .await
        .map_err(|e| anyhow::anyhow!("Worker pool task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("Worker pool initialization failed: {e}"))?;

    // --- Memory detection and probe ---
    let mem = sysinfo::detect_available_memory();
    info!(
        available_bytes = mem.available_bytes,
        source = %mem.source,
        "Memory detected"
    );

    let pre_rss = sysinfo::read_process_rss_bytes().unwrap_or(0);

    // We don't have an easy way to measure post-model-load RSS from here because
    // the leader worker loaded its model in a blocking thread before this runs.
    // The best approximation: difference between now and the pre-spawn baseline.
    let post_rss = sysinfo::read_process_rss_bytes().unwrap_or(pre_rss);
    let model_rss_per_worker = post_rss.saturating_sub(pre_rss);
    info!(
        model_rss_per_worker_mb = model_rss_per_worker / (1024 * 1024),
        "Estimated model RSS per worker"
    );

    // Compute per-worker workspace ceiling.
    let total_workspace = mem
        .available_bytes
        .saturating_sub(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_sub(OS_HEADROOM_BYTES);
    let per_worker_workspace =
        ((total_workspace as f64) * cfg_safety / (cfg_workers as f64)) as usize;

    let cost_model = if let Some(cm) = cost_model_override {
        info!(
            a = cm.a,
            b = cm.b,
            max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
            "Using pre-configured cost model (probe skipped)"
        );
        cm
    } else {
        let (a, b) = probe::run_probe(&state.pool, cfg_max_seq, per_worker_workspace).await;
        CostModel {
            a,
            b,
            max_workspace_bytes: per_worker_workspace,
        }
    };

    info!(
        a = cost_model.a,
        b = cost_model.b,
        max_workspace_mb = cost_model.max_workspace_bytes / (1024 * 1024),
        "Final cost model"
    );

    // Store tuning info in state for /health. OnceLock guarantees exactly one write.
    let tuning = TuningInfo::new(&cost_model, &mem, model_rss_per_worker);
    let _ = state.tuning.set(tuning); // always succeeds (first and only write)

    // Dense readiness probe.
    state
        .pool
        .dense(vec!["ready".into()])
        .await
        .map_err(|e| anyhow::anyhow!("Dense readiness probe failed: {e}"))?;

    // Sparse readiness probe.
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
        max_seq_length = cfg.max_seq_length,
        cache_dir = %cfg.cache_dir,
        idle_timeout_secs = cfg.idle_timeout.map(|d| d.as_secs()),
        model_variant = ?cfg.model_variant,
        memory_safety_factor = cfg.memory_safety_factor,
        auto_budget = cfg.cost_model_override.is_none(),
        "Starting bge-m3-embedding-server"
    );

    // Use conservative defaults for the initial WorkerConfig. If auto-budget
    // is enabled, the probe will derive better coefficients after the leader
    // loads. Followers are spawned before the probe runs but will receive the
    // same cost_model because they share the WorkerConfig value from spawn.
    //
    // Note: the probe updates the *running* cost_model after leader init by
    // sending Probe requests through the channel. Followers use whatever
    // cost_model was baked into their WorkerConfig at spawn time. To update
    // followers after the probe, the simplest approach is to bake the overridden
    // or conservative model at spawn and let followers inherit it; the probe only
    // needs to run on the leader to derive coefficients for the cost model stored
    // in AppState (used by /health). The binpacker in each worker uses the
    // cost_model from its own WorkerConfig — so the worker's cost_model needs to
    // be the same or we need to update it later.
    //
    // For v1: spawn all workers with conservative defaults; if auto-budget runs,
    // the derived cost_model is stored in AppState for /health display, but
    // workers keep using their spawn-time config. The bin-packer cost_model in
    // workers is what actually matters for safety. On the next server restart
    // (deploy), operators can set BGE_M3_DISABLE_AUTO_BUDGET + BGE_M3_TOKEN_BUDGET
    // to pin the probed values.
    //
    // This is a pragmatic v1 tradeoff: probe-derived tuning improves observability
    // immediately; actuating the derived cost_model into running workers requires
    // a more complex hot-reload mechanism deferred to a future PR.
    let initial_cost_model = cfg
        .cost_model_override
        .unwrap_or_else(|| CostModel::conservative(CostModel::DEFAULT_MAX_WORKSPACE));

    let (pool, init_handle) = EmbedPool::spawn(
        cfg.workers,
        PathBuf::from(&cfg.cache_dir),
        WorkerConfig {
            cost_model: initial_cost_model,
            idle_timeout: cfg.idle_timeout,
            model_variant: cfg.model_variant,
            max_seq_length: cfg.max_seq_length,
        },
    );

    let state = Arc::new(AppState {
        pool,
        ready: AtomicBool::new(false),
        max_batch: cfg.max_batch,
        total_workers: cfg.workers,
        max_seq_length: cfg.max_seq_length,
        tuning: std::sync::OnceLock::new(),
    });

    let app = build_router(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    info!(bind = %cfg.bind_addr, "Listening");

    let state_for_readiness = Arc::clone(&state);
    let cfg_max_seq = cfg.max_seq_length;
    let cfg_workers = cfg.workers;
    let cfg_safety = cfg.memory_safety_factor;
    let cost_model_override = cfg.cost_model_override;

    tokio::spawn(async move {
        if let Err(e) = run_readiness_probe(
            init_handle,
            state_for_readiness,
            cfg_max_seq,
            cfg_workers,
            cfg_safety,
            cost_model_override,
        )
        .await
        {
            tracing::error!("{e}");
            std::process::exit(1);
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
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
            max_seq_length: 8192,
            tuning: std::sync::OnceLock::new(),
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

    #[tokio::test]
    async fn readiness_probe_fails_when_init_returns_error() {
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
        let result = run_readiness_probe(handle, state, 8192, 2, 0.7, None).await;
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
        let result = run_readiness_probe(handle, state, 8192, 2, 0.7, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("panicked"));
    }

    #[tokio::test]
    async fn readiness_probe_fails_when_dense_probe_fails() {
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Ok::<(), anyhow::Error>(()) });
        let result = run_readiness_probe(handle, state, 8192, 2, 0.7, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn readiness_probe_does_not_set_ready_on_failure() {
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
        let _ = run_readiness_probe(handle, Arc::clone(&state), 8192, 2, 0.7, None).await;
        assert!(!state.ready.load(std::sync::atomic::Ordering::Acquire));
    }
}
