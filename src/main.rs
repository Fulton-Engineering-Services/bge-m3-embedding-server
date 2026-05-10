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
use crate::state::{AppState, ProbeStatus, TuningInfo};
use arc_swap::ArcSwap;
use axum::extract::DefaultBodyLimit;
use axum::http::HeaderValue;
use axum::{routing::get, routing::post, Router};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;
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
        .route("/v1/embeddings:both", post(handler::both_embeddings))
        .route("/v1/models", get(handler::models))
        .route("/health", get(handler::health))
        .layer(DefaultBodyLimit::max(2_097_152))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(TraceLayer::new_for_http())
        .layer(SetRequestIdLayer::x_request_id(UuidRequestId))
        .with_state(state)
}

/// Runs after all workers finish loading their model instances.
///
/// # Sequence
///
/// 1. Wait for worker pool initialisation to finish.
/// 2. Detect available memory; write static [`TuningInfo`] to `OnceLock`.
/// 3. Resolve the cost model — one of three paths:
///    - cost-model override set: apply immediately, `probe_status = Disabled`.
///    - EFS cache hit: apply cached `(a, b)` via `ArcSwap`, `probe_status = CacheHit`.
///    - cache miss: set `probe_status = Running`, launch background probe task.
/// 4. Run dense + sparse readiness calls to confirm the worker pool is healthy.
/// 5. Flip `state.ready = true` — `/health` returns `200 ok` from this point on.
///    If the probe is still running in the background, the bin-packer uses
///    conservative defaults until the `ArcSwap` is updated (typically ~120 s).
///
// cast_precision_loss: available_bytes and total_workspace are ≤ ~28 GB (Fargate task
//   limit), well within f64's 2^52 mantissa (~4.5 PB); cfg_workers is ≤ 32.
// cast_possible_truncation: per_worker_workspace is a byte budget; truncating
//   sub-byte fractions is intentional and harmless.
// cast_sign_loss: total_workspace is derived from saturating_sub so it is always
//   ≥ 0 before the float multiplication.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
async fn run_readiness_probe(
    init_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    state: Arc<AppState>,
    cfg_max_seq: usize,
    cfg_workers: usize,
    cfg_safety: f64,
    cost_model_override: Option<CostModel>,
    cache_dir: PathBuf,
    model_variant_str: String,
    disable_probe_cache: bool,
) -> anyhow::Result<()> {
    init_handle
        .await
        .map_err(|e| anyhow::anyhow!("Worker pool task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("Worker pool initialization failed: {e}"))?;

    // --- Memory detection ---
    let mem = sysinfo::detect_available_memory();
    info!(
        available_bytes = mem.available_bytes,
        source = %mem.source,
        "Memory detected"
    );

    // Use the per-worker RSS delta reported by each worker during load_models().
    // This is measured inside spawn_blocking immediately around the ORT session
    // creation, so it accurately captures the model-weight footprint rather than
    // the noisy pre/post snapshot that was taken here before (which read RSS
    // twice in the same instant after all workers had already loaded and
    // produced model_rss_per_worker ≈ 0 — the root cause of the 2026-05-09 OOM).
    let model_rss_per_worker = state.pool.model_rss_max_bytes();
    info!(
        model_rss_per_worker_mb = model_rss_per_worker / (1024 * 1024),
        "Measured model RSS per worker (max across all workers)"
    );

    // Compute per-worker workspace ceiling.
    let total_workspace = mem
        .available_bytes
        .saturating_sub(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_sub(OS_HEADROOM_BYTES);
    let per_worker_workspace =
        ((total_workspace as f64) * cfg_safety / (cfg_workers as f64)) as usize;

    // Compute and log worst-case peak memory when all workers run simultaneously
    // at their per-worker budget ceiling.  This is the number that must stay
    // below available_bytes to avoid OOM.
    let worst_case_peak = cfg_workers
        .saturating_mul(per_worker_workspace)
        .saturating_add(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_add(OS_HEADROOM_BYTES);
    #[allow(clippy::cast_precision_loss)]
    let utilization_pct = if mem.available_bytes > 0 {
        (worst_case_peak as f64 / mem.available_bytes as f64) * 100.0
    } else {
        0.0
    };
    info!(
        worst_case_peak_mb = worst_case_peak / (1024 * 1024),
        available_mb = mem.available_bytes / (1024 * 1024),
        utilization_pct = format!("{utilization_pct:.1}"),
        per_worker_workspace_mb = per_worker_workspace / (1024 * 1024),
        "Workspace budget computed (worst-case all-workers-peak)"
    );
    if utilization_pct > 90.0 {
        tracing::warn!(
            utilization_pct = format!("{utilization_pct:.1}"),
            "Worst-case workspace peak exceeds 90% of available memory; \
             consider lowering BGE_M3_MEMORY_SAFETY_FACTOR or BGE_M3_WORKERS"
        );
    }

    // Write static memory + budget info now so /health always shows these fields
    // even while the background probe is still running.
    let _ = state.tuning.set(TuningInfo::new(
        &mem,
        model_rss_per_worker,
        worst_case_peak,
        utilization_pct,
    ));

    // --- Cost model resolution ---
    if let Some(cm) = cost_model_override {
        info!(
            a = cm.a,
            b = cm.b,
            max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
            "Using pre-configured cost model (probe skipped)"
        );
        state.cost_model.store(Arc::new(cm));
        state
            .probe_status
            .store(ProbeStatus::Disabled as u8, Ordering::Release);
        // No background probe — release the reserved worker slot immediately.
        state.request_permits.add_permits(1);
    } else if !disable_probe_cache {
        // Try to load cached coefficients from EFS.
        if let Some((a, b)) =
            probe::try_load_probe_cache(&cache_dir, &model_variant_str, cfg_max_seq)
        {
            let cm = CostModel {
                a,
                b,
                max_workspace_bytes: per_worker_workspace,
            };
            info!(
                a,
                b,
                max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
                "Cost model loaded from EFS cache"
            );
            state.cost_model.store(Arc::new(cm));
            state
                .probe_status
                .store(ProbeStatus::CacheHit as u8, Ordering::Release);
            // Cache hit — no probe needed, release the reserved worker slot.
            state.request_permits.add_permits(1);
        } else {
            // Cache miss — launch background probe.
            state
                .probe_status
                .store(ProbeStatus::Running as u8, Ordering::Release);
            let state_bg = Arc::clone(&state);
            let model_variant_bg = model_variant_str.clone();
            tokio::spawn(async move {
                let (a, b) =
                    probe::run_probe(&state_bg.pool, cfg_max_seq, per_worker_workspace).await;
                let cm = CostModel {
                    a,
                    b,
                    max_workspace_bytes: per_worker_workspace,
                };
                info!(
                    a = cm.a,
                    b = cm.b,
                    max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
                    "Background probe complete — updating cost model"
                );
                state_bg.cost_model.store(Arc::new(cm));
                // Distinguish real fit from conservative fallback.
                let status = if (a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON
                    && (b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON
                {
                    ProbeStatus::Failed
                } else {
                    probe::save_probe_cache(&cache_dir, &model_variant_bg, cfg_max_seq, a, b);
                    ProbeStatus::Complete
                };
                state_bg.probe_status.store(status as u8, Ordering::Release);
                // Probe is done (success or failure) — release the reserved worker slot
                // so all cfg_workers permits are now available to request traffic.
                state_bg.request_permits.add_permits(1);
                info!(probe_status = status.as_str(), "Probe status updated");
            });
        }
    } else {
        // BGE_M3_DISABLE_PROBE_CACHE=1 but no override — run probe without caching.
        state
            .probe_status
            .store(ProbeStatus::Running as u8, Ordering::Release);
        let state_bg = Arc::clone(&state);
        let model_variant_bg = model_variant_str.clone();
        tokio::spawn(async move {
            let (a, b) = probe::run_probe(&state_bg.pool, cfg_max_seq, per_worker_workspace).await;
            let cm = CostModel {
                a,
                b,
                max_workspace_bytes: per_worker_workspace,
            };
            state_bg.cost_model.store(Arc::new(cm));
            let status = if (a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON
                && (b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON
            {
                ProbeStatus::Failed
            } else {
                ProbeStatus::Complete
            };
            state_bg.probe_status.store(status as u8, Ordering::Release);
            // Probe done — release the reserved worker slot.
            state_bg.request_permits.add_permits(1);
            info!(
                probe_status = status.as_str(),
                model_variant = model_variant_bg,
                "Probe complete (cache disabled)"
            );
        });
    }

    // --- Readiness checks ---
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

    // Flip the ready flag — /health starts returning 200 from here.
    // If the background probe is still running, workers use conservative defaults
    // until the ArcSwap is updated (typically within ~120 s).
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

    let disable_probe_cache = std::env::var("BGE_M3_DISABLE_PROBE_CACHE")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "yes"));

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
        disable_probe_cache,
        "Starting bge-m3-embedding-server"
    );

    // Allocate one shared cost-model handle.  Conservative defaults are used
    // until the background probe (or cache hit) updates the handle via ArcSwap.
    // All workers share the same Arc<ArcSwap<CostModel>> so a single store()
    // call in the probe task is immediately visible to every worker.
    let initial_cost_model = cfg
        .cost_model_override
        .unwrap_or_else(|| CostModel::conservative(CostModel::DEFAULT_MAX_WORKSPACE));
    let cost_model_handle = Arc::new(ArcSwap::from_pointee(initial_cost_model));

    // Request concurrency limiter.  Start with cfg_workers - 1 permits so the
    // background probe always has a worker slot free.  The probe (or any terminal
    // probe bypass) calls add_permits(1) to raise to cfg_workers once the probe
    // lifecycle ends.  Minimum is 1 so a single-worker deployment always accepts
    // at least one concurrent request (at the cost of a shared probe slot).
    let initial_permits = cfg.workers.saturating_sub(1).max(1);
    let request_permits = Arc::new(Semaphore::new(initial_permits));

    let (pool, init_handle) = EmbedPool::spawn(
        cfg.workers,
        PathBuf::from(&cfg.cache_dir),
        WorkerConfig {
            cost_model: Arc::clone(&cost_model_handle),
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
        cost_model: cost_model_handle,
        probe_status: AtomicU8::new(ProbeStatus::Running as u8),
        request_permits,
    });

    let app = build_router(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
    info!(bind = %cfg.bind_addr, "Listening");

    let state_for_readiness = Arc::clone(&state);
    let cfg_max_seq = cfg.max_seq_length;
    let cfg_workers = cfg.workers;
    let cfg_safety = cfg.memory_safety_factor;
    let cost_model_override = cfg.cost_model_override;
    let cache_dir = PathBuf::from(&cfg.cache_dir);
    let model_variant_str = cfg.model_variant.to_string();

    tokio::spawn(async move {
        if let Err(e) = run_readiness_probe(
            init_handle,
            state_for_readiness,
            cfg_max_seq,
            cfg_workers,
            cfg_safety,
            cost_model_override,
            cache_dir,
            model_variant_str,
            disable_probe_cache,
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
    use arc_swap::ArcSwap;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use http_body_util::BodyExt;
    use std::sync::atomic::{AtomicBool, AtomicU8};
    use tower::ServiceExt;

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
    async fn readiness_probe_fails_when_dense_probe_fails() {
        let state = make_test_state(false, 256);
        let handle = tokio::spawn(async { Ok::<(), anyhow::Error>(()) });
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
}
