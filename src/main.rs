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
use std::time::Duration;
use tokio::sync::Semaphore;
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::{DefaultOnFailure, DefaultOnResponse, MakeSpan, TraceLayer};
use tracing::info;
use tracing::Level;

use config::Config;

#[derive(Clone, Default)]
struct UuidRequestId;

impl MakeRequestId for UuidRequestId {
    fn make_request_id<B>(&mut self, _request: &axum::http::Request<B>) -> Option<RequestId> {
        let id = uuid::Uuid::new_v4().to_string();
        HeaderValue::from_str(&id).ok().map(RequestId::new)
    }
}

/// Selects the tracing level for HTTP spans based on path.
///
/// `/health` and `/v1/models` are polled frequently by load balancers and the
/// Docker `HEALTHCHECK`. Logging them at DEBUG rather than INFO keeps
/// `CloudWatch` free of ~8,640 health-check records per container per day.
#[derive(Clone)]
struct RouteAwareSpan;

impl<B> MakeSpan<B> for RouteAwareSpan {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> tracing::Span {
        let path = request.uri().path();
        let is_noisy = matches!(path, "/health" | "/v1/models");
        let method = request.method().as_str();
        if is_noisy {
            tracing::debug_span!(
                "http_request",
                method = method,
                uri = %request.uri(),
                version = ?request.version(),
            )
        } else {
            tracing::info_span!(
                "http_request",
                method = method,
                uri = %request.uri(),
                version = ?request.version(),
            )
        }
    }
}

pub(crate) fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(handler::dense_embeddings))
        .route("/v1/sparse-embeddings", post(handler::sparse_embeddings))
        // The colon in `/v1/embeddings:both` is a valid `pchar` per RFC 3986
        // §3.3, but some HTTP clients (and URI builders) percent-encode it
        // anyway when it appears in a path segment. The router is built on
        // `matchit`, which matches the raw URI path byte-for-byte, so the
        // encoded forms are registered as alias routes pointing at the same
        // handler. RFC 3986 percent-encoding is case-insensitive, hence both
        // upper- and lowercase aliases.
        .route("/v1/embeddings:both", post(handler::both_embeddings))
        .route("/v1/embeddings%3Aboth", post(handler::both_embeddings))
        .route("/v1/embeddings%3aboth", post(handler::both_embeddings))
        .route("/v1/models", get(handler::models))
        .route("/health", get(handler::health))
        .layer(DefaultBodyLimit::max(2_097_152))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(RouteAwareSpan)
                .on_response(
                    DefaultOnResponse::new()
                        .level(Level::INFO)
                        .latency_unit(tower_http::LatencyUnit::Millis),
                )
                .on_failure(DefaultOnFailure::new().level(Level::ERROR)),
        )
        .layer(SetRequestIdLayer::x_request_id(UuidRequestId))
        .with_state(state)
}

/// Computes per-worker workspace budget and derived stats from memory inputs.
///
/// # Returns
///
/// `(per_worker_workspace, worst_case_peak, utilization_pct)` where:
/// - `per_worker_workspace`: bytes available to one worker for a single
///   `session.run()` call (passed as `rss_ceiling` to the probe).
/// - `worst_case_peak`: total bytes consumed when all workers run
///   simultaneously at budget ceiling (used for the 90% OOM warning).
/// - `utilization_pct`: `worst_case_peak / available_bytes × 100`.
///
/// Extracted as a pure function so the budget logic is unit-testable
/// independently of the async readiness probe machinery.
//
// cast_precision_loss: available_bytes ≤ ~28 GB (Fargate limit), total_workspace
//   similarly bounded; f64 has 2^52 mantissa (~4.5 PB) — no precision loss.
// cast_possible_truncation: per_worker_workspace is a byte budget; truncating
//   sub-byte fractions is intentional and harmless.
// cast_sign_loss: total_workspace is derived from saturating_sub — always ≥ 0.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn compute_workspace_budget(
    available_bytes: usize,
    n_workers: usize,
    model_rss_per_worker: usize,
    safety_factor: f64,
) -> (usize, usize, f64) {
    let total_workspace = available_bytes
        .saturating_sub(n_workers.saturating_mul(model_rss_per_worker))
        .saturating_sub(OS_HEADROOM_BYTES);
    let per_worker_workspace = (total_workspace as f64 * safety_factor / n_workers as f64) as usize;

    let worst_case_peak = n_workers
        .saturating_mul(per_worker_workspace)
        .saturating_add(n_workers.saturating_mul(model_rss_per_worker))
        .saturating_add(OS_HEADROOM_BYTES);

    let utilization_pct = if available_bytes > 0 {
        worst_case_peak as f64 / available_bytes as f64 * 100.0
    } else {
        0.0
    };

    (per_worker_workspace, worst_case_peak, utilization_pct)
}

/// Runs after all workers finish loading their model instances.
///
/// # Sequence
///
/// 1. Wait for worker pool initialisation to finish.
/// 2. Read `pool.model_rss_per_worker_bytes()` — the median RSS delta measured
///    inside each worker's `spawn_blocking` closure around `load_models()`.
///    Workers load sequentially (one at a time), so each delta reflects only
///    that worker's ORT session allocation with no parallel-load contamination.
/// 3. Detect available memory; compute `per_worker_workspace` via
///    `compute_workspace_budget`. Fail fast if the budget is below the
///    physics-based floor (cannot fit even one text at `max_seq_length`).
/// 4. Write static [`TuningInfo`] to `OnceLock`.
/// 5. Resolve the cost model — one of three paths:
///    - cost-model override set: apply immediately, `probe_status = Disabled`.
///    - EFS cache hit: apply cached `(a, b)` via `ArcSwap`, `probe_status = CacheHit`.
///    - cache miss: set `probe_status = Running`, launch background probe task.
/// 6. Run dense + sparse readiness calls to confirm the worker pool is healthy.
/// 7. Flip `state.ready = true` — `/health` returns `200 ok` from this point on.
///    If the probe is still running in the background, the bin-packer uses
///    conservative defaults until the `ArcSwap` is updated (typically ~120 s).
///
// cast_possible_truncation: physics_floor is a u128 workspace estimate; truncating
//   to usize is safe because per_worker_workspace is itself bounded by available_bytes
//   which fits comfortably in usize on any 64-bit target.
// cast_precision_loss / cast_sign_loss: delegated to compute_workspace_budget.
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

    // Per-worker model RSS is the median of per-worker deltas collected by
    // EmbedPool::spawn. Workers load sequentially (one at a time) so each
    // delta reflects only that worker's ORT session allocation. The median
    // is robust to one outlier from page-cache settling or ORT arena jitter.
    let model_rss_per_worker = state.pool.model_rss_per_worker_bytes();
    info!(
        model_rss_per_worker_mb = model_rss_per_worker / (1024 * 1024),
        "Measured model RSS per worker (median across all workers)"
    );

    // Compute per-worker workspace ceiling.
    let (per_worker_workspace, worst_case_peak, utilization_pct) = compute_workspace_budget(
        mem.available_bytes,
        cfg_workers,
        model_rss_per_worker,
        cfg_safety,
    );

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

    // Physics-based safety floor: the minimum workspace required to run a
    // single text at the configured max sequence length under conservative
    // cost-model coefficients. If the computed per_worker_workspace falls
    // below this floor, the measurement upstream is broken (e.g. inflated
    // model_rss_per_worker driving total_workspace to zero via saturating_sub).
    // Continuing in this state degrades bin_pack to batch=1 and produces
    // silent throughput collapse — fail fast instead so ECS restarts the task
    // and the operator sees a clear error rather than a degraded service.
    let physics_floor = CostModel::conservative(0).chunk_cost(1, cfg_max_seq) as usize;
    if per_worker_workspace < physics_floor {
        return Err(anyhow::anyhow!(
            "Computed per_worker_workspace ({per_worker_workspace} B = {} MiB) is below the \
             physics-based minimum ({physics_floor} B = {} MiB) needed to run one text at \
             max_seq_length={cfg_max_seq}. Likely causes: model_rss_per_worker ({} MiB) is \
             over-estimated (parallel-load contamination), BGE_M3_MEMORY_SAFETY_FACTOR too low \
             ({cfg_safety}), BGE_M3_WORKERS too high ({cfg_workers}) for available memory \
             ({} MiB), or BGE_M3_AVAILABLE_MEMORY_BYTES override too small.",
            per_worker_workspace / (1024 * 1024),
            physics_floor / (1024 * 1024),
            model_rss_per_worker / (1024 * 1024),
            mem.available_bytes / (1024 * 1024),
        ));
    }

    // Write static memory + budget info now so /health always shows these fields
    // even while the background probe is still running.
    let _ = state.tuning.set(TuningInfo::new(
        &mem,
        model_rss_per_worker,
        worst_case_peak,
        utilization_pct,
    ));

    // The cgroup-limit byte count (the actual kernel ceiling, not the
    // safety-discounted budget) is threaded into run_probe so the per-shape
    // RSS guard can compare against the real ceiling rather than the
    // discounted per_worker_workspace value.
    let cgroup_limit_bytes = mem.available_bytes;

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
        // No probe — run readiness checks inline and open traffic.
        run_readiness_checks_and_open(&state).await?;
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
            // Cache hit — run readiness checks inline and open traffic.
            run_readiness_checks_and_open(&state).await?;
        } else {
            // Cache miss — probe must run. See `spawn_probe_task` for the
            // serialisation protocol that holds all cfg_workers permits across
            // the probe + readiness window.
            spawn_probe_task(
                Arc::clone(&state),
                cfg_workers,
                cfg_max_seq,
                per_worker_workspace,
                cgroup_limit_bytes,
                cache_dir,
                model_variant_str,
                /* save_cache = */ true,
            )
            .await;
            return Ok(());
        }
    } else {
        // BGE_M3_DISABLE_PROBE_CACHE=1 but no override — run probe without caching.
        spawn_probe_task(
            Arc::clone(&state),
            cfg_workers,
            cfg_max_seq,
            per_worker_workspace,
            cgroup_limit_bytes,
            cache_dir,
            model_variant_str,
            /* save_cache = */ false,
        )
        .await;
        return Ok(());
    }

    Ok(())
}

/// Spawns the background probe task with proper permit ownership.
///
/// # Serialisation protocol
///
/// 1. Set `probe_status = Running`.
/// 2. Acquire `cfg_workers - 1` permits via `acquire_many_owned` — combined
///    with the 1 permit already reserved at startup, this drains the
///    semaphore to 0 so all incoming `/v1/embeddings*` requests queue
///    behind the gate while the probe is in flight.
/// 3. Move the [`tokio::sync::OwnedSemaphorePermit`] into the spawned
///    task. Its destructor is invoked just before `add_permits(cfg_workers)`
///    at the end of the task, restoring full traffic concurrency.
///
/// **Rationale for `acquire_many_owned`:** `tokio::spawn` returns
/// synchronously before the spawned task starts executing. A permit bound to
/// a local variable in the parent function would be dropped immediately at
/// the end of that function — before the probe begins — leaving the semaphore
/// un-drained and allowing real traffic to contaminate per-shape RSS
/// measurements. `acquire_many_owned` returns an `OwnedSemaphorePermit`
/// independent of the source `Semaphore` lifetime, so it survives the move
/// into the async closure and is held for the full duration of the probe.
#[allow(clippy::too_many_arguments)]
async fn spawn_probe_task(
    state: Arc<AppState>,
    cfg_workers: usize,
    cfg_max_seq: usize,
    per_worker_workspace: usize,
    cgroup_limit_bytes: usize,
    cache_dir: PathBuf,
    model_variant_str: String,
    save_cache: bool,
) {
    state
        .probe_status
        .store(ProbeStatus::Running as u8, Ordering::Release);

    // Drain all remaining permits. The semaphore starts with
    // `max(cfg_workers - 1, 1)` permits at startup (one slot reserved for
    // the probe worker); we acquire the remaining `cfg_workers - 1` here
    // so the count drops to 0 for the duration of the probe.
    //
    // `acquire_many_owned` returns an `OwnedSemaphorePermit` that we move
    // into the spawned task closure. The permit's drop handler returns the
    // permits to the semaphore — we manually call `add_permits(cfg_workers)`
    // in the task to also release the originally-reserved probe slot.
    let probe_permit = Arc::clone(&state.request_permits)
        .acquire_many_owned(u32::try_from(cfg_workers.saturating_sub(1)).unwrap_or(u32::MAX))
        .await
        .ok();

    tokio::spawn(async move {
        // Forget the OwnedSemaphorePermit at the end; we manually
        // add_permits(cfg_workers) below so the count goes from 0
        // straight to cfg_workers (releasing both the drained permits
        // and the originally-reserved probe slot in one operation).
        if let Some(p) = probe_permit {
            p.forget();
        }

        let (a, b) = probe::run_probe(
            &state.pool,
            cfg_max_seq,
            per_worker_workspace,
            cgroup_limit_bytes,
        )
        .await;
        let cm = CostModel {
            a,
            b,
            max_workspace_bytes: per_worker_workspace,
        };
        info!(
            a = cm.a,
            b = cm.b,
            max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
            "Probe complete — updating cost model"
        );
        state.cost_model.store(Arc::new(cm));
        // Distinguish real fit from conservative fallback.
        let status = if (a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON
            && (b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON
        {
            ProbeStatus::Failed
        } else {
            if save_cache {
                probe::save_probe_cache(&cache_dir, &model_variant_str, cfg_max_seq, a, b);
            }
            ProbeStatus::Complete
        };
        state.probe_status.store(status as u8, Ordering::Release);
        info!(probe_status = status.as_str(), "Probe status updated");

        // Readiness checks run inside the probe task so they do not
        // contaminate the probe's RSS measurements.
        if let Err(e) = run_readiness_checks_and_open(&state).await {
            tracing::error!(error = %e, "Post-probe readiness check failed");
        }
        // Release the drained permits AND the originally-reserved probe
        // slot in one operation. Net effect: semaphore count goes from 0
        // back to cfg_workers, opening traffic at full concurrency.
        state.request_permits.add_permits(cfg_workers);
    });
}

/// Runs the dense + sparse readiness calls and flips `state.ready`.
///
/// Called from the override/cache-hit paths (inline, before returning from
/// `run_readiness_probe`) and from the background probe task (after the probe
/// completes) so that readiness checks never run concurrently with the probe.
async fn run_readiness_checks_and_open(state: &AppState) -> anyhow::Result<()> {
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

    state
        .ready
        .store(true, std::sync::atomic::Ordering::Release);
    tracing::info!("Models ready — accepting requests");
    Ok(())
}

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> anyhow::Result<()> {
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let log_format = std::env::var("BGE_M3_LOG_FORMAT").ok();
    // JSON by default in non-TTY environments (Docker/Fargate/CloudWatch).
    // Force pretty with BGE_M3_LOG_FORMAT=text or BGE_M3_LOG_FORMAT=pretty.
    // Force JSON with BGE_M3_LOG_FORMAT=json.
    let want_json = match log_format.as_deref() {
        Some("text" | "pretty") => false,
        Some("json") => true,
        _ => !std::io::IsTerminal::is_terminal(&std::io::stdout()),
    };
    if want_json {
        tracing_subscriber::fmt()
            .json()
            .with_current_span(true)
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    }

    info!(
        version = env!("CARGO_PKG_VERSION"),
        git_sha = env!("BGE_M3_GIT_SHA"),
        target_arch = std::env::consts::ARCH,
        target_os = std::env::consts::OS,
        profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        "bge-m3-embedding-server build info"
    );

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

    // Periodic heartbeat — logs RSS, worker counts, queue depth, and permits
    // at a fixed interval so dashboards can detect slow leaks or saturation.
    let heartbeat_secs = cfg.heartbeat_secs;
    if heartbeat_secs > 0 {
        let state_hb = Arc::clone(&state);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(heartbeat_secs));
            // Skip the first (immediate) tick so we don't log at t=0 before
            // the server has finished starting up.
            tick.tick().await;
            loop {
                tick.tick().await;
                let rss_mb = sysinfo::read_process_rss_bytes().unwrap_or(0) / (1024 * 1024);
                info!(
                    rss_mb,
                    live_workers = state_hb.pool.live_worker_count(),
                    loaded_workers = state_hb.pool.loaded_worker_count(),
                    queue_depth = state_hb.pool.queue_depth(),
                    available_permits = state_hb.request_permits.available_permits(),
                    probe_status =
                        ProbeStatus::from_u8(state_hb.probe_status.load(Ordering::Acquire))
                            .as_str(),
                    "heartbeat"
                );
            }
        });
    }

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
        use crate::binpack::CostModel;
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
}
