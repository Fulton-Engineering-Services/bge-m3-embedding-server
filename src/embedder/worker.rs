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

//! Blocking worker thread, request dispatch, and probe wiring.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arc_swap::ArcSwap;
use ort::value::TensorRef;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, info_span};

use super::dense::embed_dense;
use super::dual::embed_both;
use super::error::ort_err;
use super::session::{load_models, GpuSessionConfig};
use super::sm_detect::detect_sm_for_device;
use super::sparse::embed_sparse;
use super::tokenize::{build_chunk_arrays, tokenize_no_pad};
use super::trt_cache;
#[cfg(feature = "cache-gc")]
use super::trt_cache_gc;
use super::trt_warmup::{
    prewarm_persistence_postcondition_failed, prewarm_persistence_suspicious_undercount,
    trt_prewarm,
};
use super::types::{EmbedRequest, JitSuspectSender, ProbeResult};
use crate::binpack::CostModel;
use crate::config::{EpSelection, ModelVariant};
use crate::sysinfo;

/// Mirrors `trt_warmup::CACHE_HIT_THRESHOLD_MS`.  Used to classify a
/// per-request inference as a probable TRT engine cache miss so the
/// adaptive warmup task can proactively compile the engine.
const CHUNK_CACHE_HIT_THRESHOLD_MS: u64 = 5_000;

/// Returns `true` when an ORT error string indicates a TRT JIT workspace
/// overflow — a condition that may resolve with a smaller batch or halved
/// workspace budget.
///
/// Patterns verified against ORT 2.0.0-rc.12 `TensorRT` EP
/// (`ort/src/ep/tensorrt.rs`). Re-verify on every ORT version bump.
///
/// # Patterns matched
///
/// 1. **`user allocator error`** — direct CUDA allocation failure surfaced
///    by ORT's user-allocator shim during TRT kernel autotuning.
/// 2. **`could not find any implementation` + (`workspace` | `alloc`)** —
///    TRT kernel-autotuner declared no tactic fits, *and* the qualifier
///    confirms the cause is allocation-driven (otherwise this string also
///    matches genuine unsupported-op cases where retry is pointless).
/// 3. **`failed to create engine` + (`workspace` | `alloc` | `memory` |
///    `oom` | `tactic`)** — TRT EP build-time failure observed in
///    production on 2026-05-16 (request id
///    `fcc45087-2fcc-4539-9f76-f40dc0c0ec4a`, route
///    `/v1/embeddings:both`, `batch_size` 256, `prompt_tokens` 12904).
///    The qualifier is mandatory: without it, this same family also
///    covers unsupported-op and corrupted-cache cases where retrying
///    with a halved workspace is pointless and doubles caller-visible
///    latency. `alloc` subsumes `cuMemAlloc`; `memory` subsumes
///    `out of memory`.
///
/// # Known gap
///
/// The verbatim 2026-05-16 production error string did NOT include any
/// qualifier — the TRT logger appears to emit workspace/alloc detail to a
/// separate tracing target rather than propagating it into the outer
/// `Status Message`. We chose **Option A** here (require a qualifier) over
/// **Option B** (retry every `failed to create engine` unconditionally) so
/// we don't regress `does_not_match_unsupported`-style cases. If a follow-up
/// `CloudWatch` investigation confirms the TRT root-cause is reliably
/// surfaced only in a sibling `target=ort` event and never in the embed
/// error string, we may need to relax this to Option B (or pipe the TRT
/// logger output into the embed error chain). Until then, this function
/// will continue to return `false` for the verbatim production message and
/// callers will see HTTP 500 on first build failure.
///
/// Tracking: <https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/issues/78>
fn is_trt_jit_oom(e: &anyhow::Error) -> bool {
    let s = format!("{e}");
    let lowercase = s.to_lowercase();
    // "User allocator error" = direct CUDA allocation failure during TRT kernel autotuning.
    // "Could not find any implementation" qualifies only when the underlying cause is an
    // allocation failure (workspace or alloc in the message); without this qualifier it also
    // matches genuine unsupported-layer errors where retry is pointless and doubles latency.
    // "Failed to create engine" qualifies only when paired with a workspace/alloc/memory/oom/
    // tactic keyword for the same reason — see the doc-comment above.
    lowercase.contains("user allocator error")
        || (lowercase.contains("could not find any implementation")
            && (lowercase.contains("workspace") || lowercase.contains("alloc")))
        || (lowercase.contains("failed to create engine")
            && (lowercase.contains("workspace")
                || lowercase.contains("alloc")
                || lowercase.contains("memory")
                || lowercase.contains("oom")
                || lowercase.contains("tactic")))
}

/// Wraps an embed call with the standard TRT JIT-OOM retry-once-with-halved-budget
/// pattern.
///
/// If `embed_fn` fails and [`is_trt_jit_oom`] matches the error, retries once
/// with `max_workspace_bytes / 2`. Logs `trt_jit_retry` on the first attempt and
/// `trt_jit_retry_exhausted` when the retry also fails. Returns the final result.
fn embed_with_trt_retry<T, F>(
    mut embed_fn: F,
    base_cm: &CostModel,
    worker_id: usize,
    route: &'static str,
) -> anyhow::Result<T>
where
    F: FnMut(&CostModel) -> anyhow::Result<T>,
{
    match embed_fn(base_cm) {
        Ok(v) => Ok(v),
        Err(e) if is_trt_jit_oom(&e) => {
            let halved = CostModel {
                // Floor at 1 MiB to prevent integer-division from reaching 0
                // when max_workspace_bytes is very small (e.g. in tests).
                max_workspace_bytes: (base_cm.max_workspace_bytes / 2).max(1024 * 1024),
                ..*base_cm
            };
            tracing::warn!(
                worker_id,
                route,
                original_workspace_mb = base_cm.max_workspace_bytes / (1024 * 1024),
                halved_workspace_mb = halved.max_workspace_bytes / (1024 * 1024),
                error = %e,
                "trt_jit_retry"
            );
            embed_fn(&halved).map_err(|e2| {
                tracing::error!(
                    worker_id,
                    route,
                    error = %e2,
                    "trt_jit_retry_exhausted"
                );
                e2
            })
        }
        Err(e) => Err(e),
    }
}

/// Emits the `chunk_run` INFO event and, on a cache miss, notifies both the
/// JIT-suspect channel (adaptive warmup scheduling) and the engine propagation
/// broadcast channel (peer worker fast disk-load).
///
/// Returns `Some((batch_len, max_chunk_seq))` when a shape was broadcast on
/// the engine propagation channel.  The call site MUST insert this shape into
/// `warmed_local` so the originating worker self-skips its own broadcast on
/// the next `drain_engine_propagation` iteration (COR-1).
///
/// # 5000 ms threshold heuristic (COR-10)
///
/// `CHUNK_CACHE_HIT_THRESHOLD_MS` (5 s) is a **heuristic** proxy for "TRT
/// engine JIT compile occurred", not a semantic guarantee.  False negatives
/// are possible for fast-JIT small shapes; false positives are impossible
/// because a cache-hit path never exceeds ~100 ms.  The trade-off is
/// acceptable: the worst outcome of a false negative is that the adaptive
/// warmup task eventually resubmits the shape on the next real cache miss.
fn log_inference_complete(
    stats: &super::types::EmbedStats,
    worker_id: usize,
    _route: &'static str,
    jit_suspect_tx: Option<&JitSuspectSender>,
    engine_propagation_tx: Option<&tokio::sync::broadcast::Sender<(usize, usize)>>,
    batch_len: usize,
) -> Option<(usize, usize)> {
    let cache_hit = stats.inference_ms < CHUNK_CACHE_HIT_THRESHOLD_MS;
    tracing::info!(
        target: "bge_m3_embedding_server::trt_shape",
        worker_id,
        chunk_batch = batch_len,
        chunk_max_seq = stats.max_chunk_seq,
        inference_ms = stats.inference_ms,
        cache_hit,
        "chunk_run"
    );
    if !cache_hit {
        if let Some(tx) = jit_suspect_tx {
            let _ = tx.try_send((batch_len, stats.max_chunk_seq));
        }
        if let Some(tx) = engine_propagation_tx {
            let _ = tx.send((batch_len, stats.max_chunk_seq));
            return Some((batch_len, stats.max_chunk_seq));
        }
    }
    None
}

/// Drains pending broadcast notifications and runs `trt_prewarm` for each
/// new shape.
///
/// Called at the start of each worker loop iteration (between requests) so
/// peers eagerly warm their in-memory TRT profile before the next real
/// request for a new shape arrives.
///
/// `warmed_local` tracks shapes already warmed by this worker in the current
/// session.  The originating worker self-skips on subsequent drains because
/// `log_inference_complete` inserts the broadcast shape into `warmed_local`
/// at the call site before returning control to the request loop.
pub(super) fn drain_engine_propagation<F>(
    rx: &mut tokio::sync::broadcast::Receiver<(usize, usize)>,
    warmed_local: &mut std::collections::HashSet<(usize, usize)>,
    worker_id: usize,
    mut prewarm: F,
) where
    F: FnMut((usize, usize)),
{
    loop {
        match rx.try_recv() {
            Ok(shape) => {
                if warmed_local.insert(shape) {
                    prewarm(shape);
                }
            }
            Err(
                tokio::sync::broadcast::error::TryRecvError::Empty
                | tokio::sync::broadcast::error::TryRecvError::Closed,
            ) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(
                    worker_id,
                    lagged = n,
                    "engine_propagation: broadcast lagged; some shapes missed"
                );
                // Continue draining; missed shapes will be re-broadcast on
                // the next slow-inference event for that shape.
            }
        }
    }
}

pub(super) struct WorkerGuard(pub Arc<AtomicUsize>);

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        let prev = self.0.fetch_sub(1, Ordering::AcqRel);
        let live_after_drop = prev.saturating_sub(1);
        if live_after_drop == 0 {
            tracing::error!("All embedding workers have exited — pool is degraded");
        } else {
            tracing::warn!(live_after_drop, "Embedding worker exited");
        }
    }
}

/// Execution-policy configuration shared by all workers.
///
/// `cost_model` is an `Arc<ArcSwap<CostModel>>` so all workers share a single
/// handle and the background probe can update the cost model atomically after
/// fitting.  Each worker loads the current value lock-free at the start of
/// every `session.run()` call via `config.cost_model.load()`.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Quadratic-aware workspace cost model and per-worker budget.
    ///
    /// Shared across all workers via `ArcSwap`.  The background probe updates
    /// this handle once fitted coefficients are available; workers observe the
    /// new model on their next request without any coordination or restart.
    pub cost_model: Arc<ArcSwap<CostModel>>,
    /// Duration of inactivity before workers unload their model instances.
    pub idle_timeout: Option<Duration>,
    /// ONNX model variant to load (FP32, FP16, or INT8).
    pub model_variant: ModelVariant,
    /// Maximum tokenized sequence length.
    pub max_seq_length: usize,
    /// Number of intra-op threads each ORT session may use for a single
    /// `session.run()` call. Plumbed through to `load_session` at model load
    /// time. See [`crate::config::Config::intra_threads`] for sizing guidance.
    pub intra_threads: usize,
    /// ONNX Runtime execution provider selection.
    ///
    /// Forwarded to [`load_models`] at model load time so each ORT session
    /// registers the correct EP. On macOS, `CoreML` is always used regardless
    /// of this value. See [`crate::config::EpSelection`] for details.
    pub ep: EpSelection,

    /// `(batch_size, seq_len)` shapes to pre-compile as `TensorRT` engine files
    /// during worker startup.
    ///
    /// Only applied when `ep == EpSelection::TensorRt`. Sourced from
    /// `BGE_M3_TRT_WARMUP_SHAPES` via [`crate::config::Config::trt_warmup_shapes`].
    /// With multiple workers, `EmbedPool::spawn` shards the full shape list
    /// across workers (stride partition) so each GPU compiles a disjoint subset
    /// in parallel. An empty list skips pre-warming entirely.
    pub trt_warmup_shapes: Vec<(usize, usize)>,

    /// CUDA/TRT device ID for this specific worker.
    ///
    /// Set by `EmbedPool::spawn` as `worker_index % gpu_count`. Forwarded to
    /// [`super::session::execution_providers`] so the ORT session binds to the
    /// correct GPU. Ignored on CPU EP and macOS (`CoreML` is single-device).
    pub device_id: u32,

    /// Total number of GPU devices on this instance.
    ///
    /// Propagated from [`crate::config::Config::gpu_count`]. Used by
    /// `EmbedPool::spawn` to compute per-worker `device_id` values and to
    /// clamp `BGE_M3_WORKERS` for GPU execution providers.
    pub gpu_count: usize,

    /// Optional TRT workspace size cap (bytes) forwarded to ORT's TRT EP via
    /// `with_max_workspace_size`.  `None` uses ORT's built-in default.
    /// Sourced from `BGE_M3_TRT_MAX_WORKSPACE_BYTES`.
    pub trt_max_workspace_bytes: Option<usize>,

    /// Optional CUDA device memory limit (bytes) forwarded to the CUDA EP.
    /// `None` uses ORT's built-in default.
    pub gpu_mem_limit_bytes: Option<usize>,

    /// Sender half of the JIT-suspect channel created before pool spawn.
    ///
    /// After each successful inference, if `inference_ms >= CHUNK_CACHE_HIT_THRESHOLD_MS`
    /// the worker calls `try_send((batch, seq))` so the adaptive warmup task
    /// can schedule background engine compilation.  Non-blocking: if the
    /// channel is full the message is silently dropped.  `None` when adaptive
    /// warmup is disabled.
    pub jit_suspect_tx: Option<JitSuspectSender>,

    /// Sender half of the engine propagation broadcast channel.
    ///
    /// When `Some`, after any worker writes a new TRT engine plan to EFS, the
    /// worker broadcasts the `(batch, seq)` shape to all subscribed peers so
    /// they eagerly run `trt_prewarm` (~1-3s fast disk-load) instead of paying
    /// full JIT cost on the next real request.  Each worker derives its own
    /// `Receiver` via `tx.subscribe()` at startup.  `None` when
    /// `BGE_M3_ENGINE_PROPAGATION_ENABLED=0` or when the EP is not TRT.
    pub engine_propagation_tx: Option<tokio::sync::broadcast::Sender<(usize, usize)>>,

    /// When `true`, prewarm postcondition failures cause the worker to refuse
    /// to signal ready: `run_worker` returns `Err(_)` before `ready_tx.send`,
    /// the pool's init task propagates the error, and the readiness probe in
    /// `bootstrap::readiness` triggers a hard process exit. Converts the
    /// 2026-05 false-positive-readiness failure mode (every worker hits TRT
    /// `Error Code 10` mid-build, postcondition logs WARN, `/health` still
    /// returns `200 ok`, real requests then 500) into an explicit startup
    /// failure that ECS retries instead of routing traffic to.
    ///
    /// Sourced from `BGE_M3_PREWARM_STRICT`; defaults to `true`. Set to
    /// `false` to preserve pre-fix behaviour (WARN only).
    pub prewarm_strict: bool,

    /// **Destructive** stale-SM TRT engine cache GC flag.
    ///
    /// Only present when the `cache-gc` Cargo feature is compiled in.
    /// Defaults to `false` even with the feature on; flipping the
    /// runtime knob also requires `BGE_M3_TRT_CACHE_GC_ENABLED=1`. See
    /// [`crate::config::Config::trt_cache_gc_enabled`] for the multi-SM
    /// ASG hazard model.
    #[cfg(feature = "cache-gc")]
    pub trt_cache_gc_enabled: bool,
}

/// Runs a single `session.run()` for the probe, measuring RSS before and after.
///
/// The probe texts are already tokenized and padded to `pad_to` externally.
/// This function just runs inference and returns RSS deltas so `probe.rs` can
/// fit the cost model.
pub(crate) fn probe_run_dense(
    session: &mut ort::session::Session,
    ids_array: &ndarray::Array2<i64>,
    mask_array: &ndarray::Array2<i64>,
) -> Result<ProbeResult> {
    let rss_before = sysinfo::read_process_rss_bytes().unwrap_or(0);

    let ids_tensor = TensorRef::from_array_view(ids_array.view()).map_err(ort_err)?;
    let mask_tensor = TensorRef::from_array_view(mask_array.view()).map_err(ort_err)?;

    // Run inference (output discarded — we only care about RSS).
    let _outputs = session
        .run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })
        .map_err(ort_err)?;

    let rss_after = sysinfo::read_process_rss_bytes().unwrap_or(rss_before);

    Ok(ProbeResult {
        rss_before,
        rss_after,
    })
}

/// Emits a `WARN` if the oneshot reply receiver has been dropped while the
/// worker was busy with `embed_*` — meaning the client (often the router's
/// hedged race) disconnected after dispatch and the inference work is now
/// discarded. We can't interrupt ORT `session.run()` mid-call, so this is
/// observability only: operators can correlate `inference_ms` and `chunks`
/// across requests to size the router's cancellation budget.
///
/// The reply is sent unconditionally by the caller after this returns; the
/// channel layer will silently drop the value if the receiver is gone.
fn log_if_abandoned_mid_flight<T>(
    reply: &tokio::sync::oneshot::Sender<Result<(T, super::types::EmbedStats)>>,
    route: &'static str,
    worker_id: usize,
    result: &Result<(T, super::types::EmbedStats)>,
    inference_ms: u128,
) {
    if !reply.is_closed() {
        return;
    }
    let (chunks, max_chunk_seq, total_token_positions) = match result {
        Ok((_, stats)) => (
            Some(stats.chunks),
            Some(stats.max_chunk_seq),
            Some(stats.total_token_positions),
        ),
        Err(_) => (None, None, None),
    };
    let inference_ms_u64 = u64::try_from(inference_ms).unwrap_or(u64::MAX);
    tracing::warn!(
        worker_id,
        route,
        inference_ms_so_far = inference_ms_u64,
        chunks,
        max_chunk_seq,
        total_token_positions,
        "request abandoned by client during inference (work discarded; \
         ORT session.run() cannot be interrupted mid-call)"
    );
}

/// Runs one probe batch: tokenize texts, build padded arrays, call `session.run()`,
/// and return RSS deltas. Uses `embed_dense`'s no-pad tokenizer path.
fn run_probe_batch(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<ProbeResult> {
    let encodings = tokenize_no_pad(tokenizer, texts)?;
    let pad_to = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(1)
        .max(1);
    let indices: Vec<usize> = (0..texts.len()).collect();
    let (ids_array, mask_array) = build_chunk_arrays(&encodings, &indices, pad_to)?;
    probe_run_dense(session, &ids_array, &mask_array)
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
pub(super) fn run_worker(
    id: usize,
    cache_dir: PathBuf,
    rx: Arc<Mutex<mpsc::Receiver<EmbedRequest>>>,
    ready_tx: mpsc::Sender<Result<usize>>,
    live_workers: Arc<AtomicUsize>,
    loaded_workers: Arc<AtomicUsize>,
    config: WorkerConfig,
) -> Result<()> {
    let _guard = WorkerGuard(Arc::clone(&live_workers));
    let span = info_span!("worker", id = id);
    let _span_guard = span.enter();

    tracing::debug!(
        worker_id = id,
        gpu_device = config.device_id,
        ep = %config.ep,
        "worker assigned to GPU device"
    );
    info!("Loading models (worker {id})...");
    let load_start = std::time::Instant::now();
    let rt = Handle::current();

    // Measure RSS before loading so the delta accurately reflects this
    // worker's model-weight + arena-baseline contribution, not accumulated
    // OS noise.
    let pre_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(0);
    let mut initial_models = match load_models(
        &GpuSessionConfig {
            cache_dir: &cache_dir,
            model_variant: config.model_variant,
            max_seq_length: config.max_seq_length,
            intra_threads: config.intra_threads,
            ep: config.ep,
            device_id: config.device_id,
            trt_max_workspace_bytes: config.trt_max_workspace_bytes,
            gpu_mem_limit_bytes: config.gpu_mem_limit_bytes,
        },
        id == 0,
    ) {
        Ok(mut models) => {
            // Prime the ORT session arena with a tiny session.run() BEFORE
            // measuring post-load RSS. ORT lazily allocates ~1 GiB of arena
            // bookkeeping on the first run() call regardless of input size;
            // priming here folds that allocation into the per-worker model
            // RSS measurement so the workspace-budget math on the main thread
            // sees the realistic per-worker memory footprint, AND so the
            // probe sweep's per-shape `rss_delta` readings reflect only the
            // incremental workspace attributable to that shape.
            //
            // Without per-worker priming, the probe could dispatch shapes to
            // workers that have not yet done a session.run(), and each such
            // first-touch contributes ~1 GiB of arena init noise to its
            // delta — which buries the per-shape workspace signal in the
            // OLS fit.
            let prime_ids = ndarray::Array2::<i64>::zeros((1, 8));
            let prime_mask = ndarray::Array2::<i64>::ones((1, 8));
            match probe_run_dense(&mut models.0, &prime_ids, &prime_mask) {
                Ok(_) => {
                    tracing::debug!("Worker {id} arena primed");
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Worker {id} arena prime failed; first probe shape on this \
                         worker will include arena init delta"
                    );
                }
            }

            let post_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(pre_load_rss);
            tracing::info!(
                elapsed_ms = load_start.elapsed().as_millis(),
                rss_delta_mb = post_load_rss.saturating_sub(pre_load_rss) / (1024 * 1024),
                "Models loaded (worker {id})"
            );
            models
        }
        Err(e) => {
            let _ =
                rt.block_on(ready_tx.send(Err(anyhow::anyhow!("Worker {id} failed to load: {e}"))));
            return Err(e);
        }
    };

    // Destructive stale-SM cache GC (feature-gated `cache-gc` + runtime
    // `BGE_M3_TRT_CACHE_GC_ENABLED=1`). Both gates must be on for any
    // deletion to occur. When the feature is OFF this entire block is
    // physically absent from the binary. See `trt_cache_gc.rs` for the
    // multi-SM ASG hazard model.
    // Only the leader worker (id == 0) runs GC. Workers load sequentially
    // (CLAUDE.md invariant), so worker 0 always finishes GC before worker 1
    // starts — this guard is race-free and prevents spurious WARN cascades on
    // workers 1..N that would otherwise each attempt (and log) the same sweep.
    #[cfg(feature = "cache-gc")]
    if id == 0 && config.trt_cache_gc_enabled && config.ep != EpSelection::Cpu {
        let engine_cache_dir = trt_cache::engine_cache_path(&cache_dir);
        match detect_sm_for_device(config.device_id) {
            Some(current_sm) => {
                tracing::warn!(
                    target: "bge_m3_embedding_server::trt_cache_gc",
                    worker_id = id,
                    gpu_device = config.device_id,
                    current_sm = %current_sm,
                    cache_path = %engine_cache_dir.display(),
                    sidecar_suffixes = ?trt_cache_gc::ENGINE_SIDE_SUFFIXES,
                    "destructive cache GC about to run: will delete every \
                     `_smXX.engine` whose XX != current SM, plus aligned \
                     sidecars. DO NOT use this build against a shared EFS \
                     engine cache in a multi-SM ASG"
                );
                let stats = trt_cache_gc::gc_stale_sm_plans(&engine_cache_dir, &current_sm);
                tracing::warn!(
                    target: "bge_m3_embedding_server::trt_cache_gc",
                    worker_id = id,
                    gpu_device = config.device_id,
                    current_sm = %current_sm,
                    plans_deleted = stats.plans_deleted,
                    bytes_freed = stats.bytes_freed,
                    other_sms_observed = ?stats.other_sms_observed,
                    cache_path = %engine_cache_dir.display(),
                    "destructive cache GC ran: deleted other-SM engine plans \
                     from shared cache"
                );
            }
            None => {
                tracing::warn!(
                    target: "bge_m3_embedding_server::trt_cache_gc",
                    worker_id = id,
                    gpu_device = config.device_id,
                    "destructive cache GC requested but current GPU compute \
                     capability could not be detected (no nvidia-smi or \
                     unparseable output); skipping GC"
                );
            }
        }
    }

    // Detect this worker's GPU compute capability (e.g. `sm89`, `sm120`) once,
    // before TRT prewarm, so every subsequent cache enumeration restricts
    // itself to plans matching this GPU. Without this, a worker on `sm120`
    // would count a stale `_sm89.engine` plan toward `cache_hit` and report
    // a misleading `engine_count_before:3` on a fresh Blackwell deploy with
    // leftover L40S plans on the shared EFS cache — the exact 2026-05-16
    // production failure mode. `None` (detection failed) keeps legacy
    // unfiltered semantics so an operator missing `nvidia-smi` mid-deploy
    // does not see a hard regression.
    let detected_sm: Option<String> = if config.ep == EpSelection::TensorRt {
        let sm = detect_sm_for_device(config.device_id);
        if sm.is_none() {
            tracing::warn!(
                worker_id = id,
                gpu_device = config.device_id,
                "trt cache: nvidia-smi compute-capability detection failed; \
                 falling back to unfiltered engine cache counts. This re-introduces \
                 the heterogeneous-SM false-positive risk — install nvidia-smi on \
                 the container or check that the GPU is accessible."
            );
        }
        sm
    } else {
        None
    };

    // TensorRT engine pre-warming: compile engine files for each configured
    // shape before signaling readiness.  This runs BEFORE ready_tx.send() so
    // the worker is not marked ready until all TRT engines are cached —
    // `/health` correctly returns `503 loading` during the compile window.
    //
    // Each shape may take 30–120 s on first deploy; subsequent starts reuse
    // the cached `.engine` files from `{cache_dir}/trt-engines/` (seconds).
    if config.ep == EpSelection::TensorRt && !config.trt_warmup_shapes.is_empty() {
        let engine_cache_dir = trt_cache::engine_cache_path(&cache_dir);
        let total_engine_count = trt_cache::count_engine_files(&engine_cache_dir);
        let matching_engine_count =
            trt_cache::count_engine_files_for_sm(&engine_cache_dir, detected_sm.as_deref());
        // Greppable structured log line (target=bge_m3_embedding_server::trt_cache):
        // operators searching CloudWatch for `detected_sm` / `matching_engine_count`
        // get an immediate picture of the heterogeneous-cache situation. A line
        // showing `matching_engine_count:0, total_engine_count:3` is the visual
        // signature of the bug this commit fixes.
        tracing::info!(
            target: "bge_m3_embedding_server::trt_cache",
            worker_id = id,
            device_id = config.device_id,
            detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
            cache_path = %engine_cache_dir.display(),
            matching_engine_count,
            total_engine_count,
            "trt cache: SM-filtered engine plan enumeration"
        );
        tracing::info!(
            worker_id = id,
            gpu_device = config.device_id,
            detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
            shape_count = config.trt_warmup_shapes.len(),
            shapes = ?config.trt_warmup_shapes,
            "TensorRT pre-warm: worker compiling shard \
             (first run per shape takes 30–170 s; subsequent starts reuse cache)"
        );
        trt_cache::log_engine_basenames_before_prewarm_for_sm(
            &engine_cache_dir,
            detected_sm.as_deref(),
        );
        let stats = trt_prewarm(
            &mut initial_models.0,
            &config.trt_warmup_shapes,
            id,
            &cache_dir,
            detected_sm.as_deref(),
        );
        // Per-worker postcondition: if the shard reported one or more fresh
        // (non-cache-hit) compiles but the on-disk engine count did not
        // increase, surface an ERROR. This is the in-process counterpart to
        // the postcondition check at the end of the warmup-only path in
        // lib.rs, intended to catch the silent-persistence failure mode that
        // produced the 2026-05 codekeeper outage.
        if prewarm_persistence_postcondition_failed(stats.fresh_compiles, stats.engine_count_after)
        {
            tracing::error!(
                worker_id = id,
                gpu_device = config.device_id,
                detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                engine_count_delta = stats.engine_count_delta,
                cache_path = %engine_cache_dir.display(),
                "TensorRT pre-warm postcondition failed: \
                 compile-success events present but no .engine files on disk \
                 for this SM; TRT EP may be silently failing to persist engine \
                 plan files (or building plans for a different SM than this \
                 worker's device)"
            );
        } else if prewarm_persistence_suspicious_undercount(
            stats.fresh_compiles,
            stats.engine_count_after,
        ) {
            // Non-fatal: TRT EP can legitimately reuse a single `.engine`
            // file across many input shapes (engine plans are keyed by
            // fused-subgraph identity + precision + GPU SM, not by
            // `(batch, seq)`). A 1:2 ratio is tolerated silently; this
            // WARN fires only when delta * 2 < fresh_compiles AND the
            // postcondition above did not already trigger. Greppable
            // tag: "engine_count_delta is suspiciously low".
            tracing::warn!(
                worker_id = id,
                gpu_device = config.device_id,
                detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                engine_count_delta = stats.engine_count_delta,
                cache_path = %engine_cache_dir.display(),
                "TensorRT pre-warm: engine_count_delta is suspiciously low \
                 relative to fresh_compiles (delta * 2 < fresh_compiles); \
                 some engine plans may not have persisted to disk despite \
                 session.run() reporting Ok — investigate cache path \
                 resolution and EFS mount durability"
            );
        }
        tracing::info!(
            worker_id = id,
            detected_sm = detected_sm.as_deref().unwrap_or("unfiltered"),
            warmed = stats.warmed,
            skipped = stats.skipped,
            fully_cached = stats.fully_cached,
            fresh_compiles = stats.fresh_compiles,
            engine_count_before = stats.engine_count_before,
            engine_count_after = stats.engine_count_after,
            engine_count_delta = stats.engine_count_delta,
            total = config.trt_warmup_shapes.len(),
            total_compile_ms = stats.total_compile_ms,
            total_fsync_ms = stats.total_fsync_ms,
            "TensorRT pre-warm complete"
        );

        // Strict-mode escalation (BGE_M3_PREWARM_STRICT): when the prewarm
        // postcondition signals that compile-success events occurred but no
        // engine plan files persisted, refuse to signal ready. The pool's
        // init task converts the worker error into an init-handle failure,
        // and `bootstrap::readiness::run_readiness_probe` then triggers a
        // hard process exit. ECS retries the task, which is the correct
        // response to "every worker hit TRT autotuner OOM mid-build" rather
        // than serving HTTP 500 traffic from a known-broken pool.
        if should_fail_readiness(
            stats.fresh_compiles,
            stats.engine_count_after,
            config.prewarm_strict,
        ) {
            tracing::error!(
                target: "bge_m3_embedding_server::prewarm",
                worker_id = id,
                gpu_device = config.device_id,
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                cache_path = %trt_cache::engine_cache_path(&cache_dir).display(),
                prewarm_strict = config.prewarm_strict,
                "Prewarm postcondition failed and prewarm_strict=1: \
                 refusing to signal ready"
            );
            let err = anyhow::anyhow!(
                "Worker {id} prewarm postcondition failed \
                 (fresh_compiles={}, engine_count_after={}); \
                 prewarm_strict=1: refusing to signal ready",
                stats.fresh_compiles,
                stats.engine_count_after,
            );
            return Err(err);
        }
    }

    // Report the RSS delta so EmbedPool can derive the true per-worker
    // model footprint for workspace-budget calculations.
    let post_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(pre_load_rss);
    let rss_delta = post_load_rss.saturating_sub(pre_load_rss);
    info!(
        "Worker {id} models loaded — signaling ready (rss_delta_mb={})",
        rss_delta / (1024 * 1024)
    );
    let _ = rt.block_on(ready_tx.send(Ok(rss_delta)));

    let mut models: Option<(ort::session::Session, tokenizers::Tokenizer)> = Some(initial_models);

    // Tracks shapes already prewarmed by this worker via engine propagation so
    // we skip shapes we originated (which are already in our TRT profile after
    // the originating worker inserts into warmed_local before broadcasting).
    let mut warmed_local: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    // Derive per-worker broadcast receiver from the shared sender in config.
    // Each call to tx.subscribe() creates an independent receiver starting from
    // the current channel position, so workers only see shapes broadcast after
    // their own subscribe point (i.e. after model load is complete).
    let mut engine_propagation_rx = config
        .engine_propagation_tx
        .as_ref()
        .map(tokio::sync::broadcast::Sender::subscribe);

    info!("Worker {id} entering request loop");
    loop {
        // Drain peer engine-ready notifications and run trt_prewarm for any
        // new shapes so the in-memory TRT profile is extended before the next
        // real request for that shape arrives (~1-3s fast disk-load vs. full JIT).
        if config.ep == EpSelection::TensorRt {
            if let Some(ref mut bcast_rx) = engine_propagation_rx {
                if let Some((session, _)) = models.as_mut() {
                    let sm = detected_sm.as_deref();
                    drain_engine_propagation(bcast_rx, &mut warmed_local, id, |shape| {
                        let started = std::time::Instant::now();
                        let stats = trt_prewarm(session, &[shape], id, &cache_dir, sm);
                        let elapsed_ms =
                            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                        tracing::info!(
                            target: "bge_m3_embedding_server::trt_shape",
                            worker_id = id,
                            chunk_batch = shape.0,
                            chunk_max_seq = shape.1,
                            elapsed_ms,
                            warmed = stats.warmed,
                            fully_cached = stats.fully_cached,
                            detected_sm = sm.unwrap_or("unfiltered"),
                            "engine_propagation_complete"
                        );
                    });
                }
            }
        }

        let msg = if let Some(timeout) = config.idle_timeout.filter(|_| models.is_some()) {
            rt.block_on(async {
                tokio::time::timeout(timeout, async { rx.lock().await.recv().await }).await
            })
        } else {
            rt.block_on(async { Ok(rx.lock().await.recv().await) })
        };

        match msg {
            Err(_elapsed) => {
                models = None;
                loaded_workers.fetch_sub(1, Ordering::AcqRel);
                tracing::info!("Worker {id} unloaded models after idle timeout");
            }
            Ok(None) => {
                if models.is_some() {
                    loaded_workers.fetch_sub(1, Ordering::AcqRel);
                }
                info!("Worker {id} channel closed, shutting down");
                break;
            }
            Ok(Some(request)) => {
                if models.is_none() {
                    tracing::info!("Worker {id} reloading models after idle...");
                    let reload_start = std::time::Instant::now();
                    match load_models(
                        &GpuSessionConfig {
                            cache_dir: &cache_dir,
                            model_variant: config.model_variant,
                            max_seq_length: config.max_seq_length,
                            intra_threads: config.intra_threads,
                            ep: config.ep,
                            device_id: config.device_id,
                            trt_max_workspace_bytes: config.trt_max_workspace_bytes,
                            gpu_mem_limit_bytes: config.gpu_mem_limit_bytes,
                        },
                        false,
                    ) {
                        Ok(mut m) => {
                            // Prime the freshly-loaded session arena so the
                            // first incoming request after idle reload doesn't
                            // pay the ~1 GiB lazy-arena-init cost. Same
                            // rationale as the startup priming in the
                            // load-models Ok arm above.
                            let prime_ids = ndarray::Array2::<i64>::zeros((1, 8));
                            let prime_mask = ndarray::Array2::<i64>::ones((1, 8));
                            if let Err(e) = probe_run_dense(&mut m.0, &prime_ids, &prime_mask) {
                                tracing::warn!(
                                    error = %e,
                                    "Worker {id} post-reload arena prime failed"
                                );
                            }
                            models = Some(m);
                            loaded_workers.fetch_add(1, Ordering::AcqRel);
                            tracing::info!(
                                elapsed_ms = reload_start.elapsed().as_millis(),
                                "Worker {id} reloaded models"
                            );
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Worker {id} failed to reload models");
                            let err = anyhow::anyhow!("Model reload failed: {e}");
                            match request {
                                EmbedRequest::Dense { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                                EmbedRequest::Sparse { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                                EmbedRequest::Both { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                                EmbedRequest::Probe { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                                EmbedRequest::AdaptiveWarmup { ack, .. } => {
                                    let _ = ack.send(Err(err));
                                }
                            }
                            continue;
                        }
                    }
                }

                let (session, tokenizer) =
                    models.as_mut().expect("models loaded after reload check");

                match request {
                    EmbedRequest::Dense { texts, reply } => {
                        // Pre-dispatch abandonment check: the router's hedged
                        // race or the original HTTP client may have already
                        // disconnected while this request sat in the worker
                        // queue. ORT `session.run()` is a blocking C call that
                        // cannot be interrupted mid-MatMul (see CLAUDE.md
                        // "client disconnect" gotcha), so the only opportunity
                        // we have to save work is BEFORE inference starts. The
                        // post-inference check below is observability only.
                        if reply.is_closed() {
                            tracing::warn!(
                                worker_id = id,
                                route = "dense",
                                batch_size = texts.len(),
                                "request abandoned by client before dispatch — skipping inference"
                            );
                            continue;
                        }
                        let t_inference = std::time::Instant::now();
                        let cm_guard = config.cost_model.load();
                        let result = embed_with_trt_retry(
                            |cm| embed_dense(session, tokenizer, &texts, cm, config.model_variant),
                            &cm_guard,
                            id,
                            "dense",
                        )
                        .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                        let inference_ms = t_inference.elapsed().as_millis();
                        if let Ok((_, ref stats)) = result {
                            if let Some(shape) = log_inference_complete(
                                stats,
                                id,
                                "dense",
                                config.jit_suspect_tx.as_ref(),
                                config.engine_propagation_tx.as_ref(),
                                texts.len(),
                            ) {
                                warmed_local.insert(shape);
                            }
                            tracing::info!(
                                worker_id = id,
                                chunks = stats.chunks,
                                max_chunk_seq = stats.max_chunk_seq,
                                total_token_positions = stats.total_token_positions,
                                seq_len_min = stats.seq_len_min,
                                seq_len_max = stats.seq_len_max,
                                seq_len_mean = stats.seq_len_mean,
                                seq_len_p95 = stats.seq_len_p95,
                                tokenize_ms = stats.tokenize_ms,
                                inference_ms = stats.inference_ms,
                                "worker: dense embed complete"
                            );
                        }
                        log_if_abandoned_mid_flight(&reply, "dense", id, &result, inference_ms);
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Sparse { texts, reply } => {
                        if reply.is_closed() {
                            tracing::warn!(
                                worker_id = id,
                                route = "sparse",
                                batch_size = texts.len(),
                                "request abandoned by client before dispatch — skipping inference"
                            );
                            continue;
                        }
                        let t_inference = std::time::Instant::now();
                        let cm_guard = config.cost_model.load();
                        let result = embed_with_trt_retry(
                            |cm| embed_sparse(session, tokenizer, &texts, cm, config.model_variant),
                            &cm_guard,
                            id,
                            "sparse",
                        )
                        .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                        let inference_ms = t_inference.elapsed().as_millis();
                        if let Ok((_, ref stats)) = result {
                            if let Some(shape) = log_inference_complete(
                                stats,
                                id,
                                "sparse",
                                config.jit_suspect_tx.as_ref(),
                                config.engine_propagation_tx.as_ref(),
                                texts.len(),
                            ) {
                                warmed_local.insert(shape);
                            }
                            tracing::info!(
                                worker_id = id,
                                chunks = stats.chunks,
                                max_chunk_seq = stats.max_chunk_seq,
                                total_token_positions = stats.total_token_positions,
                                seq_len_min = stats.seq_len_min,
                                seq_len_max = stats.seq_len_max,
                                seq_len_mean = stats.seq_len_mean,
                                seq_len_p95 = stats.seq_len_p95,
                                tokenize_ms = stats.tokenize_ms,
                                inference_ms = stats.inference_ms,
                                "worker: sparse embed complete"
                            );
                        }
                        log_if_abandoned_mid_flight(&reply, "sparse", id, &result, inference_ms);
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Both { texts, reply } => {
                        if reply.is_closed() {
                            tracing::warn!(
                                worker_id = id,
                                route = "both",
                                batch_size = texts.len(),
                                "request abandoned by client before dispatch — skipping inference"
                            );
                            continue;
                        }
                        let t_inference = std::time::Instant::now();
                        let cm_guard = config.cost_model.load();
                        let result = embed_with_trt_retry(
                            |cm| embed_both(session, tokenizer, &texts, cm, config.model_variant),
                            &cm_guard,
                            id,
                            "both",
                        )
                        .map_err(|e| anyhow::anyhow!("Dual embed error: {e}"));
                        let inference_ms = t_inference.elapsed().as_millis();
                        if let Ok((_, ref stats)) = result {
                            if let Some(shape) = log_inference_complete(
                                stats,
                                id,
                                "both",
                                config.jit_suspect_tx.as_ref(),
                                config.engine_propagation_tx.as_ref(),
                                texts.len(),
                            ) {
                                warmed_local.insert(shape);
                            }
                            tracing::info!(
                                worker_id = id,
                                chunks = stats.chunks,
                                max_chunk_seq = stats.max_chunk_seq,
                                total_token_positions = stats.total_token_positions,
                                seq_len_min = stats.seq_len_min,
                                seq_len_max = stats.seq_len_max,
                                seq_len_mean = stats.seq_len_mean,
                                seq_len_p95 = stats.seq_len_p95,
                                tokenize_ms = stats.tokenize_ms,
                                inference_ms = stats.inference_ms,
                                "worker: both embed complete"
                            );
                        }
                        log_if_abandoned_mid_flight(&reply, "both", id, &result, inference_ms);
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Probe { texts, reply } => {
                        // Probe: tokenize once without padding, run dense inference
                        // on a single flat batch at the chunk's natural max_seq.
                        // Probes are internal — no client-disconnect path applies.
                        let result = run_probe_batch(session, tokenizer, &texts);
                        let _ = reply.send(result);
                    }
                    EmbedRequest::AdaptiveWarmup { batch, seq, ack } => {
                        // Run trt_prewarm for a single shape so the TRT EP
                        // compiles and caches the engine during an idle window.
                        // On CPU/CUDA EP this is a cheap no-op (returns Ok(0)).
                        let result: anyhow::Result<u64> = if config.ep == EpSelection::TensorRt {
                            let shape = vec![(batch, seq)];
                            let stats = trt_prewarm(
                                session,
                                &shape,
                                id,
                                &cache_dir,
                                detected_sm.as_deref(),
                            );
                            if stats.warmed > 0 || stats.fully_cached {
                                Ok(stats.total_compile_ms)
                            } else {
                                Err(anyhow::anyhow!(
                                    "adaptive warmup: no shapes warmed for \
                                     ({batch}, {seq}) on worker {id}"
                                ))
                            }
                        } else {
                            Ok(0)
                        };
                        let _ = ack.send(result);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Decides whether a worker should refuse to signal ready after its prewarm
/// sweep based on the on-disk persistence postcondition.
///
/// Returns `true` iff `strict` is `true` AND at least one of the
/// [`prewarm_persistence_postcondition_failed`] /
/// [`prewarm_persistence_suspicious_undercount`] predicates fires for the
/// given `(fresh_compiles, engine_count_after)` snapshot.
///
/// The signature deliberately accepts primitive `usize` values rather than
/// `&PrewarmStats` so the unit tests in `worker/tests/prewarm_strict.rs`
/// stay decoupled from the `trt_warmup::PrewarmStats` struct shape; this
/// also lets the predicate be reused at future call sites (e.g. an admin
/// endpoint that wants to surface the same decision) without dragging in
/// the rest of the prewarm statistics.
///
/// # Strict-mode semantics
///
/// Strict mode (`prewarm_strict=true`) only blocks readiness when
/// `engine_count_after == 0` — i.e. **complete zero-plan failure** where
/// fresh compiles occurred but not a single `.engine` file landed on disk.
/// This is the 2026-05 failure mode where every worker hit TRT autotuner OOM
/// mid-build, leaving the cache empty and every subsequent real request
/// returning HTTP 500.
///
/// **Partial undercounts** (e.g. 1 engine persisted out of 16 compiled) do
/// NOT block readiness — workers will serve traffic using the one cached shape
/// and JIT-compile any missing shapes on first request. This is acceptable:
/// partial persistence is most commonly caused by TRT's subgraph fusing
/// (multiple `(batch, seq)` shapes sharing one engine file), not by a hard
/// persistence failure.
///
/// If threshold-based undercount blocking becomes necessary in the future,
/// the [`prewarm_persistence_suspicious_undercount`] branch already has the
/// scaffolding — promote it from WARN to a readiness gate here.
fn should_fail_readiness(fresh_compiles: usize, engine_count_after: usize, strict: bool) -> bool {
    if !strict {
        return false;
    }
    prewarm_persistence_postcondition_failed(fresh_compiles, engine_count_after)
        || prewarm_persistence_suspicious_undercount(fresh_compiles, engine_count_after)
}

#[cfg(test)]
mod tests;
