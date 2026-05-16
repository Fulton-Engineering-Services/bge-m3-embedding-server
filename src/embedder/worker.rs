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

use super::adaptive_warmup::JitSuspectSender;
use super::dense::embed_dense;
use super::dual::embed_both;
use super::error::ort_err;
use super::session::load_models;
use super::sparse::embed_sparse;
use super::tokenize::{build_chunk_arrays, tokenize_no_pad};
use super::trt_cache;
use super::trt_warmup::{
    prewarm_persistence_postcondition_failed, prewarm_persistence_suspicious_undercount,
    trt_prewarm,
};
use super::types::{EmbedRequest, ProbeResult};
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
/// Patterns verified against ORT 2.0.0-rc.12 TensorRT EP
/// (`ort/src/ep/tensorrt.rs`). Re-verify on every ORT version bump.
fn is_trt_jit_oom(e: &anyhow::Error) -> bool {
    let s = format!("{e}");
    let lowercase = s.to_lowercase();
    // "User allocator error" = direct CUDA allocation failure during TRT kernel autotuning.
    // "Could not find any implementation" qualifies only when the underlying cause is an
    // allocation failure (workspace or alloc in the message); without this qualifier it also
    // matches genuine unsupported-layer errors where retry is pointless and doubles latency.
    lowercase.contains("user allocator error")
        || (lowercase.contains("could not find any implementation")
            && (lowercase.contains("workspace") || lowercase.contains("alloc")))
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
                max_workspace_bytes: base_cm.max_workspace_bytes / 2,
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

/// Emits the `chunk_run` INFO event and, on a cache miss, notifies the
/// JIT-suspect channel so the adaptive warmup task can schedule background
/// engine compilation.
fn log_inference_complete(
    stats: &super::types::EmbedStats,
    worker_id: usize,
    route: &'static str,
    jit_suspect_tx: Option<&JitSuspectSender>,
    batch_len: usize,
) {
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
    /// Sourced from `BGE_M3_GPU_VRAM_BUDGET_BYTES`.
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
        &cache_dir,
        id == 0,
        config.model_variant,
        config.max_seq_length,
        config.intra_threads,
        config.ep,
        config.device_id,
        config.trt_max_workspace_bytes,
        config.gpu_mem_limit_bytes,
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

    // TensorRT engine pre-warming: compile engine files for each configured
    // shape before signaling readiness.  This runs BEFORE ready_tx.send() so
    // the worker is not marked ready until all TRT engines are cached —
    // `/health` correctly returns `503 loading` during the compile window.
    //
    // Each shape may take 30–120 s on first deploy; subsequent starts reuse
    // the cached `.engine` files from `{cache_dir}/trt-engines/` (seconds).
    if config.ep == EpSelection::TensorRt && !config.trt_warmup_shapes.is_empty() {
        tracing::info!(
            worker_id = id,
            gpu_device = config.device_id,
            shape_count = config.trt_warmup_shapes.len(),
            shapes = ?config.trt_warmup_shapes,
            "TensorRT pre-warm: worker compiling shard \
             (first run per shape takes 30–170 s; subsequent starts reuse cache)"
        );
        trt_cache::log_engine_basenames_before_prewarm(&trt_cache::engine_cache_path(&cache_dir));
        let stats = trt_prewarm(
            &mut initial_models.0,
            &config.trt_warmup_shapes,
            id,
            &cache_dir,
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
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                engine_count_delta = stats.engine_count_delta,
                cache_path = %trt_cache::engine_cache_path(&cache_dir).display(),
                "TensorRT pre-warm postcondition failed: \
                 compile-success events present but no .engine files on disk; \
                 TRT EP may be silently failing to persist engine plan files"
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
                fresh_compiles = stats.fresh_compiles,
                engine_count_before = stats.engine_count_before,
                engine_count_after = stats.engine_count_after,
                engine_count_delta = stats.engine_count_delta,
                cache_path = %trt_cache::engine_cache_path(&cache_dir).display(),
                "TensorRT pre-warm: engine_count_delta is suspiciously low \
                 relative to fresh_compiles (delta * 2 < fresh_compiles); \
                 some engine plans may not have persisted to disk despite \
                 session.run() reporting Ok — investigate cache path \
                 resolution and EFS mount durability"
            );
        }
        tracing::info!(
            worker_id = id,
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

    info!("Worker {id} entering request loop");
    loop {
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
                        &cache_dir,
                        false,
                        config.model_variant,
                        config.max_seq_length,
                        config.intra_threads,
                        config.ep,
                        config.device_id,
                        config.trt_max_workspace_bytes,
                        config.gpu_mem_limit_bytes,
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
                            log_inference_complete(
                                stats,
                                id,
                                "dense",
                                config.jit_suspect_tx.as_ref(),
                                texts.len(),
                            );
                            tracing::info!(
                                worker_id = id,
                                chunks = stats.chunks,
                                max_chunk_seq = stats.max_chunk_seq,
                                total_token_positions = stats.total_token_positions,
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
                            log_inference_complete(
                                stats,
                                id,
                                "sparse",
                                config.jit_suspect_tx.as_ref(),
                                texts.len(),
                            );
                            tracing::info!(
                                worker_id = id,
                                chunks = stats.chunks,
                                max_chunk_seq = stats.max_chunk_seq,
                                total_token_positions = stats.total_token_positions,
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
                            log_inference_complete(
                                stats,
                                id,
                                "both",
                                config.jit_suspect_tx.as_ref(),
                                texts.len(),
                            );
                            tracing::info!(
                                worker_id = id,
                                chunks = stats.chunks,
                                max_chunk_seq = stats.max_chunk_seq,
                                total_token_positions = stats.total_token_positions,
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
                            let stats = trt_prewarm(session, &shape, id, &cache_dir);
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- is_trt_jit_oom ---

    #[test]
    fn is_trt_jit_oom_matches_user_allocator_error() {
        let e = anyhow::anyhow!("TRT: User allocator error during cuDNN workspace setup");
        assert!(is_trt_jit_oom(&e));
    }

    #[test]
    fn is_trt_jit_oom_matches_no_impl_with_workspace() {
        let e = anyhow::anyhow!("Could not find any implementation for node; workspace too small");
        assert!(is_trt_jit_oom(&e));
    }

    #[test]
    fn is_trt_jit_oom_does_not_match_no_impl_without_alloc() {
        // "Could not find any implementation" without workspace/alloc = unsupported op, not OOM
        let e = anyhow::anyhow!("Could not find any implementation for node /Reshape_3");
        assert!(!is_trt_jit_oom(&e));
    }

    #[test]
    fn is_trt_jit_oom_does_not_match_unrelated_error() {
        let e = anyhow::anyhow!("OrtStatus: NOT_IMPLEMENTED: opset 18 not supported");
        assert!(!is_trt_jit_oom(&e));
    }

    #[test]
    fn is_trt_jit_oom_does_not_match_empty() {
        let e = anyhow::anyhow!("");
        assert!(!is_trt_jit_oom(&e));
    }

    // --- embed_with_trt_retry ---

    #[test]
    fn embed_with_trt_retry_succeeds_without_retry() {
        let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
        let calls = std::cell::Cell::new(0u32);
        let result = embed_with_trt_retry(
            |_cm| {
                calls.set(calls.get() + 1);
                Ok(42u64)
            },
            &cm,
            0,
            "test",
        );
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn embed_with_trt_retry_fires_exactly_once_on_oom() {
        let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
        let calls = std::cell::Cell::new(0u32);
        let result: anyhow::Result<u64> = embed_with_trt_retry(
            |_cm| {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    Err(anyhow::anyhow!("User allocator error in TRT"))
                } else {
                    Ok(99)
                }
            },
            &cm,
            0,
            "test",
        );
        assert_eq!(result.unwrap(), 99);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn embed_with_trt_retry_halves_workspace_on_retry() {
        let original = 2 * 1024 * 1024 * 1024usize;
        let cm = CostModel::conservative(original);
        let observed_workspace = std::cell::Cell::new(0usize);
        let calls = std::cell::Cell::new(0u32);
        let _: anyhow::Result<u64> = embed_with_trt_retry(
            |cm| {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    Err(anyhow::anyhow!("User allocator error in TRT"))
                } else {
                    observed_workspace.set(cm.max_workspace_bytes);
                    Ok(0)
                }
            },
            &cm,
            0,
            "test",
        );
        assert_eq!(observed_workspace.get(), original / 2);
    }

    #[test]
    fn embed_with_trt_retry_propagates_non_oom_error_immediately() {
        let cm = CostModel::conservative(1024);
        let calls = std::cell::Cell::new(0u32);
        let result: anyhow::Result<u64> = embed_with_trt_retry(
            |_cm| {
                calls.set(calls.get() + 1);
                Err(anyhow::anyhow!("Some unrelated error"))
            },
            &cm,
            0,
            "test",
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 1, "non-OOM error must not retry");
    }

    #[test]
    fn embed_with_trt_retry_propagates_second_failure() {
        let cm = CostModel::conservative(1024);
        let calls = std::cell::Cell::new(0u32);
        let result: anyhow::Result<u64> = embed_with_trt_retry(
            |_cm| {
                calls.set(calls.get() + 1);
                Err(anyhow::anyhow!("User allocator error in TRT"))
            },
            &cm,
            0,
            "test",
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 2);
    }
}
