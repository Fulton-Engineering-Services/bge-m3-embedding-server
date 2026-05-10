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
use super::session::load_models;
use super::sparse::embed_sparse;
use super::tokenize::{build_chunk_arrays, tokenize_no_pad};
use super::types::{EmbedRequest, ProbeResult};
use crate::binpack::CostModel;
use crate::config::ModelVariant;
use crate::sysinfo;

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

    info!("Loading models (worker {id})...");
    let load_start = std::time::Instant::now();
    let rt = Handle::current();

    // Measure RSS before loading so the delta accurately reflects this
    // worker's model-weight + arena-baseline contribution, not accumulated
    // OS noise.
    let pre_load_rss = sysinfo::read_process_rss_bytes().unwrap_or(0);
    let initial_models = match load_models(
        &cache_dir,
        id == 0,
        config.model_variant,
        config.max_seq_length,
        config.intra_threads,
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
                            }
                            continue;
                        }
                    }
                }

                let (session, tokenizer) =
                    models.as_mut().expect("models loaded after reload check");

                match request {
                    EmbedRequest::Dense { texts, reply } => {
                        // Load the current cost model snapshot for this request.
                        // ArcSwap::load() is lock-free; the guard keeps the Arc
                        // alive for the duration of embed_dense.
                        let cm_guard = config.cost_model.load();
                        let result = embed_dense(
                            session,
                            tokenizer,
                            &texts,
                            &cm_guard,
                            config.model_variant,
                        )
                        .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                        if let Ok((_, ref stats)) = result {
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
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Sparse { texts, reply } => {
                        let cm_guard = config.cost_model.load();
                        let result = embed_sparse(
                            session,
                            tokenizer,
                            &texts,
                            &cm_guard,
                            config.model_variant,
                        )
                        .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                        if let Ok((_, ref stats)) = result {
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
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Both { texts, reply } => {
                        let cm_guard = config.cost_model.load();
                        let result =
                            embed_both(session, tokenizer, &texts, &cm_guard, config.model_variant)
                                .map_err(|e| anyhow::anyhow!("Dual embed error: {e}"));
                        if let Ok((_, ref stats)) = result {
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
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Probe { texts, reply } => {
                        // Probe: tokenize once without padding, run dense inference
                        // on a single flat batch at the chunk's natural max_seq.
                        let result = run_probe_batch(session, tokenizer, &texts);
                        let _ = reply.send(result);
                    }
                }
            }
        }
    }

    Ok(())
}
