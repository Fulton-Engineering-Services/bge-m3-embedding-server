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

//! `EmbedPool` async wrapper around the worker thread pool.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{info, info_span, Instrument};

use super::math::median_usize;
use super::trt_warmup::shard_shapes;
use super::types::{DualEmbedding, EmbedRequest, EmbedStats, ProbeResult, SparseEmbedding};
use super::worker::{run_worker, WorkerConfig};
use crate::config::EpSelection;

/// Sender half of the engine propagation broadcast channel.
type PropTx = tokio::sync::broadcast::Sender<(usize, usize)>;

/// Receiver half of the engine propagation broadcast channel.
type PropRx = tokio::sync::broadcast::Receiver<(usize, usize)>;

/// Creates a `(Sender, Receiver)` pair for engine propagation when enabled.
///
/// Returns `(Some(tx_clone), Some(rx))` when `enabled` is true, and
/// `(None, None)` otherwise.  Called once per worker in `EmbedPool::spawn`.
fn make_propagation_pair(enabled: bool, tx: &PropTx) -> (Option<PropTx>, Option<PropRx>) {
    if enabled {
        (Some(tx.clone()), Some(tx.subscribe()))
    } else {
        (None, None)
    }
}

/// Awaits worker `id`'s readiness signal, but also resolves immediately if the
/// worker's `JoinHandle` finishes before signaling readiness.
///
/// This is the defensive half of the leader-failure fix: `EmbedPool::spawn`
/// itself owns the original `ready_tx`, which it cannot drop until after every
/// follower has been spawned. If the worker panics — or for any future reason
/// drops its `ready_tx` clone before sending — `ready_rx.recv()` would never
/// see `None` (the init task's own clone is still alive) and the init future
/// would park forever. Selecting on the `JoinHandle` guarantees we always make
/// progress when the worker exits, regardless of whether the worker had a
/// chance to send its outcome.
///
/// Returns the worker's `rss_delta` on success. Both branches are propagated
/// as a typed `Result<usize>` so callers can short-circuit with `?` and avoid
/// duplicating error-construction logic at every spawn site.
async fn await_worker_signal(
    id: usize,
    handle: &mut JoinHandle<Result<()>>,
    ready_rx: &mut mpsc::Receiver<Result<usize>>,
) -> Result<usize> {
    tokio::select! {
        // Bias toward the explicit ready signal: if the worker both sent
        // `Err(...)` and then returned `Err(...)`, both branches are ready and
        // the in-band failure message is more actionable than the join error.
        biased;
        msg = ready_rx.recv() => match msg {
            Some(Ok(delta)) => Ok(delta),
            Some(Err(e)) => Err(anyhow::anyhow!("Worker {id} failed to load models: {e}")),
            None => Err(anyhow::anyhow!(
                "Worker {id} exited before signaling readiness"
            )),
        },
        join_res = handle => match join_res {
            Ok(Ok(())) => Err(anyhow::anyhow!(
                "Worker {id} exited cleanly without signaling readiness"
            )),
            Ok(Err(e)) => Err(anyhow::anyhow!(
                "Worker {id} exited with error before signaling ready: {e}"
            )),
            Err(panic_err) => Err(anyhow::anyhow!(
                "Worker {id} panicked before signaling ready: {panic_err}"
            )),
        },
    }
}

/// Async handle to the embedding worker thread pool.
///
/// Wraps a bounded `mpsc` channel shared by `n` `spawn_blocking` worker threads.
/// Each worker owns its own ORT session and tokenizer; the pool dispatches
/// `EmbedRequest` variants to whichever worker is free next.
///
/// Clone is cheap — the underlying channel sender and atomic counters are
/// reference-counted.
#[derive(Clone)]
pub struct EmbedPool {
    tx: mpsc::Sender<EmbedRequest>,
    live_workers: Arc<AtomicUsize>,
    /// Number of workers that currently have model instances loaded in memory.
    loaded_workers: Arc<AtomicUsize>,
    /// Median RSS delta (bytes) measured across all workers during sequential
    /// model load.
    ///
    /// Workers load one at a time (leader first, then followers in sequence).
    /// Each reports its own RSS before/after `load_models()` via `ready_tx`.
    /// The pool stores the median once all workers have signaled ready — robust
    /// to one outlier from page-cache settling or ORT arena init jitter.
    ///
    /// Used by `run_readiness_probe` to correctly deduct the model-weight
    /// footprint from the available workspace before computing per-worker
    /// budget. Returns `0` on non-Linux targets where RSS measurement is
    /// unavailable, or before the init task has completed.
    model_rss_per_worker_bytes: Arc<AtomicUsize>,
    /// Broadcast sender for cross-worker TRT engine cache propagation.
    ///
    /// When `Some`, after any worker writes a new engine plan to EFS, a
    /// `(batch, seq)` shape notification is broadcast to every subscribed
    /// worker so they eagerly run `trt_prewarm` (~1-3s fast disk-load) instead
    /// of paying full JIT cost on the next real request. `None` when
    /// `BGE_M3_ENGINE_PROPAGATION_ENABLED=0`.
    engine_propagation_tx: Option<PropTx>,
}

impl EmbedPool {
    /// Broadcasts a `(batch, seq)` shape notification to all subscribed peer
    /// workers, signaling that the TRT engine plan for this shape is now on EFS.
    ///
    /// Workers receive the notification in their main loop drain and run
    /// Broadcasts a `(batch, seq)` shape notification to all subscribed peer
    /// workers, signaling that the TRT engine plan for this shape is now on EFS.
    ///
    /// Workers receive the notification in their main loop drain and run
    /// `trt_prewarm` against their own session for a ~1-3s fast disk-load
    /// rather than paying full JIT cost on the next real request for this shape.
    ///
    /// `send` returns `Err` only when there are zero subscribers (all workers
    /// gone). This is non-fatal; the engine plan is still on EFS.
    pub fn broadcast_engine_ready(&self, shape: (usize, usize)) {
        if let Some(tx) = &self.engine_propagation_tx {
            let _ = tx.send(shape);
        }
    }

    /// Spawns `n` embedding worker threads and returns the pool plus an init
    /// handle that resolves once all workers have finished loading their models.
    ///
    /// When a GPU execution provider (`cuda` or `tensorrt`) is selected, `n` is
    /// clamped to `config.gpu_count`: each worker is pinned to a distinct CUDA
    /// device (`device_id = worker_index % gpu_count`). For TRT, the full
    /// warmup shape list is sharded across workers via a stride partition so
    /// each GPU compiles a disjoint subset in parallel, then shares the results
    /// via the EFS engine cache.
    #[allow(clippy::too_many_lines)]
    pub fn spawn(
        n: usize,
        cache_dir: PathBuf,
        config: WorkerConfig,
    ) -> (Self, JoinHandle<Result<()>>) {
        let gpu_count = config.gpu_count.max(1);
        let n = if config.ep != EpSelection::Cpu && n > gpu_count {
            tracing::warn!(
                requested = n,
                clamped = gpu_count,
                ep = %config.ep,
                gpu_count,
                "BGE_M3_WORKERS exceeds BGE_M3_GPU_COUNT for GPU EP — clamping. \
                 Set BGE_M3_GPU_COUNT to match the number of GPU devices on this instance."
            );
            gpu_count
        } else {
            n
        };
        let capacity = n * 4;
        let (tx, rx) = mpsc::channel::<EmbedRequest>(capacity);
        let rx = Arc::new(Mutex::new(rx));

        // Broadcast channel for cross-worker engine cache propagation.
        // Enabled when: TRT EP is active AND the jit_suspect_tx channel exists
        // (adaptive warmup enabled). This matches the default coupling of
        // engine_propagation_enabled to adaptive_warmup_enabled in Config.
        // Workers ignore the rx if EP is not TRT via the run_worker drain guard.
        let (engine_propagation_tx, _seed_rx) =
            tokio::sync::broadcast::channel::<(usize, usize)>(32);
        let engine_propagation_enabled =
            config.ep == EpSelection::TensorRt && config.jit_suspect_tx.is_some();
        let engine_propagation_tx_for_pool = if engine_propagation_enabled {
            Some(engine_propagation_tx.clone())
        } else {
            None
        };
        // Clone for capture into the async init task.
        let engine_propagation_tx_for_init = engine_propagation_tx.clone();

        // Channel carries Result<usize> where the Ok variant is the RSS delta
        // (bytes) measured by each worker around load_models().
        let (ready_tx, mut ready_rx) = mpsc::channel::<Result<usize>>(n);

        let live_workers = Arc::new(AtomicUsize::new(n));
        let loaded_workers = Arc::new(AtomicUsize::new(0));
        let model_rss_per_worker_bytes = Arc::new(AtomicUsize::new(0));
        let live_workers_for_init = Arc::clone(&live_workers);
        let loaded_workers_for_init = Arc::clone(&loaded_workers);
        let model_rss_for_init = Arc::clone(&model_rss_per_worker_bytes);

        let init_handle = tokio::task::spawn(
            async move {
                let mut worker_handles = Vec::with_capacity(n);

                let spawn_worker = |id: usize,
                                    ready_tx_clone: mpsc::Sender<Result<usize>>,
                                    worker_config: WorkerConfig,
                                    prop_tx: Option<PropTx>,
                                    prop_rx: Option<PropRx>|
                 -> JoinHandle<Result<()>> {
                    let rx_clone = Arc::clone(&rx);
                    let cache_dir_clone = cache_dir.clone();
                    let live_for_worker = Arc::clone(&live_workers_for_init);
                    let loaded_for_worker = Arc::clone(&loaded_workers_for_init);
                    tokio::task::spawn_blocking(move || {
                        run_worker(
                            id,
                            cache_dir_clone,
                            rx_clone,
                            ready_tx_clone,
                            live_for_worker,
                            loaded_for_worker,
                            worker_config,
                            prop_tx,
                            prop_rx,
                        )
                    })
                };

                // Build a per-worker config: assign CUDA device ID and, for TRT
                // EP with multiple workers, shard the warmup shapes across GPUs.
                // The device_id is computed as `worker_index % gpu_count` so
                // workers round-robin across available GPUs. Shard partition is
                // stride-based so the most expensive shapes land on different
                // workers — see `trt_warmup::shard_shapes` for the rationale.
                let make_worker_config = |id: usize| -> WorkerConfig {
                    let mut wc = config.clone();
                    wc.device_id =
                        u32::try_from(id).unwrap_or(0) % u32::try_from(gpu_count).unwrap_or(1);
                    if config.ep == EpSelection::TensorRt
                        && n > 1
                        && !config.trt_warmup_shapes.is_empty()
                    {
                        wc.trt_warmup_shapes = shard_shapes(&config.trt_warmup_shapes, id, n);
                        info!(
                            worker_id = id,
                            gpu_device = wc.device_id,
                            shard_shapes = wc.trt_warmup_shapes.len(),
                            total_shapes = config.trt_warmup_shapes.len(),
                            total_workers = n,
                            "TRT multi-GPU: worker assigned GPU device with warmup shard"
                        );
                    }
                    wc
                };

                // Collect per-worker RSS deltas for median aggregation.
                // Median is robust to one outlier from transient kernel snapshot
                // quirk (page-cache settling, ORT arena init jitter) while still
                // using all N independent measurements.
                let mut rss_deltas: Vec<usize> = Vec::with_capacity(n);

                // --- Phase 1: spawn leader worker (may download models) ---
                let (leader_prop_tx, leader_prop_rx) = make_propagation_pair(
                    engine_propagation_enabled,
                    &engine_propagation_tx_for_init,
                );
                let mut leader_handle = spawn_worker(
                    0,
                    ready_tx.clone(),
                    make_worker_config(0),
                    leader_prop_tx,
                    leader_prop_rx,
                );
                let leader_msg = await_worker_signal(0, &mut leader_handle, &mut ready_rx).await?;
                worker_handles.push(leader_handle);
                loaded_workers_for_init.fetch_add(1, Ordering::AcqRel);
                rss_deltas.push(leader_msg);
                info!(
                    rss_delta_mb = leader_msg / (1024 * 1024),
                    "Leader worker ready, model cache warm (1/{n})"
                );

                // --- Phase 2: spawn follower workers one at a time.
                //
                // Workers load sequentially: spawn one, await its ready signal,
                // then spawn the next. This ensures each worker's RSS delta
                // (pre/post load_models) reflects only that worker's ORT session
                // allocation — not the cumulative effect of other workers loading
                // in parallel. Parallel loading caused the 2026-05-09 measurement
                // contamination bug: all followers read post_load_rss after most
                // other sessions had already mmap'd, producing an inflated
                // rss_delta ≈ N × model_size and driving per_worker_workspace to 0.
                //
                // Startup cost: ~4-6s per worker × 6 followers ≈ 24-36s total,
                // well within the configured startPeriod (300s).
                for id in 1..n {
                    let (prop_tx, prop_rx) = make_propagation_pair(
                        engine_propagation_enabled,
                        &engine_propagation_tx_for_init,
                    );
                    let mut handle = spawn_worker(
                        id,
                        ready_tx.clone(),
                        make_worker_config(id),
                        prop_tx,
                        prop_rx,
                    );
                    let delta = await_worker_signal(id, &mut handle, &mut ready_rx).await?;
                    worker_handles.push(handle);
                    loaded_workers_for_init.fetch_add(1, Ordering::AcqRel);
                    rss_deltas.push(delta);
                    info!(
                        rss_delta_mb = delta / (1024 * 1024),
                        "Follower worker signaled ready ({}/{n})",
                        id + 1
                    );
                }

                drop(ready_tx);
                drop(worker_handles);

                // Compute and store the median delta as the per-worker model footprint.
                let median = median_usize(&mut rss_deltas);
                model_rss_for_init.store(median, Ordering::Release);
                info!(
                    median_rss_mb = median / (1024 * 1024),
                    samples = rss_deltas.len(),
                    "All workers ready — per-worker model RSS median computed"
                );

                Ok(())
            }
            .instrument(info_span!("embed_pool")),
        );

        (
            Self {
                tx,
                live_workers,
                loaded_workers,
                model_rss_per_worker_bytes,
                engine_propagation_tx: engine_propagation_tx_for_pool,
            },
            init_handle,
        )
    }

    /// Runs dense (float32) embedding inference on `texts`.
    ///
    /// # Errors
    ///
    /// - Returns `Err` if the worker channel has closed (pool shut down).
    /// - Returns `Err` if the worker drops the reply sender before responding.
    /// - Returns `Err` if the ORT session fails during inference.
    pub async fn dense(&self, texts: Vec<String>) -> Result<(Vec<Vec<f32>>, EmbedStats)> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Dense {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }

    /// Runs sparse (SPLADE-style) embedding inference on `texts`.
    ///
    /// # Errors
    ///
    /// - Returns `Err` if the worker channel has closed (pool shut down).
    /// - Returns `Err` if the worker drops the reply sender before responding.
    /// - Returns `Err` if the ORT session fails during inference.
    pub async fn sparse(&self, texts: Vec<String>) -> Result<(Vec<SparseEmbedding>, EmbedStats)> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Sparse {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }

    /// Runs a single forward pass that yields both dense and sparse embeddings.
    ///
    /// Equivalent to calling [`Self::dense`] and [`Self::sparse`] back-to-back,
    /// but uses one `session.run()` per chunk instead of two — at near-zero
    /// marginal GPU cost.
    ///
    /// # Errors
    ///
    /// - Returns `Err` if the worker channel has closed (pool shut down).
    /// - Returns `Err` if the worker drops the reply sender before responding.
    /// - Returns `Err` if the ORT session fails during inference.
    pub async fn both(&self, texts: Vec<String>) -> Result<(Vec<DualEmbedding>, EmbedStats)> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Both {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }

    /// Sends a probe request to a single worker and returns the result.
    /// Only called during init before `ready` is set.
    pub(crate) async fn probe(&self, texts: Vec<String>) -> Result<ProbeResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Probe {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }

    #[must_use]
    /// Returns the number of worker threads currently alive (not yet exited).
    pub fn live_worker_count(&self) -> usize {
        self.live_workers.load(Ordering::Acquire)
    }

    #[must_use]
    /// Returns the number of workers that currently have model instances loaded in memory.
    ///
    /// A worker transitions from loaded to unloaded after the [`crate::config::Config::idle_timeout`]
    /// elapses with no incoming requests, and back to loaded on the next request.
    pub fn loaded_worker_count(&self) -> usize {
        self.loaded_workers.load(Ordering::Acquire)
    }

    /// Returns the number of requests currently queued but not yet picked up
    /// by a worker. Uses the channel's current vs max capacity.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.tx.max_capacity().saturating_sub(self.tx.capacity())
    }

    /// Returns the median RSS delta (bytes) measured across all workers during
    /// sequential model load.
    ///
    /// This is the per-worker model-weight footprint used by
    /// `run_readiness_probe` to compute the per-worker workspace budget.
    /// Returns `0` on non-Linux targets where RSS measurement is unavailable,
    /// or before the init task has completed.
    #[must_use]
    pub fn model_rss_per_worker_bytes(&self) -> usize {
        self.model_rss_per_worker_bytes.load(Ordering::Acquire)
    }

    /// Sends an adaptive warmup request to an available worker.
    ///
    /// The worker calls `trt_prewarm` for `(batch, seq)` and replies on `ack`
    /// with the compile duration in milliseconds, or an error on failure.
    /// On non-TRT workers the reply is `Ok(0)` immediately.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the worker channel has closed (pool shut down).
    pub async fn send_adaptive_warmup(
        &self,
        batch: usize,
        seq: usize,
        ack: tokio::sync::oneshot::Sender<anyhow::Result<u64>>,
    ) -> Result<(), ()> {
        self.tx
            .send(EmbedRequest::AdaptiveWarmup { batch, seq, ack })
            .await
            .map_err(|_| ())
    }

    /// Returns a clone of the `Arc<AtomicUsize>` backing `live_worker_count`.
    ///
    /// Used by the warmup-only path in `lib.rs` to poll worker exit progress
    /// AFTER the [`EmbedPool`] itself has been dropped. Dropping the pool
    /// closes the request channel, which signals workers to break out of
    /// their receive loops and drop their ORT sessions; the live counter is
    /// the only readable signal that those drop paths have completed.
    ///
    /// Returning a clone of the raw `Arc` (rather than a snapshot) lets the
    /// caller hold a reference across the drop boundary without having to
    /// keep the pool's other state alive.
    #[must_use]
    pub fn live_workers_for_shutdown(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.live_workers)
    }
}

// ---------------------------------------------------------------------------
// Test helpers (cfg(test)-gated)
// ---------------------------------------------------------------------------

#[cfg(test)]
impl EmbedPool {
    pub(crate) fn closed_for_test() -> Self {
        let (tx, rx) = mpsc::channel::<EmbedRequest>(1);
        drop(rx);
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(0)),
            loaded_workers: Arc::new(AtomicUsize::new(0)),
            model_rss_per_worker_bytes: Arc::new(AtomicUsize::new(0)),
            engine_propagation_tx: None,
        }
    }

    pub(crate) fn with_fixed_responses(
        dense_fixture: Vec<Vec<f32>>,
        sparse_fixture: Vec<SparseEmbedding>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<EmbedRequest>(8);
        let dense = Arc::new(dense_fixture);
        let sparse = Arc::new(sparse_fixture);
        let dense_both = Arc::clone(&dense);
        let sparse_both = Arc::clone(&sparse);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                match req {
                    EmbedRequest::Dense { reply, .. } => {
                        let _ = reply.send(Ok(((*dense).clone(), EmbedStats::default())));
                    }
                    EmbedRequest::Sparse { reply, .. } => {
                        let _ = reply.send(Ok(((*sparse).clone(), EmbedStats::default())));
                    }
                    EmbedRequest::Both { reply, .. } => {
                        // Pair dense_fixture[i] with sparse_fixture[i] elementwise.
                        // Truncate to the shorter of the two so the test fixture is
                        // self-consistent.
                        let pairs: Vec<DualEmbedding> = dense_both
                            .iter()
                            .zip(sparse_both.iter())
                            .map(|(d, s)| DualEmbedding {
                                dense: d.clone(),
                                sparse: s.clone(),
                            })
                            .collect();
                        let _ = reply.send(Ok((pairs, EmbedStats::default())));
                    }
                    EmbedRequest::Probe { reply, .. } => {
                        let _ = reply.send(Ok(ProbeResult {
                            rss_before: 0,
                            rss_after: 0,
                        }));
                    }
                    EmbedRequest::AdaptiveWarmup { ack, .. } => {
                        let _ = ack.send(Ok(0));
                    }
                }
            }
        });
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(1)),
            loaded_workers: Arc::new(AtomicUsize::new(1)),
            model_rss_per_worker_bytes: Arc::new(AtomicUsize::new(0)),
            engine_propagation_tx: None,
        }
    }

    pub(crate) fn idle_for_test() -> Self {
        let (tx, _rx) = mpsc::channel::<EmbedRequest>(1);
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(1)),
            loaded_workers: Arc::new(AtomicUsize::new(0)),
            model_rss_per_worker_bytes: Arc::new(AtomicUsize::new(0)),
            engine_propagation_tx: None,
        }
    }

    /// Creates an [`EmbedPool`] backed by `with_fixed_responses` with the
    /// provided broadcast sender wired in.
    ///
    /// Used by propagation tests that need to verify `broadcast_engine_ready`
    /// without spawning real workers.
    pub(crate) fn for_propagation_test(
        dense_fixture: Vec<Vec<f32>>,
        sparse_fixture: Vec<SparseEmbedding>,
        engine_tx: PropTx,
    ) -> Self {
        let pool = Self::with_fixed_responses(dense_fixture, sparse_fixture);
        Self {
            tx: pool.tx,
            live_workers: pool.live_workers,
            loaded_workers: pool.loaded_workers,
            model_rss_per_worker_bytes: pool.model_rss_per_worker_bytes,
            engine_propagation_tx: Some(engine_tx),
        }
    }

    /// Returns the raw `Arc<AtomicUsize>` backing `model_rss_per_worker_bytes`.
    ///
    /// Test-only; allows injecting a specific value to assert aggregation logic
    /// without running actual model loads.
    pub(crate) fn model_rss_per_worker_bytes_atomic(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.model_rss_per_worker_bytes)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adaptive_warmup_tests {
    use super::*;

    /// `send_adaptive_warmup` must return `Err(())` when the channel receiver
    /// has been dropped (pool shut down).  `closed_for_test()` drops the
    /// receiver immediately after channel creation, so the very first send
    /// observes a closed channel.
    #[tokio::test]
    async fn send_adaptive_warmup_returns_err_when_channel_closed() {
        let pool = EmbedPool::closed_for_test();
        let (ack_tx, _ack_rx) = oneshot::channel();
        let result = pool.send_adaptive_warmup(1, 128, ack_tx).await;
        assert!(
            result.is_err(),
            "send_adaptive_warmup must return Err(()) when channel is closed"
        );
    }
}
