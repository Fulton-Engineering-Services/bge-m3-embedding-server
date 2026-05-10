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
use super::types::{DualEmbedding, EmbedRequest, EmbedStats, ProbeResult, SparseEmbedding};
use super::worker::{run_worker, WorkerConfig};

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
}

impl EmbedPool {
    /// Spawns `n` embedding worker threads and returns the pool plus an init
    /// handle that resolves once all workers have finished loading their models.
    pub fn spawn(
        n: usize,
        cache_dir: PathBuf,
        config: WorkerConfig,
    ) -> (Self, JoinHandle<Result<()>>) {
        let capacity = n * 4;
        let (tx, rx) = mpsc::channel::<EmbedRequest>(capacity);
        let rx = Arc::new(Mutex::new(rx));

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
                                    worker_config: WorkerConfig|
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
                        )
                    })
                };

                // Collect per-worker RSS deltas for median aggregation.
                // Median is robust to one outlier from transient kernel snapshot
                // quirk (page-cache settling, ORT arena init jitter) while still
                // using all N independent measurements.
                let mut rss_deltas: Vec<usize> = Vec::with_capacity(n);

                // --- Phase 1: spawn leader worker (may download models) ---
                worker_handles.push(spawn_worker(0, ready_tx.clone(), config.clone()));

                match ready_rx.recv().await {
                    Some(Ok(delta)) => {
                        loaded_workers_for_init.fetch_add(1, Ordering::AcqRel);
                        rss_deltas.push(delta);
                        info!(
                            rss_delta_mb = delta / (1024 * 1024),
                            "Leader worker ready, model cache warm (1/{n})"
                        );
                    }
                    Some(Err(e)) => {
                        return Err(anyhow::anyhow!("Leader worker failed to load models: {e}"));
                    }
                    None => {
                        return Err(anyhow::anyhow!(
                            "Leader worker exited before signaling readiness"
                        ));
                    }
                }

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
                    worker_handles.push(spawn_worker(id, ready_tx.clone(), config.clone()));

                    match ready_rx.recv().await {
                        Some(Ok(delta)) => {
                            loaded_workers_for_init.fetch_add(1, Ordering::AcqRel);
                            rss_deltas.push(delta);
                            info!(
                                rss_delta_mb = delta / (1024 * 1024),
                                "Follower worker signaled ready ({}/{n})",
                                id + 1
                            );
                        }
                        Some(Err(e)) => {
                            return Err(anyhow::anyhow!("Worker {id} failed to load models: {e}"));
                        }
                        None => {
                            return Err(anyhow::anyhow!(
                                "Worker {id} exited before signaling readiness ({id}/{n})"
                            ));
                        }
                    }
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
            },
            init_handle,
        )
    }

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
    pub fn live_worker_count(&self) -> usize {
        self.live_workers.load(Ordering::Acquire)
    }

    #[must_use]
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
                }
            }
        });
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(1)),
            loaded_workers: Arc::new(AtomicUsize::new(1)),
            model_rss_per_worker_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) fn idle_for_test() -> Self {
        let (tx, _rx) = mpsc::channel::<EmbedRequest>(1);
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(1)),
            loaded_workers: Arc::new(AtomicUsize::new(0)),
            model_rss_per_worker_bytes: Arc::new(AtomicUsize::new(0)),
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
