use anyhow::Result;
use fastembed::{
    EmbeddingModel, SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding,
    TextEmbedding, TextInitOptions,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{info, info_span, Instrument};

pub enum EmbedRequest {
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    // TODO(ARC-2): EmbedPool currently exposes fastembed::SparseEmbedding
    // directly, coupling callers to fastembed internals. A future
    // SparseResult newtype would decouple this.
    Sparse {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<SparseEmbedding>>>,
    },
}

/// RAII guard that decrements the live-worker counter when dropped.
/// Guarantees decrement fires on clean exit AND on panic unwind.
struct WorkerGuard(Arc<AtomicUsize>);

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        let remaining = self.0.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
        if remaining == 0 {
            tracing::error!("All embedding workers have exited — pool is degraded");
        } else {
            tracing::warn!(remaining, "Embedding worker exited");
        }
    }
}

#[derive(Clone)]
pub struct EmbedPool {
    tx: mpsc::Sender<EmbedRequest>,
    live_workers: Arc<AtomicUsize>,
}

impl EmbedPool {
    pub fn spawn(n: usize, cache_dir: PathBuf) -> (Self, JoinHandle<Result<()>>) {
        let capacity = n * 4;
        let (tx, rx) = mpsc::channel::<EmbedRequest>(capacity);
        let rx = Arc::new(Mutex::new(rx));

        // Readiness channel: each worker sends Ok(()) after loading models.
        // Capacity = n so senders never block.
        let (ready_tx, mut ready_rx) = mpsc::channel::<Result<()>>(n);

        let live_workers = Arc::new(AtomicUsize::new(n));
        let live_workers_for_init = Arc::clone(&live_workers);

        let init_handle = tokio::task::spawn(
            async move {
                let mut worker_handles = Vec::with_capacity(n);

                for id in 0..n {
                    let rx_clone = Arc::clone(&rx);
                    let cache_dir_clone = cache_dir.clone();
                    let ready_tx_clone = ready_tx.clone();
                    let live_for_worker = Arc::clone(&live_workers_for_init);

                    let handle = tokio::task::spawn_blocking(move || {
                        let _guard = WorkerGuard(Arc::clone(&live_for_worker));
                        let span = info_span!("worker", id = id);
                        let _span_guard = span.enter();

                        info!("Loading dense model (worker {id})...");
                        let mut dense_model = TextEmbedding::try_new(
                            TextInitOptions::new(EmbeddingModel::BGEM3)
                                .with_cache_dir(cache_dir_clone.clone())
                                .with_show_download_progress(id == 0),
                        )
                        .map_err(|e| anyhow::anyhow!("Failed to load dense model: {e}"))?;

                        info!("Loading sparse model (worker {id})...");
                        let mut sparse_model = SparseTextEmbedding::try_new(
                            SparseInitOptions::new(SparseModel::BGEM3)
                                .with_cache_dir(cache_dir_clone)
                                .with_show_download_progress(false),
                        )
                        .map_err(|e| anyhow::anyhow!("Failed to load sparse model: {e}"))?;

                        info!("Worker {id} models loaded — signaling ready");

                        let rt = tokio::runtime::Handle::current();
                        let _ = rt.block_on(ready_tx_clone.send(Ok(())));

                        // CONCURRENCY NOTE (COR-2): The shared-receiver pattern
                        // with Mutex serializes which worker is *waiting* for the
                        // next message — only one worker holds the lock on recv()
                        // at a time. The Mutex is released as soon as recv()
                        // returns a message, allowing the next idle worker to
                        // acquire it. Under normal load (ONNX inference takes
                        // 10-100ms per request), at most one request is queued
                        // behind the lock. This is acceptable for this service's
                        // throughput requirements.
                        info!("Worker {id} entering request loop");
                        loop {
                            let request = rt.block_on(async { rx_clone.lock().await.recv().await });

                            match request {
                                None => {
                                    info!("Worker {id} channel closed, shutting down");
                                    break;
                                }
                                Some(EmbedRequest::Dense { texts, reply }) => {
                                    let result = dense_model
                                        .embed(texts, None)
                                        .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                                    let _ = reply.send(result);
                                }
                                Some(EmbedRequest::Sparse { texts, reply }) => {
                                    let result = sparse_model
                                        .embed(texts, None)
                                        .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                                    let _ = reply.send(result);
                                }
                            }
                        }

                        Ok::<(), anyhow::Error>(())
                    });

                    worker_handles.push(handle);
                }

                // Drop our copy so recv() can detect early worker exit.
                drop(ready_tx);

                // Collect exactly n readiness signals.
                for i in 0..n {
                    match ready_rx.recv().await {
                        Some(Ok(())) => {
                            info!("Worker {i} signaled ready ({}/{n})", i + 1);
                        }
                        Some(Err(e)) => {
                            return Err(anyhow::anyhow!("Worker failed to load models: {e}"));
                        }
                        None => {
                            return Err(anyhow::anyhow!(
                                "Worker exited before signaling readiness (got {i}/{n})"
                            ));
                        }
                    }
                }

                // Workers continue running in the background. Their
                // spawn_blocking tasks are detached when handles are dropped
                // and will self-terminate when the channel closes (pool drop).
                drop(worker_handles);

                Ok(())
            }
            .instrument(info_span!("embed_pool")),
        );

        (Self { tx, live_workers }, init_handle)
    }

    pub async fn dense(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
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

    pub async fn sparse(&self, texts: Vec<String>) -> Result<Vec<SparseEmbedding>> {
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

    /// Returns the number of currently live embedding workers.
    /// Returns 0 if all workers have exited (pool is fully degraded).
    pub fn live_worker_count(&self) -> usize {
        self.live_workers.load(Ordering::Acquire)
    }
}

#[cfg(test)]
impl EmbedPool {
    /// Creates an [`EmbedPool`] with an already-closed channel for testing error paths.
    pub(crate) fn closed_for_test() -> Self {
        let (tx, rx) = mpsc::channel::<EmbedRequest>(1);
        drop(rx);
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Creates an [`EmbedPool`] that responds with fixed fixture data for testing happy paths.
    ///
    /// The pool has `live_workers = 1` so `check_ready` passes.
    /// Responds to every dense request with `dense_fixture` and every sparse request
    /// with `sparse_fixture`, regardless of input text.
    ///
    /// Note: `fastembed::SparseEmbedding` does not implement `Clone`, so the sparse
    /// fixture is consumed on the first sparse call (via `drain`). Tests that call
    /// sparse should use the fixture for a single request.
    pub(crate) fn with_fixed_responses(
        dense_fixture: Vec<Vec<f32>>,
        sparse_fixture: Vec<fastembed::SparseEmbedding>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<EmbedRequest>(8);
        let dense = Arc::new(dense_fixture);
        let sparse = Arc::new(std::sync::Mutex::new(sparse_fixture));
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                match req {
                    EmbedRequest::Dense { reply, .. } => {
                        let _ = reply.send(Ok((*dense).clone()));
                    }
                    EmbedRequest::Sparse { reply, .. } => {
                        // SparseEmbedding doesn't implement Clone — drain the fixture
                        // vec on each call. Tests that call sparse should use the
                        // fixture for a single request.
                        let result = sparse.lock().unwrap().drain(..).collect();
                        let _ = reply.send(Ok(result));
                    }
                }
            }
        });
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dense_returns_error_when_channel_closed() {
        let pool = EmbedPool::closed_for_test();
        let result = pool.dense(vec!["hello".into()]).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("channel closed"),
            "expected channel closed error"
        );
    }

    #[tokio::test]
    async fn sparse_returns_error_when_channel_closed() {
        let pool = EmbedPool::closed_for_test();
        let result = pool.sparse(vec!["hello".into()]).await;
        // SparseEmbedding doesn't implement Debug, so use .err().unwrap()
        // instead of .unwrap_err().
        let err = result.err().expect("expected an error");
        assert!(
            err.to_string().contains("channel closed"),
            "expected channel closed error"
        );
    }
}
