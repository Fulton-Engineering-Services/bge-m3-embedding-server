// =============================================================================
// SPIKE: CoreML Execution Provider feasibility (fastembed 5.x)
// =============================================================================
// Finding: `TextInitOptions` and `SparseInitOptions` in fastembed 5.x expose
// only `with_cache_dir()` and `with_show_download_progress()`. Neither provides
// a `with_execution_providers(Vec<ExecutionProvider>)` or `SessionOptions`
// injection point. CoreML EP cannot be enabled via the public fastembed API
// without forking the crate or bypassing it to use `ort` directly.
//
// Conclusion: Build natively for `aarch64-apple-darwin` with the bundled ORT.
// Vanilla ARM64 ORT automatically uses Accelerate.framework for BLAS on macOS,
// giving ~3-5× throughput improvement over the x86_64 Docker container —
// sufficient for the fleet embedding distribution goal.
//
// Future path (if CoreML becomes necessary): vendor fastembed, expose
// `InitOptionsUserDefined` with a custom `SessionBuilder` callback, and supply
// a `CoreMLExecutionProvider` built from source ORT with CoreML EP enabled.
// =============================================================================

use anyhow::Result;
use fastembed::{
    EmbeddingModel, SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding,
    TextEmbedding, TextInitOptions,
};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Handle;
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

/// Body of a single embedding worker thread.
///
/// Called from a `spawn_blocking` task. Loads models, signals readiness, then
/// processes requests from the shared channel until it closes.
#[allow(clippy::needless_pass_by_value)]
fn run_worker(
    id: usize,
    cache_dir: PathBuf,
    rx: Arc<Mutex<mpsc::Receiver<EmbedRequest>>>,
    ready_tx: mpsc::Sender<Result<()>>,
    live_workers: Arc<AtomicUsize>,
    loaded_workers: Arc<AtomicUsize>,
    idle_timeout: Option<Duration>,
) -> Result<()> {
    let _guard = WorkerGuard(Arc::clone(&live_workers));
    let span = info_span!("worker", id = id);
    let _span_guard = span.enter();

    info!("Loading models (worker {id})...");
    let load_start = std::time::Instant::now();
    let (initial_dense, initial_sparse) = load_models(&cache_dir, id == 0)?;
    tracing::info!(
        elapsed_ms = load_start.elapsed().as_millis(),
        "Models loaded (worker {id})"
    );

    info!("Worker {id} models loaded — signaling ready");
    let rt = Handle::current();
    let _ = rt.block_on(ready_tx.send(Ok(())));

    let mut dense_model: Option<TextEmbedding> = Some(initial_dense);
    let mut sparse_model: Option<SparseTextEmbedding> = Some(initial_sparse);

    // CONCURRENCY NOTE (COR-2): The shared-receiver pattern with Mutex
    // serializes which worker is *waiting* for the next message — only one
    // worker holds the lock on recv() at a time. The Mutex is released as
    // soon as recv() returns a message, allowing the next idle worker to
    // acquire it. Under normal load (ONNX inference takes 10-100ms per
    // request), at most one request is queued behind the lock.
    info!("Worker {id} entering request loop");
    loop {
        // Apply idle timeout only when models are loaded.
        // Once unloaded, wait indefinitely — no timer wakeups needed.
        let msg = if let Some(timeout) = idle_timeout.filter(|_| dense_model.is_some()) {
            rt.block_on(async {
                tokio::time::timeout(timeout, async { rx.lock().await.recv().await }).await
            })
        } else {
            rt.block_on(async { Ok(rx.lock().await.recv().await) })
        };

        match msg {
            Err(_elapsed) => {
                // Idle timeout fired while models were loaded — unload them.
                dense_model = None;
                sparse_model = None;
                loaded_workers.fetch_sub(1, Ordering::AcqRel);
                tracing::info!("Worker {id} unloaded models after idle timeout");
            }
            Ok(None) => {
                info!("Worker {id} channel closed, shutting down");
                break;
            }
            Ok(Some(request)) => {
                // Reload models if they were unloaded due to idle timeout.
                // This blocks the current request until reload completes (~10-30 s).
                if dense_model.is_none() {
                    tracing::info!("Worker {id} reloading models after idle...");
                    let reload_start = std::time::Instant::now();
                    match load_models(&cache_dir, false) {
                        Ok((d, s)) => {
                            dense_model = Some(d);
                            sparse_model = Some(s);
                            loaded_workers.fetch_add(1, Ordering::AcqRel);
                            tracing::info!(
                                elapsed_ms = reload_start.elapsed().as_millis(),
                                "Worker {id} reloaded models"
                            );
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Worker {id} failed to reload models");
                            // Send the error back to this caller and retry on the next request.
                            let err = anyhow::anyhow!("Model reload failed: {e}");
                            match request {
                                EmbedRequest::Dense { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                                EmbedRequest::Sparse { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                            }
                            continue;
                        }
                    }
                }

                // Models are guaranteed loaded at this point.
                match request {
                    EmbedRequest::Dense { texts, reply } => {
                        let result = dense_model
                            .as_mut()
                            .expect("dense model loaded after reload check")
                            .embed(texts, None)
                            .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Sparse { texts, reply } => {
                        let result = sparse_model
                            .as_mut()
                            .expect("sparse model loaded after reload check")
                            .embed(texts, None)
                            .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                        let _ = reply.send(result);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Loads both the dense and sparse BGE-M3 model instances from `cache_dir`.
///
/// Called at worker startup and again after an idle unload whenever a new
/// request arrives. Both models are always loaded and unloaded as a pair.
fn load_models(
    cache_dir: &Path,
    show_download_progress: bool,
) -> Result<(TextEmbedding, SparseTextEmbedding)> {
    let dense = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(show_download_progress),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load dense model: {e}"))?;

    let sparse = SparseTextEmbedding::try_new(
        SparseInitOptions::new(SparseModel::BGEM3)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(false),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load sparse model: {e}"))?;

    Ok((dense, sparse))
}

#[derive(Clone)]
pub struct EmbedPool {
    tx: mpsc::Sender<EmbedRequest>,
    live_workers: Arc<AtomicUsize>,
    /// Number of workers that currently have model instances loaded in memory.
    ///
    /// Decremented when a worker drops its models after an idle timeout;
    /// incremented when a worker reloads them on the next request.
    /// Used by the `/health` endpoint to distinguish the transient `"idle"`
    /// state (models unloaded, will auto-reload) from a fatal `"fail"` state.
    loaded_workers: Arc<AtomicUsize>,
}

impl EmbedPool {
    /// Spawns `n` embedding worker threads and returns the pool plus an init
    /// handle that resolves once all workers have finished loading their models.
    ///
    /// `idle_timeout` — if `Some`, workers drop their model instances after this
    /// duration of inactivity and reload them transparently on the next request.
    /// Pass `None` to keep models loaded for the lifetime of the process.
    pub fn spawn(
        n: usize,
        cache_dir: PathBuf,
        idle_timeout: Option<Duration>,
    ) -> (Self, JoinHandle<Result<()>>) {
        let capacity = n * 4;
        let (tx, rx) = mpsc::channel::<EmbedRequest>(capacity);
        let rx = Arc::new(Mutex::new(rx));

        // Readiness channel: each worker sends Ok(()) after loading models.
        // Capacity = n so senders never block.
        let (ready_tx, mut ready_rx) = mpsc::channel::<Result<()>>(n);

        let live_workers = Arc::new(AtomicUsize::new(n));
        let loaded_workers = Arc::new(AtomicUsize::new(n));
        let live_workers_for_init = Arc::clone(&live_workers);
        let loaded_workers_for_init = Arc::clone(&loaded_workers);

        let init_handle = tokio::task::spawn(
            async move {
                let mut worker_handles = Vec::with_capacity(n);

                for id in 0..n {
                    let rx_clone = Arc::clone(&rx);
                    let cache_dir_clone = cache_dir.clone();
                    let ready_tx_clone = ready_tx.clone();
                    let live_for_worker = Arc::clone(&live_workers_for_init);
                    let loaded_for_worker = Arc::clone(&loaded_workers_for_init);

                    let handle = tokio::task::spawn_blocking(move || {
                        run_worker(
                            id,
                            cache_dir_clone,
                            rx_clone,
                            ready_tx_clone,
                            live_for_worker,
                            loaded_for_worker,
                            idle_timeout,
                        )
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

        (
            Self {
                tx,
                live_workers,
                loaded_workers,
            },
            init_handle,
        )
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
    #[must_use]
    pub fn live_worker_count(&self) -> usize {
        self.live_workers.load(Ordering::Acquire)
    }

    /// Returns the number of workers that currently have model instances loaded in memory.
    ///
    /// A value of `0` with [`live_worker_count`][Self::live_worker_count] `> 0` indicates
    /// the pool is in an idle state — models will reload automatically on the next request.
    #[must_use]
    pub fn loaded_worker_count(&self) -> usize {
        self.loaded_workers.load(Ordering::Acquire)
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
            loaded_workers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Creates an [`EmbedPool`] that responds with fixed fixture data for testing happy paths.
    ///
    /// The pool has `live_workers = 1` and `loaded_workers = 1` so `check_ready` passes.
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
            loaded_workers: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Creates an [`EmbedPool`] representing the idle state — workers alive but models unloaded.
    ///
    /// `live_workers = 1`, `loaded_workers = 0`. Used to test the `/health` `"idle"` response.
    pub(crate) fn idle_for_test() -> Self {
        let (tx, _rx) = mpsc::channel::<EmbedRequest>(1);
        // _rx is intentionally dropped: tests using this pool only inspect health-state
        // atomics, never send actual embedding requests.
        Self {
            tx,
            live_workers: Arc::new(AtomicUsize::new(1)),
            loaded_workers: Arc::new(AtomicUsize::new(0)),
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

    #[test]
    fn live_worker_count_returns_zero_for_closed_pool() {
        let pool = EmbedPool::closed_for_test();
        assert_eq!(pool.live_worker_count(), 0);
    }

    #[test]
    fn loaded_worker_count_returns_zero_for_closed_pool() {
        let pool = EmbedPool::closed_for_test();
        assert_eq!(pool.loaded_worker_count(), 0);
    }

    #[test]
    fn idle_for_test_has_live_workers_but_no_loaded_workers() {
        let pool = EmbedPool::idle_for_test();
        assert_eq!(pool.live_worker_count(), 1);
        assert_eq!(pool.loaded_worker_count(), 0);
    }
}
