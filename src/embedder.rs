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
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
fn run_worker(
    id: usize,
    cache_dir: PathBuf,
    rx: Arc<Mutex<mpsc::Receiver<EmbedRequest>>>,
    ready_tx: mpsc::Sender<Result<()>>,
    live_workers: Arc<AtomicUsize>,
    loaded_workers: Arc<AtomicUsize>,
    idle_timeout: Option<Duration>,
    onnx_batch_size: usize,
) -> Result<()> {
    let _guard = WorkerGuard(Arc::clone(&live_workers));
    let span = info_span!("worker", id = id);
    let _span_guard = span.enter();

    info!("Loading models (worker {id})...");
    let load_start = std::time::Instant::now();
    let rt = Handle::current();
    let (initial_dense, initial_sparse) = match load_models(&cache_dir, id == 0) {
        Ok(models) => {
            tracing::info!(
                elapsed_ms = load_start.elapsed().as_millis(),
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

    info!("Worker {id} models loaded — signaling ready");
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
                            .embed(texts, Some(onnx_batch_size))
                            .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Sparse { texts, reply } => {
                        let result = sparse_model
                            .as_mut()
                            .expect("sparse model loaded after reload check")
                            .embed(texts, Some(onnx_batch_size))
                            .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                        let _ = reply.send(result);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Returns the execution providers to use for ONNX Runtime sessions.
///
/// On macOS (Apple Silicon), registers the `CoreML` Execution Provider to
/// dispatch model subgraphs to the Neural Engine, GPU, or Accelerate (AMX).
/// `CoreML` EP requires a source-built ORT with `-Donnxruntime_USE_COREML=ON`
/// pointed at via `ORT_LIB_LOCATION`. If the EP fails to register, ORT
/// silently falls back to the CPU EP with MLAS NEON kernels.
///
/// Configuration rationale:
/// - **`MLProgram`** — newer `CoreML` format with broader op coverage and
///   better optimisation passes; requires macOS 12+ (production targets Tahoe/26).
/// - **`FastPrediction`** — trades higher model-specialisation time and
///   memory for lower per-request latency.
/// - **Model cache** — caches the compiled `CoreML` model to
///   `{cache_dir}/coreml`, eliminating 5–15 s recompilation per session
///   load (critical for the idle-unload-reload cycle).
///
/// On all other platforms, returns an empty vec (CPU EP only).
fn execution_providers(cache_dir: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(target_os = "macos")]
    {
        let coreml_cache = cache_dir.join("coreml");
        vec![ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_specialization_strategy(ort::ep::coreml::SpecializationStrategy::FastPrediction)
            .with_model_cache_dir(coreml_cache.display().to_string())
            .build()]
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cache_dir;
        vec![]
    }
}

/// Loads both the dense and sparse BGE-M3 model instances from `cache_dir`.
///
/// Called at worker startup and again after an idle unload whenever a new
/// request arrives. Both models are always loaded and unloaded as a pair.
fn load_models(
    cache_dir: &Path,
    show_download_progress: bool,
) -> Result<(TextEmbedding, SparseTextEmbedding)> {
    let eps = execution_providers(cache_dir);

    let dense = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(show_download_progress)
            .with_execution_providers(eps.clone()),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load dense model: {e}"))?;

    let sparse = SparseTextEmbedding::try_new(
        SparseInitOptions::new(SparseModel::BGEM3)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(false)
            .with_execution_providers(eps),
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
    ///
    /// # Cold-start ordering
    ///
    /// Worker 0 (the "leader") is spawned and awaited **before** any follower
    /// workers. This guarantees the model cache is warm before followers start.
    ///
    /// `hf-hub` acquires per-blob exclusive file locks (`flock(LOCK_EX)`) during
    /// download with a hardcoded 5-second retry window. BGE-M3 ONNX models are
    /// ~2 GB — a fresh download takes minutes, far exceeding that window. If all
    /// workers start concurrently on an empty cache, followers fail with
    /// `ApiError::LockAcquisition`. The leader-then-followers ordering avoids
    /// the contention entirely; followers load from the now-warm local cache.
    ///
    /// Idle-timeout reloads are unaffected: model files remain on disk after
    /// unload, so concurrent reloads never hit the network download path.
    pub fn spawn(
        n: usize,
        cache_dir: PathBuf,
        idle_timeout: Option<Duration>,
        onnx_batch_size: usize,
    ) -> (Self, JoinHandle<Result<()>>) {
        let capacity = n * 4;
        let (tx, rx) = mpsc::channel::<EmbedRequest>(capacity);
        let rx = Arc::new(Mutex::new(rx));

        // Readiness channel: each worker sends Ok(()) after loading models.
        // Capacity = n so senders never block.
        let (ready_tx, mut ready_rx) = mpsc::channel::<Result<()>>(n);

        let live_workers = Arc::new(AtomicUsize::new(n));
        // Start at 0; incremented as each worker successfully signals readiness.
        // This avoids a stale count if the leader or a follower fails during init.
        let loaded_workers = Arc::new(AtomicUsize::new(0));
        let live_workers_for_init = Arc::clone(&live_workers);
        let loaded_workers_for_init = Arc::clone(&loaded_workers);

        let init_handle = tokio::task::spawn(
            async move {
                let mut worker_handles = Vec::with_capacity(n);

                // Closure that spawns a single worker, eliminating duplication
                // between the leader (Phase 1) and follower (Phase 2) paths.
                let spawn_worker = |id: usize,
                                    ready_tx_clone: mpsc::Sender<Result<()>>|
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
                            idle_timeout,
                            onnx_batch_size,
                        )
                    })
                };

                // --- Phase 1: spawn leader worker (may download models) ---
                worker_handles.push(spawn_worker(0, ready_tx.clone()));

                // Await leader readiness — cache is warm after this succeeds.
                match ready_rx.recv().await {
                    Some(Ok(())) => {
                        loaded_workers_for_init.fetch_add(1, Ordering::AcqRel);
                        info!("Leader worker ready, model cache warm (1/{n})");
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

                // --- Phase 2: spawn follower workers (load from warm cache) ---
                // When n == 1, both loops below are no-ops (1..1 is empty).
                for id in 1..n {
                    worker_handles.push(spawn_worker(id, ready_tx.clone()));
                }

                // Drop our copy so recv() can detect early worker exit.
                drop(ready_tx);

                // Collect follower readiness signals (n - 1 remaining).
                for i in 1..n {
                    match ready_rx.recv().await {
                        Some(Ok(())) => {
                            loaded_workers_for_init.fetch_add(1, Ordering::AcqRel);
                            info!("Follower worker signaled ready ({}/{n})", i + 1);
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

    /// Uses an impossible cache path that causes `load_models` to fail
    /// immediately without any network access or delay.
    fn bad_cache_dir() -> PathBuf {
        PathBuf::from("/dev/null/impossible")
    }

    #[tokio::test]
    async fn spawn_propagates_leader_load_failure() {
        let (pool, init_handle) = EmbedPool::spawn(1, bad_cache_dir(), None, 8);

        let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
            .await
            .expect("init_handle should resolve quickly, not hang")
            .expect("JoinHandle should not panic");

        assert!(
            result.is_err(),
            "init should return Err on leader load failure"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed to load"),
            "error should mention load failure, got: {msg}"
        );

        // COR-1: loaded_workers must be 0, not the optimistic `n`
        assert_eq!(pool.loaded_worker_count(), 0);
    }

    #[tokio::test]
    async fn spawn_multi_worker_fails_fast_on_leader_failure() {
        let (pool, init_handle) = EmbedPool::spawn(3, bad_cache_dir(), None, 8);

        let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
            .await
            .expect("init_handle should resolve quickly, not hang")
            .expect("JoinHandle should not panic");

        assert!(
            result.is_err(),
            "init should fail without spawning followers"
        );

        // loaded_workers must still be 0 — no worker succeeded
        assert_eq!(pool.loaded_worker_count(), 0);
    }

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
