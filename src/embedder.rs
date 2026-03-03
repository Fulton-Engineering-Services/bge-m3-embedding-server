use anyhow::Result;
use ndarray::ArrayView1;
use ort::value::TensorRef;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{info, info_span, Instrument};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

pub enum EmbedRequest {
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    Sparse {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<SparseEmbedding>>>,
    },
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REPO_ID: &str = "BAAI/bge-m3";
/// Pinned HF commit — prevents silent model updates and provides supply-chain
/// integrity for the ONNX weights and tokenizer. Update this hash intentionally
/// after verifying a new revision produces equivalent embeddings.
const REPO_REVISION: &str = "5617a9f61b028005a4858fdac845db406aefb181";
const MAX_SEQ_LENGTH: usize = 512;

/// CLS, PAD, SEP/EOS, UNK — excluded from sparse output.
const SPECIAL_TOKENS: [u32; 4] = [0, 1, 2, 3];

// ---------------------------------------------------------------------------
// Model download and loading
// ---------------------------------------------------------------------------

struct ModelFiles {
    onnx_path: PathBuf,
    tokenizer_path: PathBuf,
}

/// Downloads BGE-M3 model files from Hugging Face Hub (or returns cached paths).
///
/// Files are pinned to [`REPO_REVISION`] for supply-chain integrity.
fn download_model_files(cache_dir: &Path, show_progress: bool) -> Result<ModelFiles> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(show_progress)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build hf-hub API: {e}"))?;

    let repo = api.repo(hf_hub::Repo::with_revision(
        REPO_ID.to_string(),
        hf_hub::RepoType::Model,
        REPO_REVISION.to_string(),
    ));

    let onnx_path = repo
        .get("onnx/model.onnx")
        .map_err(|e| anyhow::anyhow!("Failed to get onnx/model.onnx: {e}"))?;

    // External initializer files must be co-located with model.onnx.
    repo.get("onnx/model.onnx_data")
        .map_err(|e| anyhow::anyhow!("Failed to get onnx/model.onnx_data: {e}"))?;
    repo.get("onnx/Constant_7_attr__value")
        .map_err(|e| anyhow::anyhow!("Failed to get onnx/Constant_7_attr__value: {e}"))?;

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to get tokenizer.json: {e}"))?;

    Ok(ModelFiles {
        onnx_path,
        tokenizer_path,
    })
}

/// Loads and configures the BGE-M3 tokenizer.
///
/// Truncation: `LongestFirst` at `MAX_SEQ_LENGTH` (512).
/// Padding: `BatchLongest` with `pad_id=1`, `pad_token=<pad>`.
fn load_tokenizer(tokenizer_path: &Path) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ_LENGTH,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("Failed to set truncation: {e}"))?;

    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_id: 1,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    }));

    Ok(tokenizer)
}

/// Builds an ORT session from the ONNX model file with the given execution providers.
fn load_session(
    model_path: &Path,
    eps: Vec<ort::ep::ExecutionProviderDispatch>,
) -> Result<ort::session::Session> {
    let mut builder = ort::session::Session::builder()?;
    if !eps.is_empty() {
        builder = builder.with_execution_providers(eps)?;
    }
    let session = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;
    Ok(session)
}

// ---------------------------------------------------------------------------
// Pure math helpers (testable without ORT)
// ---------------------------------------------------------------------------

/// L2-normalizes `vec` in place. If the norm is zero, leaves the vector unchanged.
fn normalize_l2(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Projects a single token's hidden state through the sparse-linear layer.
///
/// Returns `max(0, dot(hidden, weight) + bias)` (ReLU-gated score).
fn sparse_project(hidden: &[f32], weight: &ndarray::ArrayView1<f32>, bias: f32) -> f32 {
    let hidden_view = ArrayView1::from(hidden);
    (hidden_view.dot(weight) + bias).max(0.0)
}

/// Max-pools sparse scores by vocabulary token ID, excluding special tokens
/// and tokens masked by the attention mask.
///
/// Returns sorted `(indices, values)` vectors suitable for `SparseEmbedding`.
fn sparse_maxpool(ids: &[u32], mask: &[u32], scores: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let mut token_weights: HashMap<usize, f32> = HashMap::new();

    for (j, &token_id) in ids.iter().enumerate() {
        if mask[j] == 0 {
            continue;
        }
        if SPECIAL_TOKENS.contains(&token_id) {
            continue;
        }
        let score = scores[j];
        if score > 0.0 {
            token_weights
                .entry(token_id as usize)
                .and_modify(|w| *w = w.max(score))
                .or_insert(score);
        }
    }

    let mut indices: Vec<usize> = token_weights.keys().copied().collect();
    indices.sort_unstable();
    let values: Vec<f32> = indices.iter().map(|k| token_weights[k]).collect();
    (indices, values)
}

// ---------------------------------------------------------------------------
// Shared tokenization helper
// ---------------------------------------------------------------------------

/// Tokenizes a batch of texts and returns `(input_ids, attention_mask, encodings)`.
///
/// The returned `Array2<i64>` matrices have shape `[batch, seq_len]` with padding
/// applied by `BatchLongest`. The raw `Encoding` vector is returned so callers
/// that need per-token IDs (sparse path) can iterate without re-tokenizing.
fn tokenize_to_arrays(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<(
    ndarray::Array2<i64>,
    ndarray::Array2<i64>,
    Vec<tokenizers::Encoding>,
)> {
    let str_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch(str_refs, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

    if encodings.is_empty() {
        anyhow::bail!("tokenizer returned empty batch for non-empty input");
    }

    let batch_len = encodings.len();
    let seq_len = encodings[0].get_ids().len();

    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch_len * seq_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch_len * seq_len);
    for enc in &encodings {
        ids_flat.extend(enc.get_ids().iter().map(|&id| i64::from(id)));
        mask_flat.extend(enc.get_attention_mask().iter().map(|&m| i64::from(m)));
    }

    let ids_array = ndarray::Array2::from_shape_vec((batch_len, seq_len), ids_flat)?;
    let mask_array = ndarray::Array2::from_shape_vec((batch_len, seq_len), mask_flat)?;

    Ok((ids_array, mask_array, encodings))
}

// ---------------------------------------------------------------------------
// Embedding functions
// ---------------------------------------------------------------------------

/// Produces L2-normalized dense embeddings from the `sentence_embedding` output.
///
/// The BAAI/bge-m3 model outputs `sentence_embedding` with shape `[batch, 1024]`,
/// already CLS-pooled. We only need to L2-normalize each vector.
#[allow(clippy::cast_possible_truncation)]
fn embed_dense(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(batch_size) {
        let (ids_array, mask_array, _encodings) = tokenize_to_arrays(tokenizer, chunk)?;
        let batch_len = ids_array.nrows();

        let ids_tensor = TensorRef::from_array_view(ids_array.view())?;
        let mask_tensor = TensorRef::from_array_view(mask_array.view())?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })?;

        let emb = outputs["sentence_embedding"].try_extract_array::<f32>()?;

        for i in 0..batch_len {
            let row = emb.index_axis(ndarray::Axis(0), i);
            let mut vec = row
                .as_slice()
                .expect("embedding should be contiguous")
                .to_vec();
            normalize_l2(&mut vec);
            all_embeddings.push(vec);
        }
    }

    Ok(all_embeddings)
}

/// Produces sparse embeddings via the BGE-M3 sparse-linear projection.
///
/// Reads `token_embeddings` output `[batch, seq, 1024]`, projects each token's
/// hidden state through the `sparse_linear` weight vector, applies `ReLU`, and
/// performs max-pooling across token positions sharing the same vocabulary ID.
#[allow(clippy::cast_possible_truncation)]
fn embed_sparse(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    batch_size: usize,
) -> Result<Vec<SparseEmbedding>> {
    let (weight, bias) = crate::weights::sparse_linear();
    let weight_view = weight.view();

    let mut all_sparse = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(batch_size) {
        let (ids_array, mask_array, encodings) = tokenize_to_arrays(tokenizer, chunk)?;

        let ids_tensor = TensorRef::from_array_view(ids_array.view())?;
        let mask_tensor = TensorRef::from_array_view(mask_array.view())?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })?;

        let token_emb = outputs["token_embeddings"].try_extract_array::<f32>()?;

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let batch_hidden = token_emb.index_axis(ndarray::Axis(0), i);

            // Project each token's hidden state through sparse_linear.
            let scores: Vec<f32> = (0..ids.len())
                .map(|j| {
                    let hidden = batch_hidden.index_axis(ndarray::Axis(0), j);
                    let hidden_slice = hidden
                        .as_slice()
                        .expect("hidden state should be contiguous");
                    sparse_project(hidden_slice, &weight_view, *bias)
                })
                .collect();

            let (indices, values) = sparse_maxpool(ids, mask, &scores);
            all_sparse.push(SparseEmbedding { indices, values });
        }
    }

    Ok(all_sparse)
}

// ---------------------------------------------------------------------------
// Execution provider configuration
// ---------------------------------------------------------------------------

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
/// - **`coreml-profile` feature** — when compiled with
///   `--features coreml-profile`, enables `ProfileComputePlan` which logs
///   per-op hardware dispatch decisions (GPU vs CPU vs ANE) to stderr.
///   Diagnostic only; excluded from default builds by `#[cfg]`.
///
/// On all other platforms, returns an empty vec (CPU EP only).
fn execution_providers(cache_dir: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(target_os = "macos")]
    {
        let coreml_cache = cache_dir.join("coreml");
        // ARC-6: Allow overriding the CoreML specialization strategy via env var.
        // FastPrediction pre-allocates the full intermediate-tensor workspace, which
        // can exceed available RAM on low-memory Macs. Set to "default" to fall back
        // to the CoreML default strategy.
        let strategy = match std::env::var("BGE_M3_COREML_STRATEGY").ok().as_deref() {
            Some("default") => ort::ep::coreml::SpecializationStrategy::Default,
            _ => ort::ep::coreml::SpecializationStrategy::FastPrediction,
        };
        let builder = ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_specialization_strategy(strategy)
            .with_model_cache_dir(coreml_cache.display().to_string());
        #[cfg(feature = "coreml-profile")]
        let builder = builder.with_profile_compute_plan(true);
        vec![builder.build()]
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cache_dir;
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Worker infrastructure
// ---------------------------------------------------------------------------

/// Loads both the ORT session and tokenizer from `cache_dir`.
///
/// Called at worker startup and again after an idle unload whenever a new
/// request arrives. Both session and tokenizer are always loaded and unloaded
/// as a pair.
fn load_models(
    cache_dir: &Path,
    show_download_progress: bool,
) -> Result<(ort::session::Session, tokenizers::Tokenizer)> {
    let files = download_model_files(cache_dir, show_download_progress)?;
    let tokenizer = load_tokenizer(&files.tokenizer_path)?;
    let eps = execution_providers(cache_dir);
    let session = load_session(&files.onnx_path, eps)?;
    Ok((session, tokenizer))
}

/// RAII guard that decrements the live-worker counter when dropped.
/// Guarantees decrement fires on clean exit AND on panic unwind.
struct WorkerGuard(Arc<AtomicUsize>);

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
/// Groups the policy arguments that were previously passed as individual
/// positional parameters to `run_worker`, keeping the function signature
/// manageable as configuration options grow.
#[derive(Clone)]
pub(crate) struct WorkerConfig {
    /// Maximum texts per ONNX `session.run()` call.
    pub onnx_batch_size: usize,
    /// Duration of inactivity before workers unload their model instances.
    pub idle_timeout: Option<Duration>,
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
    config: WorkerConfig,
) -> Result<()> {
    let _guard = WorkerGuard(Arc::clone(&live_workers));
    let span = info_span!("worker", id = id);
    let _span_guard = span.enter();

    info!("Loading models (worker {id})...");
    let load_start = std::time::Instant::now();
    let rt = Handle::current();
    let initial_models = match load_models(&cache_dir, id == 0) {
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

    let mut models: Option<(ort::session::Session, tokenizers::Tokenizer)> = Some(initial_models);

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
        let msg = if let Some(timeout) = config.idle_timeout.filter(|_| models.is_some()) {
            rt.block_on(async {
                tokio::time::timeout(timeout, async { rx.lock().await.recv().await }).await
            })
        } else {
            rt.block_on(async { Ok(rx.lock().await.recv().await) })
        };

        match msg {
            Err(_elapsed) => {
                // Idle timeout fired while models were loaded — unload them.
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
                // Reload models if they were unloaded due to idle timeout.
                // This blocks the current request until reload completes (~10-30 s).
                if models.is_none() {
                    tracing::info!("Worker {id} reloading models after idle...");
                    let reload_start = std::time::Instant::now();
                    match load_models(&cache_dir, false) {
                        Ok(m) => {
                            models = Some(m);
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
                let (session, tokenizer) =
                    models.as_mut().expect("models loaded after reload check");

                match request {
                    EmbedRequest::Dense { texts, reply } => {
                        let result =
                            embed_dense(session, tokenizer, &texts, config.onnx_batch_size)
                                .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Sparse { texts, reply } => {
                        let result =
                            embed_sparse(session, tokenizer, &texts, config.onnx_batch_size)
                                .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                        let _ = reply.send(result);
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// EmbedPool
// ---------------------------------------------------------------------------

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
    /// `config` — execution policy shared by all workers (batch size, idle timeout).
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
        config: WorkerConfig,
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
                    let worker_config = config.clone();
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

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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
    pub(crate) fn with_fixed_responses(
        dense_fixture: Vec<Vec<f32>>,
        sparse_fixture: Vec<SparseEmbedding>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<EmbedRequest>(8);
        let dense = Arc::new(dense_fixture);
        let sparse = Arc::new(sparse_fixture);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                match req {
                    EmbedRequest::Dense { reply, .. } => {
                        let _ = reply.send(Ok((*dense).clone()));
                    }
                    EmbedRequest::Sparse { reply, .. } => {
                        let _ = reply.send(Ok((*sparse).clone()));
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let (pool, init_handle) = EmbedPool::spawn(
            1,
            bad_cache_dir(),
            WorkerConfig {
                onnx_batch_size: 8,
                idle_timeout: None,
            },
        );

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
        let (pool, init_handle) = EmbedPool::spawn(
            3,
            bad_cache_dir(),
            WorkerConfig {
                onnx_batch_size: 8,
                idle_timeout: None,
            },
        );

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
        let err = result.expect_err("expected an error");
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

    // -----------------------------------------------------------------------
    // Pure-math helper tests (no ORT session needed)
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_l2_unit_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_l2(&mut v);
        let expected_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((expected_norm - 1.0).abs() < 1e-6, "should be unit length");
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize_l2(&mut v);
        assert!(v.iter().all(|&x| x == 0.0), "zero vector should stay zero");
    }

    #[test]
    fn normalize_l2_already_unit() {
        let mut v = vec![1.0, 0.0, 0.0];
        normalize_l2(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!(v[2].abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_sign_preservation() {
        let mut v = vec![-3.0, 4.0];
        normalize_l2(&mut v);
        assert!(
            (v[0] - (-0.6)).abs() < 1e-6,
            "negative sign must be preserved"
        );
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_single_element() {
        let mut v = vec![5.0];
        normalize_l2(&mut v);
        assert!(
            (v[0] - 1.0).abs() < 1e-6,
            "single positive element normalizes to 1.0"
        );

        let mut v2 = vec![-7.0];
        normalize_l2(&mut v2);
        assert!(
            (v2[0] - (-1.0)).abs() < 1e-6,
            "single negative element normalizes to -1.0"
        );
    }

    #[test]
    fn normalize_l2_output_norm_is_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize_l2(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "output norm must equal 1.0, got {norm}"
        );
    }

    #[test]
    fn sparse_project_positive_score() {
        let weight = ndarray::array![1.0, 2.0, 3.0];
        let hidden = [1.0, 1.0, 1.0];
        // dot = 1+2+3 = 6, + bias 0.5 = 6.5, ReLU = 6.5
        let score = sparse_project(&hidden, &weight.view(), 0.5);
        assert!((score - 6.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_project_relu_clamps_negative() {
        let weight = ndarray::array![1.0, 1.0];
        let hidden = [-5.0, -5.0];
        // dot = -10, + bias 0.0 = -10, ReLU = 0
        let score = sparse_project(&hidden, &weight.view(), 0.0);
        assert!(
            score.abs() < 1e-6,
            "negative scores should be clamped to zero"
        );
    }

    #[test]
    fn sparse_project_zero_weight() {
        let weight = ndarray::array![0.0, 0.0, 0.0];
        let hidden = [1.0, 2.0, 3.0];
        // dot = 0, + bias 1.0 = 1.0, ReLU = 1.0
        let score = sparse_project(&hidden, &weight.view(), 1.0);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "zero weights with positive bias"
        );
    }

    #[test]
    fn sparse_project_negative_bias() {
        let weight = ndarray::array![1.0, 1.0];
        let hidden = [1.0, 1.0];
        // dot = 2, + bias -3.0 = -1.0, ReLU = 0.0
        let score = sparse_project(&hidden, &weight.view(), -3.0);
        assert!(score.abs() < 1e-6, "negative bias should clamp via ReLU");
    }

    #[test]
    fn sparse_maxpool_all_masked_out() {
        // All non-special tokens have mask=0 (padding) → empty output
        let ids = [100, 200, 300];
        let mask = [0, 0, 0];
        let scores = [0.5, 0.8, 0.3];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert!(indices.is_empty(), "all-masked should produce empty output");
        assert!(values.is_empty());
    }

    #[test]
    fn sparse_maxpool_basic() {
        // token_id 10 appears twice with scores 0.3, 0.7 → max = 0.7
        // token_id 20 appears once  with score  0.5     → 0.5
        let ids = [10, 20, 10];
        let mask = [1, 1, 1];
        let scores = [0.3, 0.5, 0.7];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![10, 20]); // sorted by ID
        assert!((values[0] - 0.7).abs() < 1e-6); // max(0.3, 0.7)
        assert!((values[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_maxpool_filters_special_tokens() {
        // IDs 0, 1, 2, 3 are SPECIAL_TOKENS — should be excluded
        let ids = [0, 1, 2, 3, 100];
        let mask = [1, 1, 1, 1, 1];
        let scores = [0.9, 0.9, 0.9, 0.9, 0.5];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![100]);
        assert!((values[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_maxpool_respects_attention_mask() {
        // mask=0 means the token is padding → excluded
        let ids = [100, 200];
        let mask = [1, 0];
        let scores = [0.5, 0.9];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![100]);
        assert!((values[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_maxpool_skips_zero_scores() {
        let ids = [100, 200];
        let mask = [1, 1];
        let scores = [0.0, 0.5];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![200], "zero-score tokens should be excluded");
        assert!((values[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_maxpool_empty_input() {
        let ids: [u32; 0] = [];
        let mask: [u32; 0] = [];
        let scores: [f32; 0] = [];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn sparse_maxpool_returns_sorted_indices() {
        let ids = [300, 100, 200];
        let mask = [1, 1, 1];
        let scores = [0.1, 0.2, 0.3];
        let (indices, _) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![100, 200, 300], "indices should be sorted");
    }

    // -----------------------------------------------------------------------
    // REPO_REVISION drift detection (ARC-3)
    // -----------------------------------------------------------------------

    /// Extracts the `REPO_REVISION` constant value from a source file by reading
    /// it as text and finding the `const REPO_REVISION: &str = "...";` line.
    fn extract_repo_revision(path: &str) -> String {
        let content =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("const REPO_REVISION") {
                // Extract the quoted value between the first pair of double quotes
                let start = trimmed.find('"').expect("missing opening quote");
                let end = trimmed[start + 1..]
                    .find('"')
                    .expect("missing closing quote");
                return trimmed[start + 1..start + 1 + end].to_string();
            }
        }
        panic!("REPO_REVISION not found in {path}");
    }

    #[test]
    fn repo_revision_consistent_across_all_copies() {
        let embedder = extract_repo_revision("src/embedder.rs");
        let bench = extract_repo_revision("benches/coreml.rs");
        let example = extract_repo_revision("examples/fp16_eval.rs");

        assert_eq!(
            embedder, bench,
            "REPO_REVISION mismatch: src/embedder.rs ({embedder}) != benches/coreml.rs ({bench})"
        );
        assert_eq!(
            embedder, example,
            "REPO_REVISION mismatch: src/embedder.rs ({embedder}) != examples/fp16_eval.rs ({example})"
        );
        // Sanity: should look like a git commit SHA
        assert_eq!(embedder.len(), 40, "REPO_REVISION should be a 40-char SHA");
        assert!(
            embedder.chars().all(|c| c.is_ascii_hexdigit()),
            "REPO_REVISION should be hexadecimal"
        );
    }

    // -----------------------------------------------------------------------
    // Benchmark corpus shape validation (TST-5)
    // -----------------------------------------------------------------------

    #[test]
    fn benchmark_corpus_has_expected_shape() {
        let content = std::fs::read_to_string("benches/fixtures/corpus.json")
            .expect("corpus.json must be readable from project root");
        let corpus: serde_json::Value =
            serde_json::from_str(&content).expect("corpus.json must be valid JSON");

        // Verify top-level structure
        assert!(corpus.get("metadata").is_some(), "must have 'metadata' key");
        assert!(
            corpus.get("scenarios").is_some(),
            "must have 'scenarios' key"
        );

        // Verify metadata.sources counts
        let sources = &corpus["metadata"]["sources"];
        assert_eq!(sources["knowledgebase_chunks"]["count"], 50);
        assert_eq!(sources["coordinator_vector_store"]["count"], 75);
        assert_eq!(sources["codekeeper_symbols"]["count"], 50);
        assert_eq!(sources["boundary_cases"]["count"], 9);

        // Verify scenarios have matching text counts
        let scenarios = corpus["scenarios"]
            .as_object()
            .expect("scenarios must be object");
        let expected: &[(&str, usize)] = &[
            ("document_chunks", 50),
            ("tool_descriptions", 75),
            ("code_symbols", 50),
            ("boundary_cases", 9),
        ];
        for &(name, count) in expected {
            let texts = scenarios
                .get(name)
                .and_then(|s| s.get("texts"))
                .and_then(|t| t.as_array())
                .unwrap_or_else(|| panic!("scenarios.{name}.texts must be an array"));
            assert_eq!(
                texts.len(),
                count,
                "scenarios.{name} should have {count} texts, got {}",
                texts.len()
            );
        }

        // Total texts = 184
        let total: usize = scenarios
            .values()
            .filter_map(|s| s.get("texts").and_then(|t| t.as_array()).map(Vec::len))
            .sum();
        assert_eq!(total, 184, "total corpus texts should be 184");
    }
}
