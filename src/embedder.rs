use crate::binpack::{bin_pack, CostModel};
use crate::config::ModelVariant;
use crate::sysinfo;
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

/// OS headroom reserved for kernel, stack, ORT arena, and other non-model
/// allocations. Subtracted from available memory before computing
/// per-worker workspace.
pub(crate) const OS_HEADROOM_BYTES: usize = 256 * 1024 * 1024; // 256 MiB

pub enum EmbedRequest {
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    Sparse {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<SparseEmbedding>>>,
    },
    /// Internal: used during startup probe to run a single batch and measure
    /// peak RSS delta. Workers only process this before `ready` is set.
    Probe {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<ProbeResult>>,
    },
}

/// Result of a single probe `session.run()` call.
pub(crate) struct ProbeResult {
    pub rss_before: usize,
    pub rss_after: usize,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REPO_ID: &str = "BAAI/bge-m3";
/// Pinned HF commit — prevents silent model updates and provides supply-chain
/// integrity for the ONNX weights and tokenizer. Update this hash intentionally
/// after verifying a new revision produces equivalent embeddings.
const REPO_REVISION: &str = "5617a9f61b028005a4858fdac845db406aefb181";

const XENOVA_REPO_ID: &str = "Xenova/bge-m3";
/// Pinned HF commit for the Xenova/bge-m3 FP16 (~1.08 GB) and INT8 (~568 MB) models.
/// Update intentionally after verifying equivalent embedding quality vs FP32.
const XENOVA_REPO_REVISION: &str = "4de13258303883538bd53b696b452bf8099f0858";

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
fn download_model_files(
    cache_dir: &Path,
    show_progress: bool,
    variant: ModelVariant,
) -> Result<ModelFiles> {
    let (repo_id, repo_revision) = match variant {
        ModelVariant::Fp32 => (REPO_ID, REPO_REVISION),
        ModelVariant::Fp16 | ModelVariant::Int8 => (XENOVA_REPO_ID, XENOVA_REPO_REVISION),
    };

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(show_progress)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build hf-hub API: {e}"))?;

    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        repo_revision.to_string(),
    ));

    let onnx_path = match variant {
        ModelVariant::Fp32 => {
            let path = repo
                .get("onnx/model.onnx")
                .map_err(|e| anyhow::anyhow!("Failed to get onnx/model.onnx: {e}"))?;
            repo.get("onnx/model.onnx_data")
                .map_err(|e| anyhow::anyhow!("Failed to get onnx/model.onnx_data: {e}"))?;
            repo.get("onnx/Constant_7_attr__value")
                .map_err(|e| anyhow::anyhow!("Failed to get onnx/Constant_7_attr__value: {e}"))?;
            path
        }
        ModelVariant::Fp16 => repo
            .get("onnx/model_fp16.onnx")
            .map_err(|e| anyhow::anyhow!("Failed to get onnx/model_fp16.onnx: {e}"))?,
        ModelVariant::Int8 => repo
            .get("onnx/model_int8.onnx")
            .map_err(|e| anyhow::anyhow!("Failed to get onnx/model_int8.onnx: {e}"))?,
    };

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to get tokenizer.json: {e}"))?;

    Ok(ModelFiles { onnx_path, tokenizer_path })
}

/// Loads and configures the BGE-M3 tokenizer with truncation at `max_seq_length`
/// but **no** padding. Padding is applied per-chunk in [`build_chunk_arrays`].
fn load_tokenizer(
    tokenizer_path: &Path,
    max_seq_length: usize,
) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_seq_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("Failed to set truncation: {e}"))?;

    // No BatchLongest padding here — we pad manually in build_chunk_arrays
    // so each chunk only pads to its own longest sequence.
    tokenizer.with_padding(None);

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
// Tokenization helpers (no-pad path)
// ---------------------------------------------------------------------------

/// Tokenizes `texts` without applying any padding. Returns one `Encoding` per text,
/// each truncated to the tokenizer's configured `max_length`.
fn tokenize_no_pad(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<Vec<tokenizers::Encoding>> {
    let str_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch_fast(str_refs, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    Ok(encodings)
}

/// Builds `input_ids` and `attention_mask` arrays for a single chunk.
///
/// `indices` selects which encodings from `all_encodings` belong to this chunk.
/// `pad_to` is the chunk-local maximum sequence length; all sequences are
/// right-padded with `pad_id = 1` (XLM-RoBERTa `<pad>` token).
#[allow(clippy::cast_possible_truncation)]
fn build_chunk_arrays(
    all_encodings: &[tokenizers::Encoding],
    indices: &[usize],
    pad_to: usize,
) -> Result<(ndarray::Array2<i64>, ndarray::Array2<i64>)> {
    let batch = indices.len();
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * pad_to);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * pad_to);

    for &idx in indices {
        let enc = &all_encodings[idx];
        let token_ids = enc.get_ids();
        let attn_mask = enc.get_attention_mask();
        let seq_len = token_ids.len();

        // Copy token ids and mask
        ids_flat.extend(token_ids.iter().map(|&id| i64::from(id)));
        mask_flat.extend(attn_mask.iter().map(|&m| i64::from(m)));

        // Right-pad with pad_id=1 / mask=0
        let pad = pad_to.saturating_sub(seq_len);
        ids_flat.extend(std::iter::repeat_n(1i64, pad));
        mask_flat.extend(std::iter::repeat_n(0i64, pad));
    }

    let ids_array = ndarray::Array2::from_shape_vec((batch, pad_to), ids_flat)?;
    let mask_array = ndarray::Array2::from_shape_vec((batch, pad_to), mask_flat)?;

    Ok((ids_array, mask_array))
}

// ---------------------------------------------------------------------------
// Embedding functions
// ---------------------------------------------------------------------------

/// Produces L2-normalized dense embeddings.
///
/// Tokenizes once, then uses the cost model to bin-pack into chunks that fit
/// within the workspace budget. Results are scattered back to the original
/// input order.
#[allow(clippy::cast_possible_truncation)]
fn embed_dense(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    cost_model: &CostModel,
    model_variant: ModelVariant,
) -> Result<Vec<Vec<f32>>> {
    let encodings = tokenize_no_pad(tokenizer, texts)?;
    let seq_lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();

    let chunks = bin_pack(&seq_lens, cost_model);

    // Pre-allocate output slots (one per input text, filled below).
    let mut all_embeddings: Vec<Vec<f32>> = (0..texts.len()).map(|_| Vec::new()).collect();

    for chunk_indices in &chunks {
        let chunk_max = chunk_indices
            .iter()
            .map(|&i| seq_lens[i])
            .max()
            .unwrap_or(1)
            .max(1); // guard: at least 1 to avoid 0-dim tensors

        let (ids_array, mask_array) = build_chunk_arrays(&encodings, chunk_indices, chunk_max)?;
        let batch_len = ids_array.nrows();

        let ids_tensor = TensorRef::from_array_view(ids_array.view())?;
        let mask_tensor = TensorRef::from_array_view(mask_array.view())?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })?;

        // FP32: sentence_embedding [batch, 1024] — pre-pooled CLS output.
        // FP16/INT8: last_hidden_state [batch, seq, 1024] — CLS token at position 0.
        let emb: ndarray::ArrayD<f32> = match model_variant {
            ModelVariant::Fp32 => outputs["sentence_embedding"]
                .try_extract_array::<f32>()?
                .to_owned(),
            ModelVariant::Fp16 | ModelVariant::Int8 => {
                let lhs = outputs["last_hidden_state"].try_extract_array::<f32>()?;
                lhs.index_axis(ndarray::Axis(1), 0).to_owned()
            }
        };

        for (chunk_pos, &orig_idx) in chunk_indices.iter().enumerate() {
            debug_assert!(chunk_pos < batch_len, "chunk_pos must be within batch");
            let row = emb.index_axis(ndarray::Axis(0), chunk_pos);
            let mut vec = row
                .as_slice()
                .expect("embedding should be contiguous")
                .to_vec();
            normalize_l2(&mut vec);
            all_embeddings[orig_idx] = vec;
        }
    }

    Ok(all_embeddings)
}

/// Produces sparse embeddings via the BGE-M3 sparse-linear projection.
///
/// Tokenizes once, then uses the cost model to bin-pack into chunks. Results
/// are scattered back to the original input order.
#[allow(clippy::cast_possible_truncation)]
fn embed_sparse(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    cost_model: &CostModel,
    model_variant: ModelVariant,
) -> Result<Vec<SparseEmbedding>> {
    let (weight, bias) = crate::weights::sparse_linear();
    let weight_view = weight.view();

    let encodings = tokenize_no_pad(tokenizer, texts)?;
    let seq_lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();

    let chunks = bin_pack(&seq_lens, cost_model);

    let mut all_sparse: Vec<Option<SparseEmbedding>> = (0..texts.len()).map(|_| None).collect();

    for chunk_indices in &chunks {
        let chunk_max = chunk_indices
            .iter()
            .map(|&i| seq_lens[i])
            .max()
            .unwrap_or(1)
            .max(1);

        let (ids_array, mask_array) = build_chunk_arrays(&encodings, chunk_indices, chunk_max)?;

        let ids_tensor = TensorRef::from_array_view(ids_array.view())?;
        let mask_tensor = TensorRef::from_array_view(mask_array.view())?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })?;

        // FP32: token_embeddings [batch, seq, 1024].
        // FP16/INT8: last_hidden_state [batch, seq, 1024] — same shape, different key.
        let token_emb = match model_variant {
            ModelVariant::Fp32 => outputs["token_embeddings"].try_extract_array::<f32>()?,
            ModelVariant::Fp16 | ModelVariant::Int8 => {
                outputs["last_hidden_state"].try_extract_array::<f32>()?
            }
        };

        for (chunk_pos, &orig_idx) in chunk_indices.iter().enumerate() {
            let enc = &encodings[orig_idx];
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let batch_hidden = token_emb.index_axis(ndarray::Axis(0), chunk_pos);

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
            all_sparse[orig_idx] = Some(SparseEmbedding { indices, values });
        }
    }

    Ok(all_sparse
        .into_iter()
        .map(|s| s.expect("every slot must be filled"))
        .collect())
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

    let ids_tensor = TensorRef::from_array_view(ids_array.view())?;
    let mask_tensor = TensorRef::from_array_view(mask_array.view())?;

    // Run inference (output discarded — we only care about RSS).
    let _outputs = session.run(ort::inputs! {
        "input_ids" => ids_tensor,
        "attention_mask" => mask_tensor,
    })?;

    let rss_after = sysinfo::read_process_rss_bytes().unwrap_or(rss_before);

    Ok(ProbeResult { rss_before, rss_after })
}

// ---------------------------------------------------------------------------
// Execution provider configuration
// ---------------------------------------------------------------------------

fn execution_providers(cache_dir: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(target_os = "macos")]
    {
        let coreml_cache = cache_dir.join("coreml");
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

fn load_models(
    cache_dir: &Path,
    show_download_progress: bool,
    model_variant: ModelVariant,
    max_seq_length: usize,
) -> Result<(ort::session::Session, tokenizers::Tokenizer)> {
    let files = download_model_files(cache_dir, show_download_progress, model_variant)?;
    let tokenizer = load_tokenizer(&files.tokenizer_path, max_seq_length)?;
    let eps = execution_providers(cache_dir);
    let session = load_session(&files.onnx_path, eps)?;
    Ok((session, tokenizer))
}

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
#[derive(Clone)]
pub(crate) struct WorkerConfig {
    /// Quadratic-aware workspace cost model and per-worker budget.
    pub cost_model: CostModel,
    /// Duration of inactivity before workers unload their model instances.
    pub idle_timeout: Option<Duration>,
    /// ONNX model variant to load (FP32, FP16, or INT8).
    pub model_variant: ModelVariant,
    /// Maximum tokenized sequence length.
    pub max_seq_length: usize,
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
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
    let initial_models = match load_models(
        &cache_dir,
        id == 0,
        config.model_variant,
        config.max_seq_length,
    ) {
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
                    ) {
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
                            let err = anyhow::anyhow!("Model reload failed: {e}");
                            match request {
                                EmbedRequest::Dense { reply, .. } => {
                                    let _ = reply.send(Err(err));
                                }
                                EmbedRequest::Sparse { reply, .. } => {
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
                        let result = embed_dense(
                            session,
                            tokenizer,
                            &texts,
                            &config.cost_model,
                            config.model_variant,
                        )
                        .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                        let _ = reply.send(result);
                    }
                    EmbedRequest::Sparse { texts, reply } => {
                        let result = embed_sparse(
                            session,
                            tokenizer,
                            &texts,
                            &config.cost_model,
                            config.model_variant,
                        )
                        .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
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

// ---------------------------------------------------------------------------
// EmbedPool
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct EmbedPool {
    tx: mpsc::Sender<EmbedRequest>,
    live_workers: Arc<AtomicUsize>,
    /// Number of workers that currently have model instances loaded in memory.
    loaded_workers: Arc<AtomicUsize>,
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

        let (ready_tx, mut ready_rx) = mpsc::channel::<Result<()>>(n);

        let live_workers = Arc::new(AtomicUsize::new(n));
        let loaded_workers = Arc::new(AtomicUsize::new(0));
        let live_workers_for_init = Arc::clone(&live_workers);
        let loaded_workers_for_init = Arc::clone(&loaded_workers);

        let init_handle = tokio::task::spawn(
            async move {
                let mut worker_handles = Vec::with_capacity(n);

                let spawn_worker = |id: usize,
                                    ready_tx_clone: mpsc::Sender<Result<()>>,
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

                // --- Phase 1: spawn leader worker (may download models) ---
                worker_handles.push(spawn_worker(0, ready_tx.clone(), config.clone()));

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
                for id in 1..n {
                    worker_handles.push(spawn_worker(id, ready_tx.clone(), config.clone()));
                }

                drop(ready_tx);

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

                drop(worker_handles);

                Ok(())
            }
            .instrument(info_span!("embed_pool")),
        );

        (
            Self { tx, live_workers, loaded_workers },
            init_handle,
        )
    }

    pub async fn dense(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Dense { texts, reply: reply_tx })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }

    pub async fn sparse(&self, texts: Vec<String>) -> Result<Vec<SparseEmbedding>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Sparse { texts, reply: reply_tx })
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
            .send(EmbedRequest::Probe { texts, reply: reply_tx })
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
}

// ---------------------------------------------------------------------------
// Test helpers
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
        }
    }

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
        }
    }

    pub(crate) fn idle_for_test() -> Self {
        let (tx, _rx) = mpsc::channel::<EmbedRequest>(1);
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
    use std::time::Duration;

    fn bad_cache_dir() -> PathBuf {
        PathBuf::from("/dev/null/impossible")
    }

    fn test_cost_model() -> CostModel {
        CostModel::conservative(CostModel::DEFAULT_MAX_WORKSPACE)
    }

    #[tokio::test]
    async fn spawn_propagates_leader_load_failure() {
        let (pool, init_handle) = EmbedPool::spawn(
            1,
            bad_cache_dir(),
            WorkerConfig {
                cost_model: test_cost_model(),
                idle_timeout: None,
                model_variant: crate::config::ModelVariant::Fp32,
                max_seq_length: 512,
            },
        );

        let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
            .await
            .expect("init_handle should resolve quickly, not hang")
            .expect("JoinHandle should not panic");

        assert!(result.is_err(), "init should return Err on leader load failure");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed to load"),
            "error should mention load failure, got: {msg}"
        );

        assert_eq!(pool.loaded_worker_count(), 0);
    }

    #[tokio::test]
    async fn spawn_multi_worker_fails_fast_on_leader_failure() {
        let (pool, init_handle) = EmbedPool::spawn(
            3,
            bad_cache_dir(),
            WorkerConfig {
                cost_model: test_cost_model(),
                idle_timeout: None,
                model_variant: crate::config::ModelVariant::Fp32,
                max_seq_length: 512,
            },
        );

        let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
            .await
            .expect("init_handle should resolve quickly, not hang")
            .expect("JoinHandle should not panic");

        assert!(result.is_err(), "init should fail without spawning followers");
        assert_eq!(pool.loaded_worker_count(), 0);
    }

    #[tokio::test]
    async fn dense_returns_error_when_channel_closed() {
        let pool = EmbedPool::closed_for_test();
        let result = pool.dense(vec!["hello".into()]).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("channel closed"));
    }

    #[tokio::test]
    async fn sparse_returns_error_when_channel_closed() {
        let pool = EmbedPool::closed_for_test();
        let result = pool.sparse(vec!["hello".into()]).await;
        let err = result.expect_err("expected an error");
        assert!(err.to_string().contains("channel closed"));
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
        assert!((v[0] - (-0.6)).abs() < 1e-6, "negative sign must be preserved");
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_single_element() {
        let mut v = vec![5.0];
        normalize_l2(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);

        let mut v2 = vec![-7.0];
        normalize_l2(&mut v2);
        assert!((v2[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_output_norm_is_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize_l2(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "output norm must equal 1.0, got {norm}");
    }

    #[test]
    fn sparse_project_positive_score() {
        let weight = ndarray::array![1.0, 2.0, 3.0];
        let hidden = [1.0, 1.0, 1.0];
        let score = sparse_project(&hidden, &weight.view(), 0.5);
        assert!((score - 6.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_project_relu_clamps_negative() {
        let weight = ndarray::array![1.0, 1.0];
        let hidden = [-5.0, -5.0];
        let score = sparse_project(&hidden, &weight.view(), 0.0);
        assert!(score.abs() < 1e-6, "negative scores should be clamped to zero");
    }

    #[test]
    fn sparse_project_zero_weight() {
        let weight = ndarray::array![0.0, 0.0, 0.0];
        let hidden = [1.0, 2.0, 3.0];
        let score = sparse_project(&hidden, &weight.view(), 1.0);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_project_negative_bias() {
        let weight = ndarray::array![1.0, 1.0];
        let hidden = [1.0, 1.0];
        let score = sparse_project(&hidden, &weight.view(), -3.0);
        assert!(score.abs() < 1e-6, "negative bias should clamp via ReLU");
    }

    #[test]
    fn sparse_maxpool_all_masked_out() {
        let ids = [100, 200, 300];
        let mask = [0, 0, 0];
        let scores = [0.5, 0.8, 0.3];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn sparse_maxpool_basic() {
        let ids = [10, 20, 10];
        let mask = [1, 1, 1];
        let scores = [0.3, 0.5, 0.7];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![10, 20]);
        assert!((values[0] - 0.7).abs() < 1e-6);
        assert!((values[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_maxpool_filters_special_tokens() {
        let ids = [0, 1, 2, 3, 100];
        let mask = [1, 1, 1, 1, 1];
        let scores = [0.9, 0.9, 0.9, 0.9, 0.5];
        let (indices, values) = sparse_maxpool(&ids, &mask, &scores);
        assert_eq!(indices, vec![100]);
        assert!((values[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sparse_maxpool_respects_attention_mask() {
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
        assert_eq!(indices, vec![200]);
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
        assert_eq!(indices, vec![100, 200, 300]);
    }

    // -----------------------------------------------------------------------
    // REPO_REVISION drift detection (ARC-3)
    // -----------------------------------------------------------------------

    fn extract_const_str(path: &str, const_name: &str) -> String {
        let prefix = format!("const {const_name}");
        let content =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(&prefix) {
                let start = trimmed.find('"').expect("missing opening quote");
                let end = trimmed[start + 1..]
                    .find('"')
                    .expect("missing closing quote");
                return trimmed[start + 1..start + 1 + end].to_string();
            }
        }
        panic!("{const_name} not found in {path}");
    }

    #[test]
    fn repo_revision_consistent_across_all_copies() {
        let embedder = extract_const_str("src/embedder.rs", "REPO_REVISION");
        let bench = extract_const_str("benches/coreml.rs", "REPO_REVISION");
        let example = extract_const_str("examples/fp16_eval.rs", "REPO_REVISION");

        assert_eq!(
            embedder, bench,
            "REPO_REVISION mismatch: src/embedder.rs ({embedder}) != benches/coreml.rs ({bench})"
        );
        assert_eq!(
            embedder, example,
            "REPO_REVISION mismatch: src/embedder.rs ({embedder}) != examples/fp16_eval.rs ({example})"
        );
        assert_eq!(embedder.len(), 40, "REPO_REVISION should be a 40-char SHA");
        assert!(
            embedder.chars().all(|c| c.is_ascii_hexdigit()),
            "REPO_REVISION should be hexadecimal"
        );
    }

    #[test]
    fn xenova_repo_revision_consistent_across_all_copies() {
        let embedder = extract_const_str("src/embedder.rs", "XENOVA_REPO_REVISION");
        let bench = extract_const_str("benches/coreml.rs", "XENOVA_REPO_REVISION");

        assert_eq!(
            embedder, bench,
            "XENOVA_REPO_REVISION mismatch: \
             src/embedder.rs ({embedder}) != benches/coreml.rs ({bench})"
        );
        assert_eq!(embedder.len(), 40);
        assert!(embedder.chars().all(|c| c.is_ascii_hexdigit()));
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

        assert!(corpus.get("metadata").is_some(), "must have 'metadata' key");
        assert!(corpus.get("scenarios").is_some(), "must have 'scenarios' key");

        let sources = &corpus["metadata"]["sources"];
        assert_eq!(sources["knowledgebase_chunks"]["count"], 50);
        assert_eq!(sources["coordinator_vector_store"]["count"], 75);
        assert_eq!(sources["codekeeper_symbols"]["count"], 50);
        assert_eq!(sources["boundary_cases"]["count"], 9);

        let scenarios = corpus["scenarios"].as_object().expect("scenarios must be object");
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
            assert_eq!(texts.len(), count);
        }

        let total: usize = scenarios
            .values()
            .filter_map(|s| s.get("texts").and_then(|t| t.as_array()).map(Vec::len))
            .sum();
        assert_eq!(total, 184, "total corpus texts should be 184");
    }
}
