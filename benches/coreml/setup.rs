//! Shared setup for the coreml bench: corpus loader, env-driven EP/variant
//! config, model + sparse-weight loading, and a tokenize helper.

#![allow(clippy::cast_possible_truncation)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Corpus loading
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub(crate) struct Corpus {
    pub scenarios: HashMap<String, Scenario>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Scenario {
    #[allow(dead_code)]
    pub description: String,
    pub texts: Vec<String>,
}

pub(crate) fn load_corpus() -> Corpus {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/corpus.json");
    let raw = std::fs::read_to_string(&path).expect("Failed to read corpus.json");
    serde_json::from_str(&raw).expect("Failed to parse corpus.json")
}

// ---------------------------------------------------------------------------
// EP configuration from environment
// ---------------------------------------------------------------------------

pub(crate) fn ep_name() -> String {
    std::env::var("BGE_M3_BENCH_EP").unwrap_or_else(|_| "mlas_only".into())
}

pub(crate) fn cache_dir() -> PathBuf {
    PathBuf::from(std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".into()))
}

pub(crate) fn bench_model_variant() -> String {
    std::env::var("BGE_M3_MODEL").unwrap_or_else(|_| "fp32".into())
}

/// Returns the ONNX sub-batch size.
///
/// `CoreML` EPs use 8 to avoid `MLProgram` `FastPrediction` workspace OOM;
/// MLAS uses 256 (large enough to avoid chunking in practice).
pub(crate) fn onnx_batch_size() -> usize {
    if let Ok(val) = std::env::var("BGE_M3_BENCH_ONNX_BATCH") {
        return val.parse::<usize>().unwrap_or(8);
    }
    match ep_name().as_str() {
        "mlas_only" => 256,
        _ => 8,
    }
}

pub(crate) fn build_execution_providers(cache: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
    let config = ep_name();
    let coreml_cache = cache.join("coreml");

    let base = || {
        ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_specialization_strategy(ort::ep::coreml::SpecializationStrategy::FastPrediction)
            .with_model_cache_dir(coreml_cache.display().to_string())
    };

    match config.as_str() {
        "mlas_only" => vec![],
        "coreml_all" => vec![base().build()],
        "coreml_cpu_only" => vec![base()
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUOnly)
            .build()],
        "coreml_cpu_and_gpu" => vec![base()
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU)
            .build()],
        other => panic!(
            "Unknown BGE_M3_BENCH_EP={other}. \
             Use: mlas_only, coreml_all, coreml_cpu_only, coreml_cpu_and_gpu"
        ),
    }
}

// ---------------------------------------------------------------------------
// Model loading (self-contained — benchmarks can't access pub(crate) items)
// ---------------------------------------------------------------------------

const REPO_ID: &str = "BAAI/bge-m3";
const REPO_REVISION: &str = "5617a9f61b028005a4858fdac845db406aefb181";

const XENOVA_REPO_ID: &str = "Xenova/bge-m3";
/// Pinned HF commit for Xenova/bge-m3 FP16 and INT8 models.
/// Must match `XENOVA_REPO_REVISION` in `src/embedder/model_files.rs`.
const XENOVA_REPO_REVISION: &str = "4de13258303883538bd53b696b452bf8099f0858";

/// Default bench sequence length. Set `BGE_M3_MAX_SEQ_LENGTH` to override.
/// Unlike the server binary, this const is used for bench harness sizing only;
/// the bench reads the env var at runtime so it can be overridden per run.
const MAX_SEQ_LENGTH: usize = 8192;
pub(crate) const SPECIAL_TOKENS: [u32; 4] = [0, 1, 2, 3];

pub(crate) struct BenchModels {
    pub session: RefCell<ort::session::Session>,
    pub tokenizer: tokenizers::Tokenizer,
}

// NOTE(ARC-3): Model loading and embedding functions below intentionally
// duplicate logic from src/embedder/. The bench targets a separate
// criterion-driven binary, and benchmark closures need RefCell-wrapped
// sessions and `.expect()` rather than Result propagation, so a thin
// fork is cleaner than re-using the production code paths.

pub(crate) fn load_bench_models(
    cache: &Path,
    eps: Vec<ort::ep::ExecutionProviderDispatch>,
) -> BenchModels {
    load_bench_models_for_variant(cache, eps, &bench_model_variant())
}

/// Loads model files and builds a session for the given variant string.
///
/// - `"fp16"` → Xenova/bge-m3 `onnx/model_fp16.onnx`
/// - `"int8"` → Xenova/bge-m3 `onnx/model_int8.onnx`
/// - anything else → BAAI/bge-m3 `onnx/model.onnx` (FP32)
pub(crate) fn load_bench_models_for_variant(
    cache: &Path,
    eps: Vec<ort::ep::ExecutionProviderDispatch>,
    variant: &str,
) -> BenchModels {
    let (repo_id, repo_revision) = match variant {
        "fp16" | "int8" => (XENOVA_REPO_ID, XENOVA_REPO_REVISION),
        _ => (REPO_ID, REPO_REVISION),
    };

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache.to_path_buf())
        .with_progress(true)
        .build()
        .expect("Failed to build hf-hub API");

    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        repo_revision.to_string(),
    ));

    let onnx_path = match variant {
        "fp16" => repo
            .get("onnx/model_fp16.onnx")
            .expect("Failed to get model_fp16.onnx"),
        "int8" => repo
            .get("onnx/model_int8.onnx")
            .expect("Failed to get model_int8.onnx"),
        _ => {
            let path = repo
                .get("onnx/model.onnx")
                .expect("Failed to get model.onnx");
            repo.get("onnx/model.onnx_data")
                .expect("Failed to get model.onnx_data");
            repo.get("onnx/Constant_7_attr__value")
                .expect("Failed to get Constant_7");
            path
        }
    };

    let tokenizer_path = repo
        .get("tokenizer.json")
        .expect("Failed to get tokenizer.json");

    let mut tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ_LENGTH,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .expect("Failed to set truncation");
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_id: 1,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    }));

    let mut builder = ort::session::Session::builder().expect("Failed to create session builder");
    if !eps.is_empty() {
        builder = builder
            .with_execution_providers(eps)
            .expect("Failed to set EPs");
    }
    let session = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("Failed to set opt level")
        .with_intra_threads(1)
        .expect("Failed to set threads")
        .commit_from_file(&onnx_path)
        .expect("Failed to load ONNX model");

    BenchModels {
        session: RefCell::new(session),
        tokenizer,
    }
}

// ---------------------------------------------------------------------------
// Sparse weights (bundled safetensors)
// ---------------------------------------------------------------------------

pub(crate) fn load_sparse_weights() -> (ndarray::Array1<f32>, f32) {
    let data = include_bytes!("../../src/sparse_linear.safetensors");
    let tensors = safetensors::SafeTensors::deserialize(data).expect("Invalid safetensors");
    let weight_view = tensors.tensor("weight").expect("Missing weight tensor");
    let bias_view = tensors.tensor("bias").expect("Missing bias tensor");
    let weight: Vec<f32> = weight_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let bias_data = bias_view.data();
    assert_eq!(
        bias_data.len(),
        4,
        "sparse_linear bias must be a scalar F32 (4 bytes), got {} bytes",
        bias_data.len()
    );
    let bias = f32::from_le_bytes([bias_data[0], bias_data[1], bias_data[2], bias_data[3]]);
    (ndarray::Array1::from(weight), bias)
}

// ---------------------------------------------------------------------------
// Tokenization helper
// ---------------------------------------------------------------------------

pub(crate) fn tokenize_batch(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[impl AsRef<str>],
) -> (ndarray::Array2<i64>, ndarray::Array2<i64>) {
    assert!(!texts.is_empty(), "tokenize_batch requires non-empty input");
    let str_refs: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();
    let encodings = tokenizer
        .encode_batch(str_refs, true)
        .expect("Tokenization failed");
    let batch_len = encodings.len();
    let seq_len = encodings[0].get_ids().len();
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch_len * seq_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch_len * seq_len);
    for enc in &encodings {
        ids_flat.extend(enc.get_ids().iter().map(|&id| i64::from(id)));
        mask_flat.extend(enc.get_attention_mask().iter().map(|&m| i64::from(m)));
    }
    let ids = ndarray::Array2::from_shape_vec((batch_len, seq_len), ids_flat)
        .expect("input_ids shape mismatch");
    let mask = ndarray::Array2::from_shape_vec((batch_len, seq_len), mask_flat)
        .expect("attention_mask shape mismatch");
    (ids, mask)
}
