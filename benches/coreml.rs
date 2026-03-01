// CoreML EP benchmark harness for comparing execution provider configurations.
//
// TODO(CI): These benchmarks require a full ORT build and model download, so they
// cannot run in GitHub Actions CI (which lacks ORT_LIB_LOCATION and Apple Silicon).
// End-to-end inference tests (embed_dense/embed_sparse with a real model) are also
// missing from the unit test suite for the same reason. Consider adding a CI job on
// a self-hosted Apple Silicon runner, or a lightweight integration test that mocks
// the ORT session to validate tokenization + post-processing without model weights.
//
// Measures dense and sparse embedding inference at the ORT level,
// bypassing the HTTP server and worker pool to isolate ONNX Runtime performance.
//
// Configuration via environment variables:
//
//   BGE_M3_BENCH_EP       Execution provider config (default: mlas_only)
//                         Values: mlas_only, coreml_all, coreml_cpu_only, coreml_cpu_and_gpu
//
//   BGE_M3_CACHE_DIR      Model cache directory (default: /tmp/bge-m3-cache)
//
// Usage:
//
//   # Baseline (MLAS NEON only, no CoreML)
//   cargo bench --bench coreml -- --save-baseline mlas_only
//
//   # CoreML with Accelerate/AMX (CPU only)
//   BGE_M3_BENCH_EP=coreml_cpu_only cargo bench --bench coreml -- --baseline mlas_only
//
//   # CoreML with GPU + CPU dispatch
//   BGE_M3_BENCH_EP=coreml_all cargo bench --bench coreml -- --baseline mlas_only
//
// Requires ORT_LIB_LOCATION at build time for custom ORT with CoreML EP.

#![allow(clippy::cast_possible_truncation)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::ArrayView1;
use ort::value::TensorRef;

// ---------------------------------------------------------------------------
// Corpus loading
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Corpus {
    scenarios: HashMap<String, Scenario>,
}

#[derive(serde::Deserialize)]
struct Scenario {
    #[allow(dead_code)]
    description: String,
    texts: Vec<String>,
}

fn load_corpus() -> Corpus {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/corpus.json");
    let raw = std::fs::read_to_string(&path).expect("Failed to read corpus.json");
    serde_json::from_str(&raw).expect("Failed to parse corpus.json")
}

// ---------------------------------------------------------------------------
// EP configuration from environment
// ---------------------------------------------------------------------------

fn ep_name() -> String {
    std::env::var("BGE_M3_BENCH_EP").unwrap_or_else(|_| "mlas_only".into())
}

fn cache_dir() -> PathBuf {
    PathBuf::from(std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".into()))
}

/// Returns the ONNX sub-batch size.
///
/// `CoreML` EPs use 8 to avoid `MLProgram` `FastPrediction` workspace OOM;
/// MLAS uses 256 (large enough to avoid chunking in practice).
fn onnx_batch_size() -> usize {
    if let Ok(val) = std::env::var("BGE_M3_BENCH_ONNX_BATCH") {
        return val.parse::<usize>().unwrap_or(8);
    }
    match ep_name().as_str() {
        "mlas_only" => 256,
        _ => 8,
    }
}

fn build_execution_providers(cache: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
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
const MAX_SEQ_LENGTH: usize = 512;
const SPECIAL_TOKENS: [u32; 4] = [0, 1, 2, 3];

struct BenchModels {
    session: RefCell<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

fn load_bench_models(cache: &Path, eps: Vec<ort::ep::ExecutionProviderDispatch>) -> BenchModels {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache.to_path_buf())
        .with_progress(true)
        .build()
        .expect("Failed to build hf-hub API");

    let repo = api.repo(hf_hub::Repo::with_revision(
        REPO_ID.to_string(),
        hf_hub::RepoType::Model,
        REPO_REVISION.to_string(),
    ));

    let onnx_path = repo
        .get("onnx/model.onnx")
        .expect("Failed to get model.onnx");
    repo.get("onnx/model.onnx_data")
        .expect("Failed to get model.onnx_data");
    repo.get("onnx/Constant_7_attr__value")
        .expect("Failed to get Constant_7");

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

fn load_sparse_weights() -> (ndarray::Array1<f32>, f32) {
    let data = include_bytes!("../src/weights/sparse_linear.safetensors");
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
// Embedding helpers
// ---------------------------------------------------------------------------

fn tokenize_batch(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[impl AsRef<str>],
) -> (
    ndarray::Array2<i64>,
    ndarray::Array2<i64>,
    ndarray::Array2<i64>,
) {
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
    let ids = ndarray::Array2::from_shape_vec((batch_len, seq_len), ids_flat).unwrap();
    let mask = ndarray::Array2::from_shape_vec((batch_len, seq_len), mask_flat).unwrap();
    let type_ids = ndarray::Array2::<i64>::zeros((batch_len, seq_len));
    (ids, mask, type_ids)
}

fn bench_embed_dense(
    session: &RefCell<ort::session::Session>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[impl AsRef<str>],
    batch_size: usize,
) -> Vec<Vec<f32>> {
    let mut all = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(batch_size) {
        let (ids, mask, type_ids) = tokenize_batch(tokenizer, chunk);
        let ids_t = TensorRef::from_array_view(ids.view()).unwrap();
        let mask_t = TensorRef::from_array_view(mask.view()).unwrap();
        let type_t = TensorRef::from_array_view(type_ids.view()).unwrap();
        let mut sess = session.borrow_mut();
        let outputs = sess
            .run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => type_t,
            })
            .expect("session.run failed");
        let emb = outputs["sentence_embedding"]
            .try_extract_array::<f32>()
            .expect("Failed to extract sentence_embedding");
        for i in 0..chunk.len() {
            let row = emb.index_axis(ndarray::Axis(0), i);
            let slice = row.as_slice().expect("contiguous");
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = if norm > 0.0 {
                slice.iter().map(|x| x / norm).collect()
            } else {
                slice.to_vec()
            };
            all.push(normalized);
        }
    }
    all
}

fn bench_embed_sparse(
    session: &RefCell<ort::session::Session>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[impl AsRef<str>],
    batch_size: usize,
    weight: &ndarray::Array1<f32>,
    bias: f32,
) {
    let weight_view = weight.view();
    for chunk in texts.chunks(batch_size) {
        let str_refs: Vec<&str> = chunk.iter().map(AsRef::as_ref).collect();
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
        let ids = ndarray::Array2::from_shape_vec((batch_len, seq_len), ids_flat).unwrap();
        let mask = ndarray::Array2::from_shape_vec((batch_len, seq_len), mask_flat).unwrap();
        let type_ids = ndarray::Array2::<i64>::zeros((batch_len, seq_len));
        let ids_t = TensorRef::from_array_view(ids.view()).unwrap();
        let mask_t = TensorRef::from_array_view(mask.view()).unwrap();
        let type_t = TensorRef::from_array_view(type_ids.view()).unwrap();
        let mut sess = session.borrow_mut();
        let outputs = sess
            .run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => type_t,
            })
            .expect("session.run failed");
        let token_emb = outputs["token_embeddings"]
            .try_extract_array::<f32>()
            .expect("Failed to extract token_embeddings");
        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let attn = enc.get_attention_mask();
            let batch_hidden = token_emb.index_axis(ndarray::Axis(0), i);
            let mut token_weights: HashMap<usize, f32> = HashMap::new();
            for j in 0..ids.len() {
                if attn[j] == 0 {
                    continue;
                }
                let token_id = ids[j];
                if SPECIAL_TOKENS.contains(&token_id) {
                    continue;
                }
                let hidden = batch_hidden.index_axis(ndarray::Axis(0), j);
                let hidden_slice = hidden.as_slice().expect("contiguous");
                let hidden_view = ArrayView1::from(hidden_slice);
                let score = (hidden_view.dot(&weight_view) + bias).max(0.0);
                if score > 0.0 {
                    token_weights
                        .entry(token_id as usize)
                        .and_modify(|w| *w = w.max(score))
                        .or_insert(score);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dense embedding benchmarks
// ---------------------------------------------------------------------------

fn bench_dense(c: &mut Criterion) {
    let corpus = load_corpus();
    let cache = cache_dir();
    let eps = build_execution_providers(&cache);
    let label = ep_name();
    let onnx_bs = onnx_batch_size();

    eprintln!("[bench] Loading model with EP={label} ...");
    let models = load_bench_models(&cache, eps);

    // Warmup — triggers CoreML model compilation on first run.
    eprintln!("[bench] Dense warmup inference (onnx_batch_size={onnx_bs}) ...");
    bench_embed_dense(
        &models.session,
        &models.tokenizer,
        &["warmup text for CoreML compilation"],
        onnx_bs,
    );
    eprintln!("[bench] Dense model ready.");

    let mut names: Vec<&String> = corpus.scenarios.keys().collect();
    names.sort();

    let mut group = c.benchmark_group("dense");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for name in &names {
        let scenario = &corpus.scenarios[*name];

        let single = &scenario.texts[0..1];
        group.bench_with_input(BenchmarkId::new("single", *name), &single, |b, texts| {
            b.iter(|| {
                bench_embed_dense(&models.session, &models.tokenizer, texts, onnx_bs);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("batch", *name),
            &scenario.texts,
            |b, texts| {
                b.iter(|| {
                    bench_embed_dense(&models.session, &models.tokenizer, texts, onnx_bs);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Sparse embedding benchmarks
// ---------------------------------------------------------------------------

fn bench_sparse(c: &mut Criterion) {
    let corpus = load_corpus();
    let cache = cache_dir();
    let eps = build_execution_providers(&cache);
    let label = ep_name();
    let onnx_bs = onnx_batch_size();

    eprintln!("[bench] Loading model with EP={label} (for sparse) ...");
    let models = load_bench_models(&cache, eps);
    let (weight, bias) = load_sparse_weights();

    eprintln!("[bench] Sparse warmup inference (onnx_batch_size={onnx_bs}) ...");
    bench_embed_sparse(
        &models.session,
        &models.tokenizer,
        &["warmup text for CoreML compilation"],
        onnx_bs,
        &weight,
        bias,
    );
    eprintln!("[bench] Sparse model ready.");

    let mut names: Vec<&String> = corpus.scenarios.keys().collect();
    names.sort();

    let mut group = c.benchmark_group("sparse");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for name in &names {
        let scenario = &corpus.scenarios[*name];

        let single = &scenario.texts[0..1];
        group.bench_with_input(BenchmarkId::new("single", *name), &single, |b, texts| {
            b.iter(|| {
                bench_embed_sparse(
                    &models.session,
                    &models.tokenizer,
                    texts,
                    onnx_bs,
                    &weight,
                    bias,
                );
            });
        });

        group.bench_with_input(
            BenchmarkId::new("batch", *name),
            &scenario.texts,
            |b, texts| {
                b.iter(|| {
                    bench_embed_sparse(
                        &models.session,
                        &models.tokenizer,
                        texts,
                        onnx_bs,
                        &weight,
                        bias,
                    );
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_dense, bench_sparse);
criterion_main!(benches);
