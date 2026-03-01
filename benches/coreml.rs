// CoreML EP benchmark harness for comparing execution provider configurations.
//
// Measures dense and sparse embedding inference at the fastembed API level,
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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fastembed::{
    EmbeddingModel, SparseModel, SparseTextEmbedding, TextEmbedding,
};

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
    PathBuf::from(
        std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".into()),
    )
}

/// Returns the ONNX sub-batch size to pass to `embed()`.
///
/// Mirrors the production default: `CoreML` EPs use `Some(8)` to avoid
/// `MLProgram` `FastPrediction` workspace OOM kills; MLAS uses `None`
/// (fastembed default, no pre-allocation issue).
///
/// Override with `BGE_M3_BENCH_ONNX_BATCH=<n>`.
fn onnx_batch_size() -> Option<usize> {
    if let Ok(val) = std::env::var("BGE_M3_BENCH_ONNX_BATCH") {
        return val.parse::<usize>().ok();
    }
    match ep_name().as_str() {
        "mlas_only" => None,
        _ => Some(8),
    }
}

fn build_execution_providers(
    cache: &Path,
) -> Vec<ort::ep::ExecutionProviderDispatch> {
    let config = ep_name();
    let coreml_cache = cache.join("coreml");

    // Base CoreML builder with shared options. On non-Apple platforms the EP
    // silently fails to register and ORT falls back to the CPU EP — equivalent
    // to mlas_only.
    let base = || {
        ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_specialization_strategy(
                ort::ep::coreml::SpecializationStrategy::FastPrediction,
            )
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
// Dense embedding benchmarks
// ---------------------------------------------------------------------------

fn bench_dense(c: &mut Criterion) {
    let corpus = load_corpus();
    let cache = cache_dir();
    let eps = build_execution_providers(&cache);
    let label = ep_name();

    eprintln!("[bench] Loading dense model with EP={label} ...");
    let onnx_bs = onnx_batch_size();
    let mut model = TextEmbedding::try_new(
        fastembed::TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_cache_dir(cache.clone())
            .with_show_download_progress(false)
            .with_execution_providers(eps),
    )
    .expect("Failed to load dense model");

    // Warmup — triggers CoreML model compilation on first run.
    eprintln!("[bench] Dense warmup inference (onnx_batch_size={onnx_bs:?}) ...");
    model
        .embed(vec!["warmup text for CoreML compilation"], onnx_bs)
        .expect("Dense warmup failed");
    eprintln!("[bench] Dense model ready.");

    // Sort scenario names for deterministic ordering across runs.
    let mut names: Vec<&String> = corpus.scenarios.keys().collect();
    names.sort();

    // Use a fixed group name so Criterion can compare across EP configs
    // via --save-baseline / --baseline. The EP label is logged to stderr.
    let mut group = c.benchmark_group("dense");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for name in &names {
        let scenario = &corpus.scenarios[*name];

        // Single-text latency (first text from scenario).
        let single = &scenario.texts[0..1];
        group.bench_with_input(
            BenchmarkId::new("single", *name),
            &single,
            |b, texts| {
                b.iter(|| model.embed(*texts, onnx_bs).expect("embed failed"));
            },
        );

        // Full-batch throughput (all texts from scenario).
        group.bench_with_input(
            BenchmarkId::new("batch", *name),
            &scenario.texts,
            |b, texts| {
                b.iter(|| model.embed(texts, onnx_bs).expect("embed failed"));
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

    eprintln!("[bench] Loading sparse model with EP={label} ...");
    let onnx_bs = onnx_batch_size();
    let mut model = SparseTextEmbedding::try_new(
        fastembed::SparseInitOptions::new(SparseModel::BGEM3)
            .with_cache_dir(cache.clone())
            .with_show_download_progress(false)
            .with_execution_providers(eps),
    )
    .expect("Failed to load sparse model");

    eprintln!("[bench] Sparse warmup inference (onnx_batch_size={onnx_bs:?}) ...");
    model
        .embed(vec!["warmup text for CoreML compilation"], onnx_bs)
        .expect("Sparse warmup failed");
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
        group.bench_with_input(
            BenchmarkId::new("single", *name),
            &single,
            |b, texts| {
                b.iter(|| model.embed(*texts, onnx_bs).expect("embed failed"));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch", *name),
            &scenario.texts,
            |b, texts| {
                b.iter(|| model.embed(texts, onnx_bs).expect("embed failed"));
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
