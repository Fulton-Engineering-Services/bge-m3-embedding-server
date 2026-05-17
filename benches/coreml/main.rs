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
//
// File layout (under benches/coreml/):
//   - main.rs:   criterion harness wiring + the cross-variant `quality` group.
//   - setup.rs:  corpus + EP config + model loading + sparse weights + tokenize.
//   - dense.rs:  `bench_dense` group + `bench_embed_dense` helper.
//   - sparse.rs: `bench_sparse` group + `bench_embed_sparse` helper.

#![allow(clippy::cast_possible_truncation)]

mod dense;
mod setup;
mod sparse;

use criterion::{Criterion, criterion_group, criterion_main};

use crate::dense::{bench_dense, bench_embed_dense};
use crate::setup::{
    bench_model_variant, build_execution_providers, cache_dir, load_bench_models_for_variant,
    load_corpus, load_sparse_weights, onnx_batch_size,
};
use crate::sparse::{bench_embed_sparse, bench_sparse};

// ---------------------------------------------------------------------------
// Quality benchmark — cosine similarity vs FP32 baseline
// ---------------------------------------------------------------------------

/// Compares embedding quality of the configured variant against the FP32 baseline.
///
/// Skipped automatically when `BGE_M3_MODEL=fp32` (no comparison needed).
///
/// For each scenario in the corpus, embeds all texts with both FP32 (MLAS,
/// no EPs) and the configured variant, then reports cosine similarity stats
/// to stderr. Dense embeddings are already L2-normalized so dot-product
/// equals cosine similarity directly.
///
/// Usage:
///
///   `BGE_M3_MODEL=fp16` cargo bench --bench coreml -- quality
///   `BGE_M3_MODEL=int8` cargo bench --bench coreml -- quality
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn bench_quality(c: &mut Criterion) {
    let variant = bench_model_variant();
    if variant == "fp32" {
        eprintln!("[quality] Skipping: BGE_M3_MODEL=fp32, no comparison baseline needed.");
        return;
    }
    if std::env::var("BGE_M3_SKIP_QUALITY").is_ok() {
        eprintln!("[quality] Skipping: BGE_M3_SKIP_QUALITY set (quality data already recorded).");
        return;
    }

    let corpus = load_corpus();
    let cache = cache_dir();
    let onnx_bs = onnx_batch_size();
    let (weight, bias) = load_sparse_weights();

    // Collect all texts in sorted-scenario order for deterministic results.
    let mut names: Vec<&String> = corpus.scenarios.keys().collect();
    names.sort();
    let all_texts: Vec<String> = names
        .iter()
        .flat_map(|n| corpus.scenarios[*n].texts.iter().cloned())
        .collect();

    eprintln!("[quality] Loading FP32 baseline (MLAS, no EPs) for {variant} comparison ...");
    let fp32_models = load_bench_models_for_variant(&cache, vec![], "fp32");

    eprintln!("[quality] Loading {variant} variant ...");
    let variant_eps = build_execution_providers(&cache);
    let variant_models = load_bench_models_for_variant(&cache, variant_eps, &variant);

    eprintln!(
        "[quality] Computing FP32 reference embeddings ({} texts) ...",
        all_texts.len()
    );
    let fp32_dense = bench_embed_dense(
        &fp32_models.session,
        &fp32_models.tokenizer,
        &all_texts,
        onnx_bs,
        "fp32",
    );
    let fp32_sparse = bench_embed_sparse(
        &fp32_models.session,
        &fp32_models.tokenizer,
        &all_texts,
        onnx_bs,
        &weight,
        bias,
        "fp32",
    );

    eprintln!(
        "[quality] Computing {variant} embeddings ({} texts) ...",
        all_texts.len()
    );
    let variant_dense = bench_embed_dense(
        &variant_models.session,
        &variant_models.tokenizer,
        &all_texts,
        onnx_bs,
        &variant,
    );
    let variant_sparse = bench_embed_sparse(
        &variant_models.session,
        &variant_models.tokenizer,
        &all_texts,
        onnx_bs,
        &weight,
        bias,
        &variant,
    );

    // Dense cosine similarity (embeddings are L2-normalized → dot product = cosine).
    let dense_sims: Vec<f32> = fp32_dense
        .iter()
        .zip(variant_dense.iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
        .collect();
    let dense_mean: f32 = dense_sims.iter().sum::<f32>() / dense_sims.len() as f32;
    let dense_min: f32 = dense_sims.iter().copied().fold(f32::INFINITY, f32::min);
    let mut dense_sorted = dense_sims.clone();
    dense_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p5_idx = dense_sorted.len() / 20; // 5th percentile index: floor(n * 0.05)
    let dense_p5 = dense_sorted[p5_idx];
    eprintln!(
        "[quality] Dense  {variant} vs fp32 — n={}, mean={:.6}, p5={:.6}, min={:.6}",
        dense_sims.len(),
        dense_mean,
        dense_p5,
        dense_min,
    );

    // Sparse cosine similarity over HashMap representation.
    let sparse_sims: Vec<f32> = fp32_sparse
        .iter()
        .zip(variant_sparse.iter())
        .map(|(ref_map, var_map)| {
            let dot: f32 = ref_map
                .iter()
                .filter_map(|(k, v)| var_map.get(k).map(|w| v * w))
                .sum();
            let norm_a: f32 = ref_map.values().map(|v| v * v).sum::<f32>().sqrt();
            let norm_b: f32 = var_map.values().map(|v| v * v).sum::<f32>().sqrt();
            if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            }
        })
        .collect();
    let sparse_mean: f32 = sparse_sims.iter().sum::<f32>() / sparse_sims.len() as f32;
    let sparse_min: f32 = sparse_sims.iter().copied().fold(f32::INFINITY, f32::min);
    let mut sparse_sorted = sparse_sims.clone();
    sparse_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sparse_p5_idx = sparse_sorted.len() / 20; // 5th percentile index: floor(n * 0.05)
    let sparse_p5 = sparse_sorted[sparse_p5_idx];
    eprintln!(
        "[quality] Sparse {variant} vs fp32 — n={}, mean={:.6}, p5={:.6}, min={:.6}",
        sparse_sims.len(),
        sparse_mean,
        sparse_p5,
        sparse_min,
    );

    // Benchmark the similarity computation itself (pure dot-product cost).
    let mut group = c.benchmark_group("quality");
    group.sample_size(10);
    group.bench_function(format!("dense_cosine_sim_{variant}_vs_fp32"), |b| {
        b.iter(|| {
            let _: Vec<f32> = std::hint::black_box(&fp32_dense)
                .iter()
                .zip(std::hint::black_box(&variant_dense).iter())
                .map(|(a, bv)| a.iter().zip(bv.iter()).map(|(x, y)| x * y).sum::<f32>())
                .collect();
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_dense, bench_sparse, bench_quality);
criterion_main!(benches);
