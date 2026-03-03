#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::redundant_closure_for_method_calls
)]
//! Phase A — FP16 vs FP32 embedding fidelity evaluation.
//!
//! Generates embeddings for every text in the production benchmark corpus using
//! both the FP32 (BAAI/bge-m3) and FP16 (Xenova/bge-m3) ONNX models, then
//! computes precision metrics:
//!
//! **Dense**: per-text cosine similarity, max absolute element difference.
//! **Sparse**: Jaccard index of active token indices, Pearson correlation of
//! shared-index weights.
//!
//! ```bash
//! # Requires:
//! #   1. FP32 model cached at BGE_M3_CACHE_DIR (default /tmp/bge-m3-cache)
//! #   2. FP16 model at FP16_MODEL_PATH (default /tmp/bge-m3-cache/xenova-bge-m3-fp16/model_fp16.onnx)
//! #
//! # Download FP16 model:
//! #   mkdir -p /tmp/bge-m3-cache/xenova-bge-m3-fp16
//! #   curl -L https://huggingface.co/Xenova/bge-m3/resolve/main/onnx/model_fp16.onnx \
//! #        -o /tmp/bge-m3-cache/xenova-bge-m3-fp16/model_fp16.onnx
//! #
//! # Run:
//! #   cargo run --example fp16_eval --release
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::{env, fs};

use anyhow::{Context, Result};
use ndarray::ArrayView1;
use ort::value::TensorRef;
use serde::Deserialize;

// ── Corpus (shared with benches/coreml.rs) ──────────────────────────────

#[derive(Deserialize)]
struct Corpus {
    scenarios: HashMap<String, Scenario>,
}

#[derive(Deserialize)]
struct Scenario {
    #[allow(dead_code)]
    description: String,
    texts: Vec<String>,
}

fn load_corpus() -> Result<Corpus> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/corpus.json");
    let raw = fs::read_to_string(&path).context("Failed to read corpus.json")?;
    serde_json::from_str(&raw).context("Failed to parse corpus.json")
}

// ── Constants ───────────────────────────────────────────────────────────

const REPO_ID: &str = "BAAI/bge-m3";
const REPO_REVISION: &str = "5617a9f61b028005a4858fdac845db406aefb181";
const MAX_SEQ_LENGTH: usize = 512;
const SPECIAL_TOKENS: [u32; 4] = [0, 1, 2, 3];

// ── Sparse weight loading (bundled safetensors) ─────────────────────────

struct SparseLinearWeights {
    weight: Vec<f32>, // [1024]
    bias: f32,
}

// NOTE(ARC-4): Model loading, tokenization, and embedding functions below
// intentionally duplicate logic from src/embedder.rs. The production code is
// a binary crate (no lib target), so examples cannot import from it. This
// example also has legitimate differences: no batch_size chunking (pairwise
// comparison), last_hidden_state fallback for Xenova FP16 model variant, and
// its own SparseLinearWeights/SparseFp16 types for FP16 evaluation output.

fn load_sparse_weights() -> SparseLinearWeights {
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
    assert_eq!(weight.len(), 1024, "sparse_linear weight must be [1024]");
    SparseLinearWeights { weight, bias }
}

// ── Model loading ───────────────────────────────────────────────────────

fn download_model_files(cache_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(true)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build hf-hub API: {e}"))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        REPO_ID.to_string(),
        hf_hub::RepoType::Model,
        REPO_REVISION.to_string(),
    ));
    let onnx_path = repo.get("onnx/model.onnx").context("model.onnx")?;
    repo.get("onnx/model.onnx_data")
        .context("model.onnx_data")?;
    repo.get("onnx/Constant_7_attr__value")
        .context("Constant_7")?;
    let tokenizer_path = repo.get("tokenizer.json").context("tokenizer.json")?;
    Ok((onnx_path, tokenizer_path))
}

fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ_LENGTH,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("Truncation: {e}"))?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_id: 1,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    }));
    Ok(tokenizer)
}

fn load_session(model_path: &Path) -> Result<ort::session::Session> {
    let session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;
    Ok(session)
}

// ── Embedding functions ─────────────────────────────────────────────────

struct SparseFp16 {
    indices: Vec<usize>,
    values: Vec<f32>,
}

fn embed_dense(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let str_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch(str_refs, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
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

    let ids_t = TensorRef::from_array_view(ids_array.view())?;
    let mask_t = TensorRef::from_array_view(mask_array.view())?;

    let outputs = session.run(ort::inputs! {
        "input_ids" => ids_t,
        "attention_mask" => mask_t,
    })?;

    // Xenova/bge-m3 FP16 exports last_hidden_state [batch, seq, 1024].
    // CLS-pool by selecting position 0 along the sequence axis.
    let lhs = outputs["last_hidden_state"].try_extract_array::<f32>()?;
    let emb = lhs.index_axis(ndarray::Axis(1), 0);

    let mut result = Vec::with_capacity(batch_len);
    for i in 0..batch_len {
        let row = emb.index_axis(ndarray::Axis(0), i);
        let slice = row.as_slice().expect("contiguous");
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            slice.iter().map(|x| x / norm).collect()
        } else {
            slice.to_vec()
        };
        result.push(normalized);
    }
    Ok(result)
}

fn embed_sparse(
    session: &mut ort::session::Session,
    tokenizer: &tokenizers::Tokenizer,
    weights: &SparseLinearWeights,
    texts: &[String],
) -> Result<Vec<SparseFp16>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let weight_arr = ArrayView1::from(&weights.weight);
    let mut results = Vec::with_capacity(texts.len());

    for text in texts {
        let encodings = tokenizer
            .encode_batch(vec![text.as_str()], true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let encoding = &encodings[0];
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        let seq_len = input_ids.len();
        let ids_array = ndarray::Array2::from_shape_vec((1, seq_len), input_ids.clone())?;
        let mask_array = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.clone())?;

        let ids_t = TensorRef::from_array_view(ids_array.view())?;
        let mask_t = TensorRef::from_array_view(mask_array.view())?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })?;

        // The output name depends on the model variant:
        // - BAAI/bge-m3: "token_embeddings"
        // - Xenova/bge-m3 FP16: "last_hidden_state"
        //
        // NOTE(COR-4): Both extractions use `try_extract_array::<f32>()`. For the
        // FP16 model variant (Xenova/bge-m3), this relies on ORT automatically
        // promoting FP16 tensor data to F32 during extraction. Verified with
        // ort 2.0.0-rc.11 on MLAS and CoreML EPs. If a future ORT version
        // changes this behavior, extraction will fail with a type mismatch error.
        let token_emb = if let Ok(t) = outputs["token_embeddings"].try_extract_array::<f32>() {
            t
        } else {
            outputs["last_hidden_state"].try_extract_array::<f32>()?
        };

        let mut token_weights: HashMap<usize, f32> = HashMap::new();
        for seq_idx in 0..seq_len {
            if attention_mask[seq_idx] == 0 {
                continue;
            }
            let token_id = input_ids[seq_idx] as u32;
            if SPECIAL_TOKENS.contains(&token_id) {
                continue;
            }
            let batch0 = token_emb.index_axis(ndarray::Axis(0), 0);
            let hidden = batch0.index_axis(ndarray::Axis(0), seq_idx);
            let hidden_slice = hidden
                .as_slice()
                .expect("hidden state should be contiguous");
            let hidden_view = ArrayView1::from(hidden_slice);
            let score = (hidden_view.dot(&weight_arr) + weights.bias).max(0.0);
            if score > 0.0 {
                token_weights
                    .entry(token_id as usize)
                    .and_modify(|w| *w = w.max(score))
                    .or_insert(score);
            }
        }

        let mut indices: Vec<usize> = token_weights.keys().copied().collect();
        indices.sort_unstable();
        let values: Vec<f32> = indices.iter().map(|i| token_weights[i]).collect();
        results.push(SparseFp16 { indices, values });
    }

    Ok(results)
}

// ── Metrics ─────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0_f64, f64::max)
}

fn jaccard_index(a: &[usize], b: &[usize]) -> f64 {
    let set_a: HashSet<usize> = a.iter().copied().collect();
    let set_b: HashSet<usize> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 1.0;
    }
    intersection as f64 / union as f64
}

/// Pearson correlation of weights on shared (intersection) indices.
fn weight_correlation(
    a_indices: &[usize],
    a_values: &[f32],
    b_indices: &[usize],
    b_values: &[f32],
) -> f64 {
    let a_map: HashMap<usize, f32> = a_indices
        .iter()
        .copied()
        .zip(a_values.iter().copied())
        .collect();
    let b_map: HashMap<usize, f32> = b_indices
        .iter()
        .copied()
        .zip(b_values.iter().copied())
        .collect();

    let shared: Vec<usize> = a_map
        .keys()
        .filter(|k| b_map.contains_key(k))
        .copied()
        .collect();

    if shared.len() < 2 {
        // With 0 shared indices correlation is undefined (NaN).
        // With exactly 1 shared index variance is zero, so Pearson correlation
        // is also undefined — return NaN rather than a misleading 1.0.
        return f64::NAN;
    }

    let a_vals: Vec<f64> = shared.iter().map(|k| f64::from(a_map[k])).collect();
    let b_vals: Vec<f64> = shared.iter().map(|k| f64::from(b_map[k])).collect();

    let n = shared.len() as f64;
    let mean_a = a_vals.iter().sum::<f64>() / n;
    let mean_b = b_vals.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..shared.len() {
        let da = a_vals[i] - mean_a;
        let db = b_vals[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return 1.0; // constant values are perfectly correlated
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn stats(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    (min, mean, max)
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cache_dir =
        PathBuf::from(env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".into()));
    let fp16_model_path = PathBuf::from(
        env::var("FP16_MODEL_PATH")
            .unwrap_or_else(|_| "/tmp/bge-m3-cache/xenova-bge-m3-fp16/model_fp16.onnx".into()),
    );

    println!("Phase A — FP16 vs FP32 Embedding Fidelity Evaluation");
    println!("=====================================================\n");

    // ── Load corpus ──
    let corpus = load_corpus()?;
    let total_texts: usize = corpus.scenarios.values().map(|s| s.texts.len()).sum();
    println!(
        "Corpus: {} scenarios, {} total texts\n",
        corpus.scenarios.len(),
        total_texts
    );

    // ── Load FP32 models via direct ORT ──
    println!("[1/5] Loading FP32 model (BAAI/bge-m3 via hf-hub)...");
    let (onnx_path, tokenizer_path) = download_model_files(&cache_dir)?;
    let fp32_tokenizer = load_tokenizer(&tokenizer_path)?;
    let mut fp32_session = load_session(&onnx_path)?;
    println!("       FP32 model ready.");

    // ── Load FP16 model ──
    println!("[2/5] Loading FP16 model (Xenova/bge-m3)...");
    let fp16_size = fs::metadata(&fp16_model_path)
        .with_context(|| format!("stat {}", fp16_model_path.display()))?
        .len();
    println!(
        "       FP16 model: {:.1} MB",
        fp16_size as f64 / 1_048_576.0
    );
    let mut fp16_session = load_session(&fp16_model_path)?;
    // Reuse the same tokenizer — both BAAI and Xenova models use XLM-RoBERTa tokenizer.
    let fp16_tokenizer = load_tokenizer(&tokenizer_path)?;
    println!("       FP16 model ready.");

    // ── Load sparse weights ──
    println!("[3/5] Loading sparse weights from bundled safetensors...");
    let sparse_weights = load_sparse_weights();
    println!(
        "       weight: [{}], bias: {:.6}",
        sparse_weights.weight.len(),
        sparse_weights.bias
    );
    println!("       Sparse weights ready.\n");

    // ── Run comparisons ──
    let mut scenario_names: Vec<&String> = corpus.scenarios.keys().collect();
    scenario_names.sort();

    let mut all_dense_cosines = Vec::new();
    let mut all_dense_diffs = Vec::new();
    let mut all_sparse_jaccards = Vec::new();
    let mut all_sparse_correlations = Vec::new();

    for name in &scenario_names {
        let scenario = &corpus.scenarios[*name];
        let texts = &scenario.texts;
        println!(
            "── {} ({} texts) ─────────────────────────",
            name,
            texts.len()
        );

        // Dense comparison
        let fp32_embs = embed_dense(&mut fp32_session, &fp32_tokenizer, texts)?;
        let fp16_embs = embed_dense(&mut fp16_session, &fp16_tokenizer, texts)?;

        let mut cosines = Vec::with_capacity(texts.len());
        let mut diffs = Vec::with_capacity(texts.len());

        for (fp32, fp16) in fp32_embs.iter().zip(fp16_embs.iter()) {
            cosines.push(cosine_similarity(fp32, fp16));
            diffs.push(max_abs_diff(fp32, fp16));
        }

        let (c_min, c_mean, c_max) = stats(&cosines);
        let (d_min, d_mean, d_max) = stats(&diffs);
        println!("  Dense cosine sim:   min={c_min:.6}  mean={c_mean:.6}  max={c_max:.6}");
        println!("  Dense max abs diff: min={d_min:.6}  mean={d_mean:.6}  max={d_max:.6}");

        all_dense_cosines.extend_from_slice(&cosines);
        all_dense_diffs.extend_from_slice(&diffs);

        // Sparse comparison
        let fp32_sp = embed_sparse(&mut fp32_session, &fp32_tokenizer, &sparse_weights, texts)?;
        let fp16_sp = embed_sparse(&mut fp16_session, &fp16_tokenizer, &sparse_weights, texts)?;

        let mut jaccards = Vec::with_capacity(texts.len());
        let mut correlations = Vec::with_capacity(texts.len());

        for (fp32, fp16) in fp32_sp.iter().zip(fp16_sp.iter()) {
            jaccards.push(jaccard_index(&fp32.indices, &fp16.indices));
            correlations.push(weight_correlation(
                &fp32.indices,
                &fp32.values,
                &fp16.indices,
                &fp16.values,
            ));
        }

        let (j_min, j_mean, j_max) = stats(&jaccards);
        let valid_corrs: Vec<f64> = correlations
            .iter()
            .copied()
            .filter(|c| !c.is_nan())
            .collect();
        let (r_min, r_mean, r_max) = stats(&valid_corrs);
        println!("  Sparse Jaccard:     min={j_min:.6}  mean={j_mean:.6}  max={j_max:.6}");
        println!(
            "  Sparse weight corr: min={r_min:.6}  mean={r_mean:.6}  max={r_max:.6}  (n={})",
            valid_corrs.len()
        );
        println!();

        all_sparse_jaccards.extend_from_slice(&jaccards);
        all_sparse_correlations.extend(valid_corrs);
    }

    // ── Summary ──
    println!("══════════════════════════════════════════════════════");
    println!("OVERALL ({total_texts} texts)");
    println!("══════════════════════════════════════════════════════");

    let (c_min, c_mean, c_max) = stats(&all_dense_cosines);
    let (d_min, d_mean, d_max) = stats(&all_dense_diffs);
    let (j_min, j_mean, j_max) = stats(&all_sparse_jaccards);
    let (r_min, r_mean, r_max) = stats(&all_sparse_correlations);

    println!("  Dense cosine sim:   min={c_min:.6}  mean={c_mean:.6}  max={c_max:.6}");
    println!("  Dense max abs diff: min={d_min:.6}  mean={d_mean:.6}  max={d_max:.6}");
    println!("  Sparse Jaccard:     min={j_min:.6}  mean={j_mean:.6}  max={j_max:.6}");
    println!(
        "  Sparse weight corr: min={r_min:.6}  mean={r_mean:.6}  max={r_max:.6}  (n={})",
        all_sparse_correlations.len()
    );
    println!();

    // ── Pass/fail against Phase A targets ──
    println!("Phase A Targets:");
    let dense_pass = c_min > 0.999;
    let diff_pass = d_max < 0.01;
    let jaccard_pass = j_min > 0.95;
    let corr_pass = r_min > 0.99;

    println!(
        "  Dense cosine > 0.999:   {} (min={c_min:.6})",
        if dense_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Max abs diff < 0.01:    {} (max={d_max:.6})",
        if diff_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Sparse Jaccard > 0.95:  {} (min={j_min:.6})",
        if jaccard_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Sparse corr > 0.99:     {} (min={r_min:.6})",
        if corr_pass { "PASS" } else { "FAIL" }
    );

    let all_pass = dense_pass && diff_pass && jaccard_pass && corr_pass;
    println!(
        "\nOverall: {}",
        if all_pass {
            "ALL TARGETS MET — FP16 is suitable for production"
        } else {
            "SOME TARGETS MISSED — review metrics before proceeding"
        }
    );

    if all_pass {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Phase A fidelity targets not met"))
    }
}
