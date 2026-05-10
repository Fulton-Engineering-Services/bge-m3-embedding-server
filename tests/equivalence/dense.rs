//! Dense embedding equivalence cases.
//!
//! `equivalence_all_seq_lengths` iterates over every fixture sequence length
//! recorded in `manifest.json` and asserts that the server's dense embedding
//! pipeline reproduces the reference vectors stored in
//! `reference_dense_seq_*.npy` within the per-model cosine tolerances
//! returned by [`super::helpers::cosine_tolerances_for`].

use crate::helpers::{
    cosine_similarity, cosine_tolerances_for, fixture_dir, l2_normalize, load_npy_f32,
    locate_model_file, locate_tokenizer, Tolerances,
};

/// Runs the full equivalence check for each fixture sequence length.
///
/// Requires `BGE_M3_EQUIVALENCE_TEST=1` and a warm model cache.
/// Set `BGE_M3_CACHE_DIR` to point to your model cache; defaults to `/tmp/bge-m3-cache`.
/// Set `BGE_M3_MODEL` to `fp32`, `fp16`, or `int8` (default `fp16`).
///
/// Run with:
/// ```sh
/// BGE_M3_EQUIVALENCE_TEST=1 BGE_M3_CACHE_DIR=/tmp/bge-m3-cache \
///   cargo test --test equivalence -- equivalence_all_seq_lengths --ignored --nocapture
/// ```
#[test]
#[ignore = "requires BGE_M3_EQUIVALENCE_TEST=1 and model download"]
fn equivalence_all_seq_lengths() {
    if std::env::var("BGE_M3_EQUIVALENCE_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP: BGE_M3_EQUIVALENCE_TEST != 1");
        return;
    }

    let manifest_path = fixture_dir().join("manifest.json");
    assert!(
        manifest_path.exists(),
        "Fixture manifest not found at {}. \
         Run scripts/generate_equivalence_fixtures.py first.",
        manifest_path.display()
    );

    let raw = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let model_str = std::env::var("BGE_M3_MODEL").unwrap_or_else(|_| "fp16".to_string());
    let tolerances = cosine_tolerances_for(&model_str);

    eprintln!("Model: {model_str}, tolerances: {tolerances:?}");

    for entry in manifest["files"].as_array().unwrap() {
        let seq = entry["seq_length"].as_u64().unwrap() as usize;
        eprintln!("\n=== seq_length={seq} ===");
        run_equivalence_for_seq(seq, &model_str, &tolerances);
    }
}

fn run_equivalence_for_seq(seq: usize, model_str: &str, tolerances: &Tolerances) {
    let tag = format!("{seq:04}");
    let cache_dir =
        std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".to_string());

    // Load fixture texts.
    let texts_path = fixture_dir().join(format!("texts_seq_{tag}.json"));
    let texts_raw = std::fs::read_to_string(&texts_path)
        .unwrap_or_else(|_| panic!("texts_seq_{tag}.json not found"));
    let texts: Vec<String> = serde_json::from_str(&texts_raw).expect("texts JSON");

    // Load reference dense embeddings (npy format).
    let dense_path = fixture_dir().join(format!("reference_dense_seq_{tag}.npy"));
    let reference_dense = load_npy_f32(&dense_path);

    let n = texts.len();
    assert_eq!(reference_dense.len() % 1024, 0);
    assert_eq!(reference_dense.len() / 1024, n);

    eprintln!("  Texts: {n}, seq: {seq}");

    // Load and run the ONNX model.
    let model_path = locate_model_file(&cache_dir, model_str)
        .expect("Could not locate model. Set BGE_M3_CACHE_DIR.");

    eprintln!("  Model: {}", model_path.display());

    let mut sess = ort::session::Session::builder()
        .expect("builder")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("opt")
        .with_intra_threads(1)
        .expect("threads")
        .commit_from_file(&model_path)
        .expect("session load");

    // Tokenize using the HuggingFace tokenizer (pure Rust via `tokenizers` crate).
    // Note: we use tokenizers directly here to stay in Rust — no Python needed
    // for the test itself. The tokenizer.json is downloaded along with the model.
    let tokenizer_path =
        locate_tokenizer(&cache_dir, model_str).expect("Could not locate tokenizer.json");

    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("tokenizer load");
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: seq,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .expect("truncation");
    // No BatchLongest — we pad manually to max seq in this batch.
    tokenizer.with_padding(None);

    let str_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch_fast(str_refs, true)
        .expect("tokenize");

    let pad_to = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(1);
    let mut ids_flat: Vec<i64> = Vec::with_capacity(n * pad_to);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(n * pad_to);
    for enc in &encodings {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        let seq_len = ids.len();
        ids_flat.extend(ids.iter().map(|&id| i64::from(id)));
        mask_flat.extend(mask.iter().map(|&m| i64::from(m)));
        let pad = pad_to.saturating_sub(seq_len);
        ids_flat.extend(std::iter::repeat_n(1i64, pad));
        mask_flat.extend(std::iter::repeat_n(0i64, pad));
    }

    let ids_arr = ndarray::Array2::from_shape_vec((n, pad_to), ids_flat).expect("ids array");
    let mask_arr = ndarray::Array2::from_shape_vec((n, pad_to), mask_flat).expect("mask array");

    eprintln!("  Running inference at shape ({n}, {pad_to})...");
    let ids_tensor = ort::value::TensorRef::from_array_view(ids_arr.view()).expect("ids tensor");
    let mask_tensor = ort::value::TensorRef::from_array_view(mask_arr.view()).expect("mask tensor");

    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })
        .expect("inference");

    // Extract dense embeddings and L2-normalize.
    let dense_out: Vec<f32> = if model_str == "fp32" {
        let emb = outputs["sentence_embedding"]
            .try_extract_array::<f32>()
            .expect("sentence_embedding");
        emb.as_slice().expect("contiguous").to_vec()
    } else {
        let lhs = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .expect("last_hidden_state");
        // CLS pool: take first token of each example.
        let mut pooled = Vec::with_capacity(n * 1024);
        for i in 0..n {
            let offset = i * pad_to * 1024;
            pooled.extend_from_slice(&lhs.as_slice().expect("contiguous")[offset..offset + 1024]);
        }
        pooled
    };

    let computed_dense: Vec<Vec<f32>> = dense_out
        .chunks(1024)
        .map(|row| {
            let mut v = row.to_vec();
            l2_normalize(&mut v);
            v
        })
        .collect();

    // Compute cosine similarities.
    let cosines: Vec<f64> = computed_dense
        .iter()
        .enumerate()
        .map(|(i, computed)| {
            let reference = &reference_dense[i * 1024..(i + 1) * 1024];
            f64::from(cosine_similarity(computed, reference))
        })
        .collect();

    let mean_cosine = cosines.iter().sum::<f64>() / cosines.len() as f64;
    let mut sorted_cosines = cosines.clone();
    sorted_cosines.sort_by(f64::total_cmp);
    let p5_idx = (cosines.len() as f64 * 0.05) as usize;
    let p5_cosine = sorted_cosines[p5_idx];

    eprintln!(
        "  Cosine sim — mean: {mean_cosine:.4}, p5: {p5_cosine:.4} \
         (thresholds: mean>={}, p5>={})",
        tolerances.mean_cosine, tolerances.p5_cosine
    );

    assert!(
        mean_cosine >= tolerances.mean_cosine,
        "seq={seq}: mean cosine {mean_cosine:.4} < {} for model={model_str}",
        tolerances.mean_cosine
    );
    assert!(
        p5_cosine >= tolerances.p5_cosine,
        "seq={seq}: p5 cosine {p5_cosine:.4} < {} for model={model_str}",
        tolerances.p5_cosine
    );

    eprintln!("  PASS seq={seq} model={model_str}");
}
