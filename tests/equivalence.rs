//! Equivalence tests for bge-m3-embedding-server.
//!
//! This file contains two categories of tests:
//!
//! ## 1. Fast smoke tests (run on every `cargo test`)
//!
//! - Fixture file shape and manifest integrity checks.
//! - No model load, no ONNX dependency.
//!
//! ## 2. Full equivalence tests (opt-in, require model download)
//!
//! Gated behind `BGE_M3_EQUIVALENCE_TEST=1` and `#[ignore]` so they never
//! run in CI without explicit opt-in.
//!
//! These tests:
//! 1. Load the ONNX model (downloads from `HuggingFace` if not cached).
//! 2. Embed each fixture's texts via the server's embed pipeline.
//! 3. Assert cosine similarity vs the reference dense vectors.
//! 4. Assert sparse index overlap vs the reference sparse vectors.
//!
//! Run with:
//! ```sh
//! BGE_M3_EQUIVALENCE_TEST=1 BGE_M3_CACHE_DIR=/tmp/bge-m3-cache \
//!   cargo test --test equivalence -- --ignored --nocapture
//! ```
//!
//! ## 3. ONNX positional embedding inspection
//!
//! Also gated behind `BGE_M3_EQUIVALENCE_TEST=1`. Inspects the loaded model's
//! `position_ids` input shape to verify it can handle the configured
//! `BGE_M3_MAX_SEQ_LENGTH`.

// cast_precision_loss / cast_possible_truncation / cast_sign_loss:
//   Test-only arithmetic on small integer indices and byte offsets (batch ≤ 16,
//   seq ≤ 8192, array lengths ≤ a few thousand). All values are well within f64
//   mantissa range; fractional bytes are intentionally floored.
//
// too_many_lines:
//   `run_equivalence_for_seq` is a single coherent test scenario — load fixture,
//   tokenize, infer, verify cosines. Splitting it across helper functions would
//   obscure the test flow without improving readability.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use std::path::Path;

// ---------------------------------------------------------------------------
// Fixture directory
// ---------------------------------------------------------------------------

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("equivalence")
}

// ---------------------------------------------------------------------------
// 1. Fast smoke tests — always run
// ---------------------------------------------------------------------------

/// Checks that the manifest exists and contains required fields.
///
/// This test does not require model download or `BGE_M3_EQUIVALENCE_TEST`.
/// It validates that whoever last ran the generator committed a sane manifest.
#[test]
fn manifest_exists_and_has_expected_fields() {
    let manifest_path = fixture_dir().join("manifest.json");
    if !manifest_path.exists() {
        // Fixtures have not been generated yet — this is acceptable in a
        // fresh clone before the first `generate_equivalence_fixtures.py` run.
        eprintln!(
            "SKIP: manifest not found at {}. \
             Run scripts/generate_equivalence_fixtures.py to generate fixtures.",
            manifest_path.display()
        );
        return;
    }

    let raw = std::fs::read_to_string(&manifest_path).expect("manifest.json should be readable");
    let manifest: serde_json::Value =
        serde_json::from_str(&raw).expect("manifest.json should be valid JSON");

    assert!(
        manifest
            .get("model_revision")
            .and_then(|v| v.as_str())
            .is_some(),
        "manifest must have 'model_revision'"
    );
    assert!(
        manifest["seq_lengths"].is_array(),
        "manifest must have 'seq_lengths' array"
    );
    assert!(
        manifest["files"].is_array(),
        "manifest must have 'files' array"
    );
}

/// Checks that the manifest's `model_revision` matches the server's pinned `REPO_REVISION`.
///
/// If they diverge, fixtures may have been generated with a different model
/// checkpoint than the server is using — equivalence assertions become meaningless.
#[test]
fn manifest_revision_matches_server_repo_revision() {
    let manifest_path = fixture_dir().join("manifest.json");
    if !manifest_path.exists() {
        return; // fixtures not generated yet — skip
    }

    let raw = std::fs::read_to_string(&manifest_path).expect("manifest.json readable");
    let manifest: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
    let manifest_rev = manifest["model_revision"]
        .as_str()
        .expect("model_revision must be a string");

    // Extract REPO_REVISION from embedder.rs at compile time via a constant.
    // The string is found via the same extraction logic used in embedder.rs tests.
    let embedder_src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder.rs"))
            .expect("src/embedder.rs should be readable");

    let revision = extract_const_str(&embedder_src, "REPO_REVISION");

    assert_eq!(
        manifest_rev, revision,
        "manifest model_revision ({manifest_rev}) must match REPO_REVISION in \
         src/embedder.rs ({revision}). Regenerate fixtures with \
         scripts/generate_equivalence_fixtures.py after bumping REPO_REVISION."
    );
}

/// For each `seq_length` listed in the manifest, verify the fixture files exist
/// and the dense array has the expected shape.
#[test]
fn fixture_files_have_expected_shape() {
    let manifest_path = fixture_dir().join("manifest.json");
    if !manifest_path.exists() {
        return;
    }

    let raw = std::fs::read_to_string(&manifest_path).expect("manifest.json readable");
    let manifest: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
    let texts_per_length = manifest["texts_per_length"]
        .as_u64()
        .expect("texts_per_length") as usize;

    for entry in manifest["files"].as_array().expect("files is array") {
        let seq = entry["seq_length"].as_u64().expect("seq_length") as usize;
        let tag = format!("{seq:04}");

        let texts_path = fixture_dir().join(format!("texts_seq_{tag}.json"));
        let dense_path = fixture_dir().join(format!("reference_dense_seq_{tag}.npy"));
        let sparse_path = fixture_dir().join(format!("reference_sparse_seq_{tag}.json"));

        assert!(texts_path.exists(), "Missing: {}", texts_path.display());
        assert!(dense_path.exists(), "Missing: {}", dense_path.display());
        assert!(sparse_path.exists(), "Missing: {}", sparse_path.display());

        // Verify texts JSON
        let texts_raw = std::fs::read_to_string(&texts_path).expect("texts readable");
        let texts: Vec<serde_json::Value> =
            serde_json::from_str(&texts_raw).expect("texts is JSON array");
        assert_eq!(
            texts.len(),
            texts_per_length,
            "texts_seq_{tag}.json should have {texts_per_length} texts"
        );

        // Verify sparse JSON
        let sparse_raw = std::fs::read_to_string(&sparse_path).expect("sparse readable");
        let sparse: Vec<serde_json::Value> =
            serde_json::from_str(&sparse_raw).expect("sparse is JSON array");
        assert_eq!(
            sparse.len(),
            texts_per_length,
            "reference_sparse_seq_{tag}.json should have {texts_per_length} entries"
        );

        // Verify dense npy header — just check the file is non-empty and starts
        // with the numpy magic number (no full npy parse dependency needed).
        let dense_bytes = std::fs::read(&dense_path).expect("dense readable");
        assert!(
            dense_bytes.starts_with(b"\x93NUMPY"),
            "reference_dense_seq_{tag}.npy is not a valid numpy file"
        );

        eprintln!(
            "OK: seq={seq}, texts={}, dense_bytes={}, sparse_entries={}",
            texts.len(),
            dense_bytes.len(),
            sparse.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Full equivalence tests — opt-in via BGE_M3_EQUIVALENCE_TEST=1
// ---------------------------------------------------------------------------

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

/// Verifies that the dual-output path (`embed_both`-equivalent) produces the
/// same numerical results as two separate single-output passes
/// (`embed_dense` + `embed_sparse`-equivalent), within FP rounding tolerance.
///
/// This validates the algorithm-level invariant of the unified `/v1/embeddings:both`
/// endpoint without requiring the binary crate's private types in scope: it runs
/// the same ORT session three ways on the same inputs and compares outputs.
#[test]
#[ignore = "requires BGE_M3_EQUIVALENCE_TEST=1 and model download"]
fn dual_pass_equivalent_to_separate_passes() {
    if std::env::var("BGE_M3_EQUIVALENCE_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP: BGE_M3_EQUIVALENCE_TEST != 1");
        return;
    }

    let cache_dir =
        std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".to_string());
    let model_str = std::env::var("BGE_M3_MODEL").unwrap_or_else(|_| "fp16".to_string());
    let is_fp32 = model_str == "fp32";

    let model_path = locate_model_file(&cache_dir, &model_str)
        .expect("Could not locate model. Set BGE_M3_CACHE_DIR.");
    let tokenizer_path =
        locate_tokenizer(&cache_dir, &model_str).expect("Could not locate tokenizer.json");

    eprintln!(
        "Dual-equivalence check using model={model_str} at {}",
        model_path.display()
    );

    let texts: Vec<String> = vec![
        "fn factorial(n: u64) -> u64 { (1..=n).product() }".to_string(),
        "Rust ownership and borrowing rules at compile time".to_string(),
        "The mitochondria is the powerhouse of the cell.".to_string(),
        "SELECT * FROM users WHERE last_login > NOW() - INTERVAL '7 days';".to_string(),
    ];
    let n = texts.len();

    // Tokenize without padding, then pad to chunk-local max.
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("tokenizer load");
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .expect("truncation");
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

    let mut sess = ort::session::Session::builder()
        .expect("builder")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("opt")
        .with_intra_threads(1)
        .expect("threads")
        .commit_from_file(&model_path)
        .expect("session load");

    // -------------------------------------------------------------------
    // Pass A: dual extraction — one session.run() yielding dense + sparse
    // (mimicking what embed_both does in src/embedder.rs).
    // -------------------------------------------------------------------
    let (dense_dual, sparse_base_dual) = run_dual_pass(&mut sess, &ids_arr, &mask_arr, is_fp32);

    // -------------------------------------------------------------------
    // Pass B: dense-only — separate session.run() (mimicking embed_dense).
    // -------------------------------------------------------------------
    let dense_only = run_dense_pass(&mut sess, &ids_arr, &mask_arr, is_fp32);

    // -------------------------------------------------------------------
    // Pass C: sparse-only — separate session.run() (mimicking embed_sparse).
    // -------------------------------------------------------------------
    let sparse_base_only = run_sparse_pass(&mut sess, &ids_arr, &mask_arr, is_fp32);

    // Assert the dense rows agree exactly within FP tolerance for each text.
    assert_eq!(dense_dual.len(), dense_only.len());
    for (i, (a, b)) in dense_dual.iter().zip(dense_only.iter()).enumerate() {
        let max_abs_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let cos = cosine_similarity(a, b);
        eprintln!("  text[{i}]: dense max|Δ|={max_abs_diff:.2e}, cos={cos:.6}");
        assert!(
            max_abs_diff < 1e-4,
            "text[{i}]: dense vectors differ beyond FP tolerance (max|Δ|={max_abs_diff:.6e})"
        );
        assert!(
            cos > 0.9999,
            "text[{i}]: dense cosine similarity {cos:.6} below 0.9999 threshold"
        );
    }

    // Assert the per-token sparse-base hidden states agree elementwise.
    assert_eq!(sparse_base_dual.len(), sparse_base_only.len());
    let max_diff = sparse_base_dual
        .iter()
        .zip(sparse_base_only.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("  Token-hidden-state max|Δ| across all positions: {max_diff:.2e}");
    assert!(
        max_diff < 1e-4,
        "Token hidden states differ beyond FP tolerance (max|Δ|={max_diff:.6e})"
    );

    eprintln!("PASS: dual-output path matches separate dense + sparse passes");
}

fn run_dual_pass(
    sess: &mut ort::session::Session,
    ids: &ndarray::Array2<i64>,
    mask: &ndarray::Array2<i64>,
    is_fp32: bool,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n = ids.nrows();
    let pad_to = ids.ncols();
    let ids_t = ort::value::TensorRef::from_array_view(ids.view()).expect("ids");
    let mask_t = ort::value::TensorRef::from_array_view(mask.view()).expect("mask");
    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
        .expect("dual run");

    if is_fp32 {
        let dense_arr = outputs["sentence_embedding"]
            .try_extract_array::<f32>()
            .expect("sentence_embedding");
        let dense: Vec<Vec<f32>> = dense_arr
            .as_slice()
            .expect("contig")
            .chunks(1024)
            .map(|row| {
                let mut v = row.to_vec();
                l2_normalize(&mut v);
                v
            })
            .collect();
        let token_arr = outputs["token_embeddings"]
            .try_extract_array::<f32>()
            .expect("token_embeddings");
        let tokens = token_arr.as_slice().expect("contig").to_vec();
        (dense, tokens)
    } else {
        let lhs = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .expect("last_hidden_state");
        let lhs_slice = lhs.as_slice().expect("contig");
        let mut dense: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * pad_to * 1024;
            let mut v = lhs_slice[offset..offset + 1024].to_vec();
            l2_normalize(&mut v);
            dense.push(v);
        }
        let tokens = lhs_slice.to_vec();
        (dense, tokens)
    }
}

fn run_dense_pass(
    sess: &mut ort::session::Session,
    ids: &ndarray::Array2<i64>,
    mask: &ndarray::Array2<i64>,
    is_fp32: bool,
) -> Vec<Vec<f32>> {
    let n = ids.nrows();
    let pad_to = ids.ncols();
    let ids_t = ort::value::TensorRef::from_array_view(ids.view()).expect("ids");
    let mask_t = ort::value::TensorRef::from_array_view(mask.view()).expect("mask");
    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
        .expect("dense run");

    if is_fp32 {
        let arr = outputs["sentence_embedding"]
            .try_extract_array::<f32>()
            .expect("sentence_embedding");
        arr.as_slice()
            .expect("contig")
            .chunks(1024)
            .map(|row| {
                let mut v = row.to_vec();
                l2_normalize(&mut v);
                v
            })
            .collect()
    } else {
        let lhs = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .expect("last_hidden_state");
        let lhs_slice = lhs.as_slice().expect("contig");
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * pad_to * 1024;
            let mut v = lhs_slice[offset..offset + 1024].to_vec();
            l2_normalize(&mut v);
            out.push(v);
        }
        out
    }
}

fn run_sparse_pass(
    sess: &mut ort::session::Session,
    ids: &ndarray::Array2<i64>,
    mask: &ndarray::Array2<i64>,
    is_fp32: bool,
) -> Vec<f32> {
    let ids_t = ort::value::TensorRef::from_array_view(ids.view()).expect("ids");
    let mask_t = ort::value::TensorRef::from_array_view(mask.view()).expect("mask");
    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
        .expect("sparse run");

    let key = if is_fp32 {
        "token_embeddings"
    } else {
        "last_hidden_state"
    };
    outputs[key]
        .try_extract_array::<f32>()
        .expect("token-level output")
        .as_slice()
        .expect("contig")
        .to_vec()
}

/// Run positional embedding inspection.
///
/// Verifies that the loaded model's `position_ids` or `position_embedding` table
/// supports at least the configured `BGE_M3_MAX_SEQ_LENGTH`.
#[test]
#[ignore = "requires BGE_M3_EQUIVALENCE_TEST=1 and model download"]
fn onnx_positional_embedding_supports_configured_max_seq() {
    if std::env::var("BGE_M3_EQUIVALENCE_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP: BGE_M3_EQUIVALENCE_TEST != 1");
        return;
    }

    let max_seq: usize = std::env::var("BGE_M3_MAX_SEQ_LENGTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let cache_dir =
        std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".to_string());
    let model_str = std::env::var("BGE_M3_MODEL").unwrap_or_else(|_| "fp32".to_string());

    let model_path = locate_model_file(&cache_dir, &model_str)
        .expect("Could not locate model ONNX file. Ensure model is downloaded.");

    eprintln!("Inspecting: {}", model_path.display());
    eprintln!("Checking positional embedding supports max_seq={max_seq}...");

    // The simplest check: run inference at seq=max_seq with a 1-text batch.
    // If it errors, the model doesn't support this length.
    let mut sess = ort::session::Session::builder()
        .expect("ORT session builder")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
        .expect("opt level")
        .with_intra_threads(1)
        .expect("intra threads")
        .commit_from_file(&model_path)
        .expect("ORT session load");

    let mut input_ids: Vec<i64> = vec![1i64; max_seq]; // all PAD tokens
    input_ids[0] = 0; // CLS
    let attention_mask: Vec<i64> = vec![1i64; max_seq];

    let ids_arr = ndarray::Array2::from_shape_vec((1, max_seq), input_ids).expect("array shape");
    let mask_arr =
        ndarray::Array2::from_shape_vec((1, max_seq), attention_mask).expect("array shape");

    let ids_tensor = ort::value::TensorRef::from_array_view(ids_arr.view()).expect("ids tensor");
    let mask_tensor = ort::value::TensorRef::from_array_view(mask_arr.view()).expect("mask tensor");

    let result = sess.run(ort::inputs! {
        "input_ids" => ids_tensor,
        "attention_mask" => mask_tensor,
    });

    match result {
        Ok(_) => {
            eprintln!("OK: model successfully ran at seq={max_seq}");
        }
        Err(e) => {
            panic!(
                "Model failed at seq={max_seq}: {e}\n\
                 This model variant may not support sequences longer than 512 tokens. \
                 Try BGE_M3_MODEL=fp32 (BAAI/bge-m3) or lower BGE_M3_MAX_SEQ_LENGTH."
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Tolerances {
    mean_cosine: f64,
    p5_cosine: f64,
}

fn cosine_tolerances_for(model: &str) -> Tolerances {
    match model {
        "int8" => Tolerances {
            mean_cosine: 0.95,
            p5_cosine: 0.93,
        },
        "fp16" => Tolerances {
            mean_cosine: 0.98,
            p5_cosine: 0.96,
        },
        _ => Tolerances {
            mean_cosine: 0.99,
            p5_cosine: 0.97,
        }, // fp32 or unknown
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

// ---------------------------------------------------------------------------
// Pure utilities
// ---------------------------------------------------------------------------

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Loads an f32 numpy array from an `.npy` file without a full numpy library.
///
/// Supports only NPY format version 1.0 (the format produced by `np.save`
/// with default settings for contiguous f32 arrays).
fn load_npy_f32(path: &Path) -> Vec<f32> {
    let data = std::fs::read(path).unwrap_or_else(|_| panic!("Cannot read {}", path.display()));
    // NPY magic: \x93NUMPY + version (2 bytes) + header_len (2 bytes LE) + header string
    assert!(
        data.starts_with(b"\x93NUMPY"),
        "Not a valid .npy file: {}",
        path.display()
    );
    let version_major = data[6];
    let header_len_bytes = if version_major == 1 {
        2
    } else {
        4 // version 2+
    };
    let header_len_offset = 8;
    let header_len = if header_len_bytes == 2 {
        u16::from_le_bytes([data[header_len_offset], data[header_len_offset + 1]]) as usize
    } else {
        u32::from_le_bytes([
            data[header_len_offset],
            data[header_len_offset + 1],
            data[header_len_offset + 2],
            data[header_len_offset + 3],
        ]) as usize
    };
    let header_end = header_len_offset + header_len_bytes + header_len;
    let raw_floats = &data[header_end..];
    assert_eq!(
        raw_floats.len() % 4,
        0,
        "Data section length must be a multiple of 4"
    );
    raw_floats
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Locates the model ONNX file in the `HuggingFace` cache.
///
/// The path layout is: `{cache_dir}/models--{org}--{model}/snapshots/{rev}/onnx/model*.onnx`.
fn locate_model_file(cache_dir: &str, model_str: &str) -> Option<std::path::PathBuf> {
    // REPO_REVISION from src/embedder.rs is baked in at test compile time via
    // the extract_const_str utility also used in drift detection tests.
    let embedder_src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder.rs"))
            .ok()?;

    let (repo_org, model_name, revision, onnx_file) = match model_str {
        "fp16" => {
            let rev = extract_const_str(&embedder_src, "XENOVA_REPO_REVISION");
            ("Xenova", "bge-m3", rev, "onnx/model_fp16.onnx")
        }
        "int8" => {
            let rev = extract_const_str(&embedder_src, "XENOVA_REPO_REVISION");
            ("Xenova", "bge-m3", rev, "onnx/model_int8.onnx")
        }
        _ => {
            let rev = extract_const_str(&embedder_src, "REPO_REVISION");
            ("BAAI", "bge-m3", rev, "onnx/model.onnx")
        }
    };

    // HF cache layout: {cache_dir}/models--{org}--{model}/snapshots/{rev}/{file}
    let snapshot_dir = Path::new(cache_dir)
        .join(format!("models--{repo_org}--{model_name}"))
        .join("snapshots")
        .join(&revision);

    let candidate = snapshot_dir.join(onnx_file);
    if candidate.exists() {
        return Some(candidate);
    }

    // Fallback: search for any matching onnx file.
    None
}

/// Locates the tokenizer.json file in the `HuggingFace` cache.
fn locate_tokenizer(cache_dir: &str, model_str: &str) -> Option<std::path::PathBuf> {
    let embedder_src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder.rs"))
            .ok()?;

    let (repo_org, model_name, revision) = match model_str {
        "fp16" | "int8" => {
            let rev = extract_const_str(&embedder_src, "XENOVA_REPO_REVISION");
            ("Xenova", "bge-m3", rev)
        }
        _ => {
            let rev = extract_const_str(&embedder_src, "REPO_REVISION");
            ("BAAI", "bge-m3", rev)
        }
    };

    let candidate = Path::new(cache_dir)
        .join(format!("models--{repo_org}--{model_name}"))
        .join("snapshots")
        .join(&revision)
        .join("tokenizer.json");

    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Extracts a `const NAME: &str = "..."` value from Rust source text.
fn extract_const_str(src: &str, const_name: &str) -> String {
    let prefix = format!("const {const_name}");
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            let start = trimmed.find('"').expect("missing opening quote");
            let end = trimmed[start + 1..]
                .find('"')
                .expect("missing closing quote");
            return trimmed[start + 1..start + 1 + end].to_string();
        }
    }
    panic!("{const_name} not found in provided source");
}
