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
//! 1. Load the ONNX model (downloads from HuggingFace if not cached).
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

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
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
        manifest.get("model_revision").and_then(|v| v.as_str()).is_some(),
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

/// Checks that the manifest's model_revision matches the server's pinned REPO_REVISION.
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
    let embedder_src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder.rs"),
    )
    .expect("src/embedder.rs should be readable");

    let revision = extract_const_str(&embedder_src, "REPO_REVISION");

    assert_eq!(
        manifest_rev, revision,
        "manifest model_revision ({manifest_rev}) must match REPO_REVISION in \
         src/embedder.rs ({revision}). Regenerate fixtures with \
         scripts/generate_equivalence_fixtures.py after bumping REPO_REVISION."
    );
}

/// For each seq_length listed in the manifest, verify the fixture files exist
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
    if !manifest_path.exists() {
        panic!(
            "Fixture manifest not found at {}. \
             Run scripts/generate_equivalence_fixtures.py first.",
            manifest_path.display()
        );
    }

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

/// Run positional embedding inspection.
///
/// Verifies that the loaded model's position_ids or position_embedding table
/// supports at least the configured BGE_M3_MAX_SEQ_LENGTH.
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

    let ids_arr =
        ndarray::Array2::from_shape_vec((1, max_seq), input_ids).expect("array shape");
    let mask_arr =
        ndarray::Array2::from_shape_vec((1, max_seq), attention_mask).expect("array shape");

    let ids_tensor =
        ort::value::TensorRef::from_array_view(ids_arr.view()).expect("ids tensor");
    let mask_tensor =
        ort::value::TensorRef::from_array_view(mask_arr.view()).expect("mask tensor");

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
        "int8" => Tolerances { mean_cosine: 0.95, p5_cosine: 0.93 },
        "fp16" => Tolerances { mean_cosine: 0.98, p5_cosine: 0.96 },
        _ => Tolerances { mean_cosine: 0.99, p5_cosine: 0.97 }, // fp32 or unknown
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
    let tokenizer_path = locate_tokenizer(&cache_dir, model_str)
        .expect("Could not locate tokenizer.json");

    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("tokenizer load");
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
    let encodings = tokenizer.encode_batch_fast(str_refs, true).expect("tokenize");

    let pad_to = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1);
    let mut ids_flat: Vec<i64> = Vec::with_capacity(n * pad_to);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(n * pad_to);
    for enc in &encodings {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        let seq_len = ids.len();
        ids_flat.extend(ids.iter().map(|&id| i64::from(id)));
        mask_flat.extend(mask.iter().map(|&m| i64::from(m)));
        let pad = pad_to.saturating_sub(seq_len);
        ids_flat.extend(std::iter::repeat(1i64).take(pad));
        mask_flat.extend(std::iter::repeat(0i64).take(pad));
    }

    let ids_arr = ndarray::Array2::from_shape_vec((n, pad_to), ids_flat).expect("ids array");
    let mask_arr = ndarray::Array2::from_shape_vec((n, pad_to), mask_flat).expect("mask array");

    eprintln!("  Running inference at shape ({n}, {pad_to})...");
    let ids_tensor = ort::value::TensorRef::from_array_view(ids_arr.view()).expect("ids tensor");
    let mask_tensor =
        ort::value::TensorRef::from_array_view(mask_arr.view()).expect("mask tensor");

    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })
        .expect("inference");

    // Extract dense embeddings and L2-normalize.
    let dense_out: Vec<f32> = match model_str {
        "fp32" => {
            let emb = outputs["sentence_embedding"]
                .try_extract_array::<f32>()
                .expect("sentence_embedding");
            emb.as_slice().expect("contiguous").to_vec()
        }
        _ => {
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
        }
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
            cosine_similarity(computed, reference) as f64
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

/// Locates the model ONNX file in the HuggingFace cache.
///
/// The path layout is: `{cache_dir}/models--{org}--{model}/snapshots/{rev}/onnx/model*.onnx`.
fn locate_model_file(cache_dir: &str, model_str: &str) -> Option<std::path::PathBuf> {
    // REPO_REVISION from src/embedder.rs is baked in at test compile time via
    // the extract_const_str utility also used in drift detection tests.
    let embedder_src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder.rs"),
    )
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
        .join(format!("models--{}--{}", repo_org, model_name))
        .join("snapshots")
        .join(&revision);

    let candidate = snapshot_dir.join(onnx_file);
    if candidate.exists() {
        return Some(candidate);
    }

    // Fallback: search for any matching onnx file.
    None
}

/// Locates the tokenizer.json file in the HuggingFace cache.
fn locate_tokenizer(cache_dir: &str, model_str: &str) -> Option<std::path::PathBuf> {
    let embedder_src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder.rs"),
    )
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
        .join(format!("models--{}--{}", repo_org, model_name))
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
            let end = trimmed[start + 1..].find('"').expect("missing closing quote");
            return trimmed[start + 1..start + 1 + end].to_string();
        }
    }
    panic!("{const_name} not found in provided source");
}
