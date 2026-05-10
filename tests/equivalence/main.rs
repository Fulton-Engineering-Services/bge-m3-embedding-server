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
//!
//! ## File layout
//!
//! Tests are organised under `tests/equivalence/`:
//! - `main.rs`: harness, env-var gating, smoke tests, ONNX positional check.
//! - `helpers.rs`: fixture loading, cosine similarity, NPY parsing, model
//!   path resolution, `REPO_REVISION` extraction.
//! - `dense.rs`: dense equivalence cases (`equivalence_all_seq_lengths`).
//! - `sparse.rs`: reserved for future sparse-only equivalence cases.
//! - `hybrid.rs`: dual-output (`/v1/embeddings:both`) cases.

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

mod dense;
mod helpers;
mod hybrid;
mod sparse;

use std::path::Path;

use crate::helpers::{extract_const_str, fixture_dir, locate_model_file};

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

    // Extract REPO_REVISION from the server source. The pinned revisions
    // moved to src/embedder/model_files.rs in the source layout refactor.
    let embedder_src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder/model_files.rs"),
    )
    .expect("src/embedder/model_files.rs should be readable");

    let revision = extract_const_str(&embedder_src, "REPO_REVISION");

    assert_eq!(
        manifest_rev, revision,
        "manifest model_revision ({manifest_rev}) must match REPO_REVISION in \
         src/embedder/model_files.rs ({revision}). Regenerate fixtures with \
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
// 3. ONNX positional embedding inspection
// ---------------------------------------------------------------------------

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
