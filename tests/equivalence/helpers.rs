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

//! Shared utilities for the equivalence integration test:
//! - fixture file location
//! - cosine similarity / L2 normalization
//! - NPY parsing
//! - HF cache layout resolution
//! - `REPO_REVISION` extraction from server source

use std::path::Path;

pub(crate) fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("equivalence")
}

#[derive(Debug)]
pub(crate) struct Tolerances {
    pub mean_cosine: f64,
    pub p5_cosine: f64,
}

pub(crate) fn cosine_tolerances_for(model: &str) -> Tolerances {
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

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) fn load_npy_f32(path: &Path) -> Vec<f32> {
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
pub(crate) fn locate_model_file(cache_dir: &str, model_str: &str) -> Option<std::path::PathBuf> {
    // REPO_REVISION constants live in src/embedder/model_files.rs after the
    // source layout refactor. Read the file textually and scrape the value
    // — same approach used by the drift-detection tests.
    let embedder_src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder/model_files.rs"),
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
pub(crate) fn locate_tokenizer(cache_dir: &str, model_str: &str) -> Option<std::path::PathBuf> {
    let embedder_src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/embedder/model_files.rs"),
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
pub(crate) fn extract_const_str(src: &str, const_name: &str) -> String {
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
