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

use ndarray::Array1;
use std::sync::OnceLock;

/// Bundled sparse-linear projection weights for BGE-M3 sparse embedding.
///
/// # Provenance
///
/// Extracted from the fastembed-rs crate (v4) which bundles the same file
/// from the BAAI/bge-m3 checkpoint. The weights implement the sparse-linear
/// layer described in the BGE-M3 paper: a single linear projection
/// `hidden_size → 1` that maps each token's 1024-d hidden state to a scalar
/// relevance score, followed by `ReLU` activation and max-pooling by vocab ID.
///
/// - **Source checkpoint**: `BAAI/bge-m3` (HF commit `5617a9f61b02800`)
/// - **Tensors**: `weight` shape `[1024]` (F32), `bias` scalar (F32)
/// - **File SHA-256**: `a2601321f01abbb696d171a58a65ff35be1603d9cbc22c647dfe34b4568dd690`
/// - **File size**: 4,236 bytes
static WEIGHTS_BYTES: &[u8] = include_bytes!("sparse_linear.safetensors");

static SPARSE_LINEAR: OnceLock<(Array1<f32>, f32)> = OnceLock::new();

/// Returns the sparse-linear projection weights used by BGE-M3 sparse embedding.
///
/// The safetensors file contains a weight vector `[1024]` and a scalar bias.
/// Parsed once on first call and cached for the lifetime of the process.
pub(crate) fn sparse_linear() -> &'static (Array1<f32>, f32) {
    SPARSE_LINEAR.get_or_init(|| {
        let tensors = safetensors::SafeTensors::deserialize(WEIGHTS_BYTES)
            .expect("embedded sparse_linear.safetensors must be valid");

        let weight_view = tensors
            .tensor("weight")
            .expect("sparse_linear must contain 'weight' tensor");
        let bias_view = tensors
            .tensor("bias")
            .expect("sparse_linear must contain 'bias' tensor");

        let weight_data = weight_view.data();
        assert_eq!(
            weight_data.len() % 4,
            0,
            "weight tensor byte length must be a multiple of 4, got {}",
            weight_data.len()
        );
        let weight: Vec<f32> = weight_data
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
        (Array1::from(weight), bias)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_linear_loads_correct_shape() {
        let (weight, bias) = sparse_linear();
        assert_eq!(weight.len(), 1024);
        // Known bias value from BAAI/bge-m3 sparse_linear.safetensors
        assert!(
            (*bias - 0.045_196_53).abs() < 1e-6,
            "bias should be ~0.04520, got {bias}"
        );
        assert!(bias.is_finite(), "bias must be finite");
        // TST-6: verify all weights are finite and not all-zero
        assert!(
            weight.iter().all(|w| w.is_finite()),
            "all weight elements must be finite"
        );
        assert!(
            weight.iter().any(|&w| w != 0.0),
            "weight vector must not be all-zero"
        );
    }

    #[test]
    fn sparse_linear_is_idempotent() {
        let a = sparse_linear();
        let b = sparse_linear();
        assert!(std::ptr::eq(a, b), "should return the same cached ref");
    }

    #[test]
    fn bundled_file_is_valid_safetensors() {
        // Verify the embedded bytes parse without panic and contain expected tensors.
        let tensors = safetensors::SafeTensors::deserialize(WEIGHTS_BYTES)
            .expect("WEIGHTS_BYTES must be valid safetensors");
        assert!(tensors.tensor("weight").is_ok(), "must contain 'weight'");
        assert!(tensors.tensor("bias").is_ok(), "must contain 'bias'");
    }

    #[test]
    fn bundled_file_size_matches() {
        // Size pinned to detect accidental replacement or corruption.
        assert_eq!(WEIGHTS_BYTES.len(), 4236, "expected 4,236 bytes");
    }

    #[test]
    fn bundled_file_sha256_matches() {
        use sha2::Digest;
        use std::fmt::Write;
        // Documented provenance hash — any change to the bundled file must update this.
        const EXPECTED_SHA256: &str =
            "a2601321f01abbb696d171a58a65ff35be1603d9cbc22c647dfe34b4568dd690";
        let digest = {
            let mut hasher = sha2::Sha256::new();
            hasher.update(WEIGHTS_BYTES);
            hasher.finalize()
        };
        let hex = digest.iter().fold(String::new(), |mut s, b| {
            write!(s, "{b:02x}").expect("hex write");
            s
        });
        assert_eq!(
            hex, EXPECTED_SHA256,
            "bundled sparse_linear.safetensors SHA-256 mismatch"
        );
    }
}
