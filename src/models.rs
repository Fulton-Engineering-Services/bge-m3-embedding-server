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

//! Request and response model types for the embedding API endpoints.
//!
//! Dense types are OpenAI-compatible. Sparse and dual types are BGE-M3
//! specific; they extend the same request shape with additional output fields.

use serde::{Deserialize, Deserializer, Serialize};

// ---------------------------------------------------------------------------
// TextInput — accepts either a single string or an array of strings
// ---------------------------------------------------------------------------

/// Newtype wrapping a `Vec<String>` that deserializes from either
/// `"a single string"` or `["array", "of", "strings"]`.
#[derive(Debug, PartialEq)]
pub struct TextInput(pub Vec<String>);

impl<'de> Deserialize<'de> for TextInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrArray {
            Single(String),
            Multiple(Vec<String>),
        }

        match StringOrArray::deserialize(deserializer)? {
            StringOrArray::Single(s) => Ok(TextInput(vec![s])),
            StringOrArray::Multiple(v) => Ok(TextInput(v)),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense embedding types
// ---------------------------------------------------------------------------

/// Request body for the dense embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct DenseRequest {
    /// Input texts to generate embeddings for.
    pub input: TextInput,
    /// Accepted for `OpenAI` API compatibility; value is ignored — always uses BGE-M3.
    pub model: Option<String>,
}

/// Top-level response for the dense embeddings endpoint (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct DenseResponse {
    /// Always `"list"`.
    pub object: &'static str,
    /// Always `"bge-m3"`.
    pub model: &'static str,
    /// Per-document dense embedding entries, one per input text.
    pub data: Vec<DenseEmbeddingData>,
    /// Aggregate token usage estimates.
    pub usage: Usage,
}

/// Per-document dense embedding entry.
#[derive(Debug, Serialize)]
pub struct DenseEmbeddingData {
    /// Always `"embedding"`.
    pub object: &'static str,
    /// Zero-based position of this document in the request's input array.
    pub index: usize,
    /// L2-normalized 1024-dimensional dense embedding vector.
    pub embedding: Vec<f32>,
}

/// Token usage counters.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Estimated input token count (approximated as `chars / 4 + 1` per text).
    pub prompt_tokens: usize,
    /// Same as `prompt_tokens` — embedding models have no completion tokens.
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Sparse embedding types
// ---------------------------------------------------------------------------

/// Request body for the sparse embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct SparseRequest {
    /// Input texts to generate sparse embeddings for.
    pub input: TextInput,
}

/// Top-level response for the sparse embeddings endpoint.
#[derive(Debug, Serialize)]
pub struct SparseResponse {
    /// Per-document sparse embedding entries, one per input text.
    pub data: Vec<SparseEmbeddingData>,
}

/// Per-document sparse embedding entry.
#[derive(Debug, Serialize)]
pub struct SparseEmbeddingData {
    /// Zero-based position of this document in the request's input array.
    pub index: usize,
    /// Non-zero vocabulary token weights for this document.
    pub sparse_values: SparseValues,
}

/// Parallel arrays of token indices and their weights.
#[derive(Debug, Serialize)]
pub struct SparseValues {
    /// Sorted vocabulary token IDs with non-zero ReLU-gated weight.
    pub indices: Vec<u32>,
    /// ReLU-gated weights corresponding to each index, in the same order.
    pub values: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Dual embedding types (single forward pass yielding both dense and sparse)
// ---------------------------------------------------------------------------

/// Request body for the unified dense + sparse embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct DualRequest {
    /// Input texts to generate dense and sparse embeddings for.
    pub input: TextInput,
    /// Accepted for `OpenAI` API compatibility; always uses BGE-M3.
    pub model: Option<String>,
}

/// Top-level response for the unified dense + sparse embeddings endpoint.
#[derive(Debug, Serialize)]
pub struct DualResponse {
    /// Always `"list"`.
    pub object: &'static str,
    /// Always `"bge-m3"`.
    pub model: &'static str,
    /// Per-document paired dense + sparse embedding entries.
    pub data: Vec<DualEmbeddingData>,
    /// Aggregate token usage estimates.
    pub usage: Usage,
}

/// Per-document paired dense + sparse embedding entry.
#[derive(Debug, Serialize)]
pub struct DualEmbeddingData {
    /// Zero-based position of this document in the request's input array.
    pub index: usize,
    /// L2-normalized 1024-dimensional dense embedding vector.
    pub embedding: Vec<f32>,
    /// Non-zero vocabulary token weights for this document.
    pub sparse_values: SparseValues,
}

// ---------------------------------------------------------------------------
// Models list types
// ---------------------------------------------------------------------------

/// Top-level response for GET /v1/models (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    /// Always `"list"`.
    pub object: &'static str,
    /// List of available model entries.
    pub data: Vec<ModelEntry>,
}

/// A single model entry.
#[derive(Debug, Serialize)]
pub struct ModelEntry {
    /// Model identifier — always `"bge-m3"`.
    pub id: &'static str,
    /// Always `"model"`.
    pub object: &'static str,
}

#[cfg(test)]
mod tests;
