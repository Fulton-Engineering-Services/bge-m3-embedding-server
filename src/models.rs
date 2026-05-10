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
    pub input: TextInput,
    pub model: Option<String>,
}

/// Top-level response for the dense embeddings endpoint (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct DenseResponse {
    pub object: &'static str,
    pub model: &'static str,
    pub data: Vec<DenseEmbeddingData>,
    pub usage: Usage,
}

/// Per-document dense embedding entry.
#[derive(Debug, Serialize)]
pub struct DenseEmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// Token usage counters.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Sparse embedding types
// ---------------------------------------------------------------------------

/// Request body for the sparse embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct SparseRequest {
    pub input: TextInput,
}

/// Top-level response for the sparse embeddings endpoint.
#[derive(Debug, Serialize)]
pub struct SparseResponse {
    pub data: Vec<SparseEmbeddingData>,
}

/// Per-document sparse embedding entry.
#[derive(Debug, Serialize)]
pub struct SparseEmbeddingData {
    pub index: usize,
    pub sparse_values: SparseValues,
}

/// Parallel arrays of token indices and their weights.
#[derive(Debug, Serialize)]
pub struct SparseValues {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Dual embedding types (single forward pass yielding both dense and sparse)
// ---------------------------------------------------------------------------

/// Request body for the unified dense + sparse embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct DualRequest {
    pub input: TextInput,
    /// Accepted for `OpenAI` API compatibility; always uses BGE-M3.
    pub model: Option<String>,
}

/// Top-level response for the unified dense + sparse embeddings endpoint.
#[derive(Debug, Serialize)]
pub struct DualResponse {
    pub object: &'static str,
    pub model: &'static str,
    pub data: Vec<DualEmbeddingData>,
    pub usage: Usage,
}

/// Per-document paired dense + sparse embedding entry.
#[derive(Debug, Serialize)]
pub struct DualEmbeddingData {
    pub index: usize,
    pub embedding: Vec<f32>,
    pub sparse_values: SparseValues,
}

// ---------------------------------------------------------------------------
// Models list types
// ---------------------------------------------------------------------------

/// Top-level response for GET /v1/models (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

/// A single model entry.
#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: &'static str,
    pub object: &'static str,
}

#[cfg(test)]
mod tests;
