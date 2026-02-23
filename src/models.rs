use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub index: usize,
    pub sparse_values: SparseValues,
}

#[derive(Serialize)]
pub struct SparseValues {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}
