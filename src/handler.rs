//! HTTP handlers for the embedding service.
//!
//! Submodules:
//! - `common`: shared input validation and readiness helpers.
//! - `dense`: `POST /v1/embeddings` (OpenAI-compatible dense embeddings).
//! - `sparse`: `POST /v1/sparse-embeddings` (BGE-M3 SPLADE-style sparse embeddings).
//! - `both`: `POST /v1/embeddings:both` (paired dense + sparse output in one pass).
//! - `health`: `GET /health` (readiness + tuning details).
//! - `models`: `GET /v1/models` (fleet discovery).

mod both;
mod common;
mod dense;
mod health;
mod models;
mod sparse;

pub use both::both_embeddings;
pub use dense::dense_embeddings;
pub use health::health;
pub use models::models;
pub use sparse::sparse_embeddings;

#[cfg(test)]
mod tests;
