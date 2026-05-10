//! Worker-pool–driven BGE-M3 embedding service.
//!
//! Submodules:
//! - `types`: public DTOs and the internal `EmbedRequest` enum.
//! - `error`: small `ort::Error → anyhow::Error` adapter.
//! - `model_files`: hf-hub download / cache layout for the ONNX model files.
//! - `tokenize`: tokenizer load + no-pad tokenization + chunk-array build.
//! - `session`: ORT execution-provider config and session loading.
//! - `math`: pure dense/sparse math helpers (testable without ORT).
//! - `dense`: dense embedding pipeline.
//! - `sparse`: BGE-M3 SPLADE-style sparse embedding pipeline.
//! - `dual`: paired dense + sparse embedding pipeline (one forward pass).
//! - `worker`: blocking worker thread, request dispatch, probe wiring.
//! - `pool`: `EmbedPool` async wrapper and test helpers.

mod dense;
mod dual;
mod error;
mod math;
mod model_files;
mod pool;
mod session;
mod sparse;
mod tokenize;
mod types;
mod worker;

pub use pool::EmbedPool;
pub(crate) use types::OS_HEADROOM_BYTES;
pub(crate) use worker::WorkerConfig;

// `SparseEmbedding` is referenced by tests via `crate::embedder::SparseEmbedding`,
// but is not used outside the module in the non-test build. The cfg(test) gate
// keeps the binary's unused-import lint clean while preserving the call-site path
// for tests.
#[cfg(test)]
pub(crate) use types::SparseEmbedding;
