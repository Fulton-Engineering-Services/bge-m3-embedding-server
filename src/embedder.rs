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
//! - `trt_cache`: `TensorRT` engine-cache path construction, inspection, and
//!   durability (fsync after compile).
//! - `trt_warmup`: `TensorRT` engine pre-warming during worker startup.
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
mod trt_cache;
mod trt_warmup;
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
