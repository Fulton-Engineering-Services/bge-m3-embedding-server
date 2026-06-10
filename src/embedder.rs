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
//! - `sm_detect`: per-device GPU compute-capability detection (`smXY`) used
//!   to filter the TRT engine cache by the worker's own SM.
//! - `math`: pure dense/sparse math helpers (testable without ORT).
//! - `dense`: dense embedding pipeline.
//! - `sparse`: BGE-M3 SPLADE-style sparse embedding pipeline.
//! - `dual`: paired dense + sparse embedding pipeline (one forward pass).
//! - `jit_guard`: in-band `TensorRT` JIT admission guard that refuses chunks
//!   whose sequence length is dangerous and uncovered by the warmed engine
//!   profile, preventing the process-killing pathological autotuner allocation.
//! - `trt_cache`: `TensorRT` engine-cache path construction, inspection, and
//!   durability (fsync after compile). Submodules: `paths`, `inspect`,
//!   `enumerate`, `prewarm_log`, `fsync`.
//! - `trt_cache_gc` (feature `cache-gc`, off by default): destructive
//!   stale-SM engine plan garbage collection — only present in dedicated
//!   maintenance / dev binaries.
//! - `trt_warmup`: `TensorRT` engine pre-warming during worker startup.
//! - `worker`: blocking worker thread, request dispatch, probe wiring.
//!   Submodules: `config`, `guard`, `trt_retry`, `propagation`, `probe`,
//!   `prewarm_strict`, `startup`, `run`, `dispatch`, `logging`.
//! - `pool`: `EmbedPool` async wrapper and test helpers.
//! - `adaptive_warmup`: adaptive in-process background warmup loop for TRT
//!   engine cache miss recovery.

pub(crate) mod adaptive_warmup;
mod dense;
mod dual;
mod error;
pub(crate) mod jit_guard;
mod math;
mod model_files;
mod pool;
mod session;
pub(crate) mod sm_detect;
mod sparse;
mod tokenize;
pub(crate) mod trt_cache;
#[cfg(feature = "cache-gc")]
pub(crate) mod trt_cache_gc;
mod trt_warmup;
mod types;
mod worker;

pub use pool::EmbedPool;
pub(crate) use types::{JitSuspectSender, OS_HEADROOM_BYTES};
pub(crate) use worker::WorkerConfig;

// `SparseEmbedding` is referenced by tests via `crate::embedder::SparseEmbedding`,
// but is not used outside the module in the non-test build. The cfg(test) gate
// keeps the binary's unused-import lint clean while preserving the call-site path
// for tests.
#[cfg(test)]
pub(crate) use types::SparseEmbedding;
