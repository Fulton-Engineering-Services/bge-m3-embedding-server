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

//! Public DTOs and the internal `EmbedRequest` enum exchanged between the
//! pool and the worker threads.

use anyhow::Result;
use tokio::sync::oneshot;

/// Sparse embedding output from the BGE-M3 sparse-linear projection layer.
///
/// Represents a document as a sparse vector over the tokenizer vocabulary.
/// Token IDs with zero ReLU-gated score are omitted.
#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    /// Sorted vocabulary token IDs with non-zero ReLU-gated weight.
    pub indices: Vec<usize>,
    /// Corresponding ReLU-gated projection scores, in the same order as `indices`.
    pub values: Vec<f32>,
}

/// Paired dense + sparse embeddings produced from a single forward pass.
#[derive(Debug, Clone)]
pub struct DualEmbedding {
    pub dense: Vec<f32>,
    pub sparse: SparseEmbedding,
}

/// OS headroom reserved for kernel, stack, ORT arena, and other non-model
/// allocations. Subtracted from available memory before computing
/// per-worker workspace.
pub(crate) const OS_HEADROOM_BYTES: usize = 256 * 1024 * 1024; // 256 MiB

/// Per-request diagnostic statistics captured inside the worker and forwarded
/// to the handler layer for inclusion in the completion log event.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmbedStats {
    /// Number of bin-packed chunks the batch was split into.
    pub chunks: usize,
    /// Maximum tokenized sequence length across all chunks.
    pub max_chunk_seq: usize,
    /// Total token-positions processed (sum of `seq_len` for all inputs).
    pub total_token_positions: usize,
    /// Time spent tokenizing all inputs (milliseconds).
    pub tokenize_ms: u64,
    /// Total time spent in ORT `session.run()` across all chunks (milliseconds).
    pub inference_ms: u64,
    /// Minimum token sequence length across all inputs in the batch.
    pub seq_len_min: usize,
    /// Maximum token sequence length across all inputs in the batch.
    pub seq_len_max: usize,
    /// Mean token sequence length across all inputs (integer, truncated).
    pub seq_len_mean: usize,
    /// 95th-percentile token sequence length across all inputs in the batch.
    ///
    /// Index is `(n * 95) / 100` on a sorted copy of the per-input lengths.
    pub seq_len_p95: usize,
}

pub(crate) enum EmbedRequest {
    /// Dense (float32) embedding inference on a batch of texts.
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<(Vec<Vec<f32>>, EmbedStats)>>,
    },
    /// Sparse (SPLADE-style) embedding inference on a batch of texts.
    Sparse {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<(Vec<SparseEmbedding>, EmbedStats)>>,
    },
    /// Computes dense and sparse embeddings from a single forward pass per chunk.
    Both {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<(Vec<DualEmbedding>, EmbedStats)>>,
    },
    /// Internal: used during startup probe to run a single batch and measure
    /// peak RSS delta. Workers only process this before `ready` is set.
    Probe {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<ProbeResult>>,
    },
    /// Adaptive background warmup: asks a worker to compile (or confirm as
    /// cached) the TRT engine for `(batch, seq)`.  The worker replies on
    /// `ack` with the compile duration in milliseconds, or an error if the
    /// shape failed.  Only meaningful on TRT EP; on CPU/CUDA workers the
    /// worker returns `Ok(0)` immediately.
    AdaptiveWarmup {
        batch: usize,
        seq: usize,
        ack: oneshot::Sender<anyhow::Result<u64>>,
    },
}

/// Sender half of the JIT-suspect channel.
///
/// Workers hold an optional clone of this sender and call `try_send`
/// (non-blocking, drops if full) after any inference whose `inference_ms`
/// equals or exceeds the TRT cache-hit threshold.
pub(crate) type JitSuspectSender = tokio::sync::mpsc::Sender<(usize, usize)>;

/// Result of a single probe `session.run()` call.
pub(crate) struct ProbeResult {
    /// Process RSS (bytes) measured immediately before `session.run()`.
    pub rss_before: usize,
    /// Process RSS (bytes) measured immediately after `session.run()`.
    pub rss_after: usize,
}
