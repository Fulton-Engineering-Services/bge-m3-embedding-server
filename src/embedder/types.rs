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

#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    pub indices: Vec<usize>,
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
}

pub(crate) enum EmbedRequest {
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<(Vec<Vec<f32>>, EmbedStats)>>,
    },
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
}

/// Result of a single probe `session.run()` call.
pub(crate) struct ProbeResult {
    pub rss_before: usize,
    pub rss_after: usize,
}
