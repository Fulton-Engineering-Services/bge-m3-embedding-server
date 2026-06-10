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

//! Engine propagation broadcast drain and post-inference cache-miss signaling.

use super::trt_retry::CHUNK_CACHE_HIT_THRESHOLD_MS;
use crate::embedder::types::{EmbedStats, JitSuspectSender};

/// Emits the `chunk_run` INFO event and, on a cache miss, notifies both the
/// JIT-suspect channel (adaptive warmup scheduling) and the engine propagation
/// broadcast channel (peer worker fast disk-load).
///
/// Returns `Some((batch_len, max_chunk_seq))` when a shape was broadcast on
/// the engine propagation channel.  The call site MUST insert this shape into
/// `warmed_local` so the originating worker self-skips its own broadcast on
/// the next `drain_engine_propagation` iteration (COR-1).
///
/// # 5000 ms threshold heuristic (COR-10)
///
/// `CHUNK_CACHE_HIT_THRESHOLD_MS` (5 s) is a **heuristic** proxy for "TRT
/// engine JIT compile occurred", not a semantic guarantee.  False negatives
/// are possible for fast-JIT small shapes; false positives are impossible
/// because a cache-hit path never exceeds ~100 ms.  The trade-off is
/// acceptable: the worst outcome of a false negative is that the adaptive
/// warmup task eventually resubmits the shape on the next real cache miss.
pub(super) fn log_inference_complete(
    stats: &EmbedStats,
    worker_id: usize,
    _route: &'static str,
    jit_suspect_tx: Option<&JitSuspectSender>,
    engine_propagation_tx: Option<&tokio::sync::broadcast::Sender<(usize, usize)>>,
    batch_len: usize,
) -> Option<(usize, usize)> {
    let cache_hit = stats.inference_ms < CHUNK_CACHE_HIT_THRESHOLD_MS;
    tracing::info!(
        target: "bge_m3_embedding_server::trt_shape",
        worker_id,
        chunk_batch = batch_len,
        chunk_max_seq = stats.max_chunk_seq,
        inference_ms = stats.inference_ms,
        cache_hit,
        "chunk_run"
    );
    if !cache_hit {
        if let Some(tx) = jit_suspect_tx {
            let _ = tx.try_send((batch_len, stats.max_chunk_seq));
        }
        if let Some(tx) = engine_propagation_tx {
            let _ = tx.send((batch_len, stats.max_chunk_seq));
            return Some((batch_len, stats.max_chunk_seq));
        }
    }
    None
}

/// Drains pending broadcast notifications and runs `trt_prewarm` for each
/// new shape.
///
/// Called at the start of each worker loop iteration (between requests) so
/// peers eagerly warm their in-memory TRT profile before the next real
/// request for a new shape arrives.
///
/// `warmed_local` tracks shapes already warmed by this worker in the current
/// session.  The originating worker self-skips on subsequent drains because
/// `log_inference_complete` inserts the broadcast shape into `warmed_local`
/// at the call site before returning control to the request loop.
pub(super) fn drain_engine_propagation<F>(
    rx: &mut tokio::sync::broadcast::Receiver<(usize, usize)>,
    warmed_local: &mut std::collections::HashSet<(usize, usize)>,
    worker_id: usize,
    mut prewarm: F,
) where
    F: FnMut((usize, usize)),
{
    loop {
        match rx.try_recv() {
            Ok(shape) => {
                if warmed_local.insert(shape) {
                    prewarm(shape);
                }
            }
            Err(
                tokio::sync::broadcast::error::TryRecvError::Empty
                | tokio::sync::broadcast::error::TryRecvError::Closed,
            ) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(
                    worker_id,
                    lagged = n,
                    "engine_propagation: broadcast lagged; some shapes missed"
                );
                // Continue draining; missed shapes will be re-broadcast on
                // the next slow-inference event for that shape.
            }
        }
    }
}
