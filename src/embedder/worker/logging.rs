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

//! Observability helpers for abandoned in-flight requests.

use anyhow::Result;

/// Emits a `WARN` if the oneshot reply receiver has been dropped while the
/// worker was busy with `embed_*` — meaning the client (often the router's
/// hedged race) disconnected after dispatch and the inference work is now
/// discarded. We can't interrupt ORT `session.run()` mid-call, so this is
/// observability only: operators can correlate `inference_ms` and `chunks`
/// across requests to size the router's cancellation budget.
///
/// The reply is sent unconditionally by the caller after this returns; the
/// channel layer will silently drop the value if the receiver is gone.
use crate::embedder::types::EmbedStats;

pub(super) fn log_if_abandoned_mid_flight<T>(
    reply: &tokio::sync::oneshot::Sender<Result<(T, EmbedStats)>>,
    route: &'static str,
    worker_id: usize,
    result: &Result<(T, EmbedStats)>,
    inference_ms: u128,
) {
    if !reply.is_closed() {
        return;
    }
    let (chunks, max_chunk_seq, total_token_positions) = match result {
        Ok((_, stats)) => (
            Some(stats.chunks),
            Some(stats.max_chunk_seq),
            Some(stats.total_token_positions),
        ),
        Err(_) => (None, None, None),
    };
    let inference_ms_u64 = u64::try_from(inference_ms).unwrap_or(u64::MAX);
    tracing::warn!(
        worker_id,
        route,
        inference_ms_so_far = inference_ms_u64,
        chunks,
        max_chunk_seq,
        total_token_positions,
        "request abandoned by client during inference (work discarded; \
         ORT session.run() cannot be interrupted mid-call)"
    );
}
