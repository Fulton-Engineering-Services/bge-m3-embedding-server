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

//! Integration test for `broadcast_engine_ready` — verifies that a
//! successful adaptive-warmup compile propagates the compiled shape to all
//! engine-propagation subscribers via the pool's broadcast channel.

use tokio::sync::mpsc;

use super::super::{run_adaptive_warmup_loop, AdaptiveWarmupConfig};
use crate::embedder::pool::EmbedPool;

// ─── broadcast_engine_ready integration ──────────────────────────────────────

/// After a successful adaptive warmup compile, the shape must be broadcast to
/// all engine-propagation subscribers.
///
/// Uses `EmbedPool::for_propagation_test` to wire a known broadcast sender into
/// the pool. The test subscribes a receiver before the loop runs and asserts
/// the shape arrives after the compile succeeds.
#[tokio::test]
async fn successful_adaptive_warmup_broadcasts_shape_to_pool() {
    let (bcast_tx, mut bcast_rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);

    // Build a pool backed by fixed responses (AdaptiveWarmup returns Ok(0))
    // with the broadcast sender wired in.
    let pool = EmbedPool::for_propagation_test(vec![], vec![], bcast_tx);

    // Prepare a JIT-suspect channel pre-seeded with the shape to compile.
    let (jit_tx, jit_rx) = mpsc::channel::<(usize, usize)>(64);
    jit_tx.send((4, 512)).await.unwrap();
    drop(jit_tx);

    let cfg = AdaptiveWarmupConfig {
        enabled: true,
        quiet_secs: 0,
        max_shapes_per_hour: 12,
    };

    // Run the loop as a background task.
    let loop_handle = tokio::spawn(run_adaptive_warmup_loop(cfg, pool, jit_rx));

    // Give the loop time to drain the suspect channel, fire the compile, and
    // call broadcast_engine_ready. Try yields first, fall back to a short sleep.
    let mut received = None;
    for _ in 0..20 {
        tokio::task::yield_now().await;
        if let Ok(shape) = bcast_rx.try_recv() {
            received = Some(shape);
            break;
        }
    }
    if received.is_none() {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        received = bcast_rx.try_recv().ok();
    }
    loop_handle.abort();

    assert_eq!(
        received,
        Some((4, 512)),
        "broadcast_engine_ready must be called after a successful adaptive warmup compile"
    );
}
