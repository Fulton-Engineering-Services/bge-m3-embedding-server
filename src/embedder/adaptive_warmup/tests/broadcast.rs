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

//! Integration tests for `broadcast_engine_ready` — verifies that:
//! - A TRT compile (`compile_ms` > 0) propagates the shape to subscribers.
//! - A non-TRT no-op (`compile_ms` == 0) does NOT broadcast (COR-7).

use tokio::sync::mpsc;

use super::super::{AdaptiveWarmupConfig, run_adaptive_warmup_loop};
use crate::embedder::pool::EmbedPool;

// ─── broadcast_engine_ready integration ──────────────────────────────────────

/// After a successful TRT compile (`compile_ms` > 0), the shape must be
/// broadcast to all engine-propagation subscribers.
///
/// Uses `EmbedPool::for_trt_propagation_test` which returns `Ok(100)` for
/// `AdaptiveWarmup`, simulating a real TRT engine compile.
#[tokio::test]
async fn trt_adaptive_warmup_broadcasts_shape_when_compile_ms_nonzero() {
    let (bcast_tx, mut bcast_rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);

    let pool = EmbedPool::for_trt_propagation_test(bcast_tx, 100);

    let (jit_tx, jit_rx) = mpsc::channel::<(usize, usize)>(64);
    jit_tx.send((4, 512)).await.unwrap();
    drop(jit_tx);

    let cfg = AdaptiveWarmupConfig {
        enabled: true,
        quiet_secs: 0,
        max_shapes_per_hour: 12,
    };

    let loop_handle = tokio::spawn(run_adaptive_warmup_loop(cfg, pool, jit_rx));

    let received = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        loop {
            if let Ok(shape) = bcast_rx.try_recv() {
                return Some(shape);
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or(None);
    loop_handle.abort();

    assert_eq!(
        received,
        Some((4, 512)),
        "broadcast_engine_ready must be called after a TRT compile (compile_ms > 0)"
    );
}

/// When a non-TRT worker returns `Ok(0)` for `AdaptiveWarmup`, the adaptive warmup
/// loop must NOT call `broadcast_engine_ready` (COR-7).
///
/// Uses `for_propagation_test` which returns `Ok(0)` for `AdaptiveWarmup` (the
/// non-TRT no-op path). No broadcast should arrive within the timeout window.
#[tokio::test]
async fn non_trt_adaptive_warmup_does_not_broadcast_for_ok_zero() {
    let (bcast_tx, mut bcast_rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);

    let pool = EmbedPool::for_propagation_test(vec![], vec![], bcast_tx);

    let (jit_tx, jit_rx) = mpsc::channel::<(usize, usize)>(64);
    jit_tx.send((4, 512)).await.unwrap();
    drop(jit_tx);

    let cfg = AdaptiveWarmupConfig {
        enabled: true,
        quiet_secs: 0,
        max_shapes_per_hour: 12,
    };

    let loop_handle = tokio::spawn(run_adaptive_warmup_loop(cfg, pool, jit_rx));

    let received = tokio::time::timeout(std::time::Duration::from_millis(200), async {
        loop {
            if let Ok(shape) = bcast_rx.try_recv() {
                return Some(shape);
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or(None);
    loop_handle.abort();

    assert_eq!(
        received, None,
        "broadcast_engine_ready must NOT be called for a non-TRT Ok(0) compile (COR-7)"
    );
}
