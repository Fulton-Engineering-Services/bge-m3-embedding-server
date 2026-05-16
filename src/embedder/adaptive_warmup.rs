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

//! Adaptive in-process background warmup loop.
//!
//! Detects TRT engine cache misses during live inference (shapes whose
//! total `inference_ms >= CACHE_HIT_THRESHOLD_MS`) and compiles their engines
//! during idle windows (queue empty for `quiet_secs` consecutive seconds)
//! so subsequent requests hit the cache.
//!
//! ## Integration
//!
//! 1. **Before** spawning the worker pool, create a bounded channel with
//!    [`mpsc::channel::<(usize, usize)>`] (capacity 64 is sufficient).
//! 2. Store the sender half in [`WorkerConfig::jit_suspect_tx`].  Workers call
//!    `try_send((batch, seq))` after any inference whose `inference_ms` exceeds
//!    the cache-hit threshold.
//! 3. **After** spawning the pool, call [`spawn_adaptive_warmup`] with the
//!    receiver half and a clone of the pool.
//!
//! The background task accumulates suspected miss shapes, waits for an idle
//! window, and compiles one shape at a time via [`EmbedPool::send_adaptive_warmup`].
//! Successfully compiled shapes are not re-submitted in the current session.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::pool::EmbedPool;

/// Sender half of the JIT-suspect channel.
///
/// Workers hold an optional clone of this sender and call `try_send`
/// (non-blocking, drops if full) after any inference whose `inference_ms`
/// equals or exceeds the TRT cache-hit threshold.
pub(crate) type JitSuspectSender = mpsc::Sender<(usize, usize)>;

/// Configuration for the adaptive background warmup task.
pub(crate) struct AdaptiveWarmupConfig {
    /// Whether adaptive warmup is enabled. When `false`, [`spawn_adaptive_warmup`]
    /// is a no-op.
    pub enabled: bool,
    /// Seconds of continuous idle (zero queue depth) required before the task
    /// fires a warmup shape. Prevents warmup interference with live inference.
    pub quiet_secs: u64,
    /// Maximum number of shapes the task will compile per rolling hour. Acts as
    /// a rate-limiter to prevent runaway warmup loops on high-traffic deployments.
    pub max_shapes_per_hour: u32,
    /// Path to the model cache directory (passed to the worker for `trt_prewarm`).
    #[allow(dead_code)]
    pub cache_dir: PathBuf,
}

/// Spawns the adaptive background warmup task.
///
/// Must be called **after** [`EmbedPool::spawn`] so a pool clone is available.
/// The caller creates the JIT suspect channel before spawning the pool and
/// passes the receiver half here; the sender half is stored in
/// [`WorkerConfig::jit_suspect_tx`].
///
/// Does nothing if `config.enabled` is `false`.
pub(crate) fn spawn_adaptive_warmup(
    config: AdaptiveWarmupConfig,
    pool: EmbedPool,
    rx: mpsc::Receiver<(usize, usize)>,
) {
    if !config.enabled {
        return;
    }
    tokio::spawn(async move {
        run_adaptive_warmup_loop(config, pool, rx).await;
    });
}

async fn run_adaptive_warmup_loop(
    config: AdaptiveWarmupConfig,
    pool: EmbedPool,
    mut rx: mpsc::Receiver<(usize, usize)>,
) {
    let mut pending: Vec<(usize, usize)> = Vec::new();
    let mut warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut shapes_this_hour: u32 = 0;
    let mut hour_start = Instant::now();

    loop {
        // Drain the JIT-suspect channel into pending (non-blocking).
        drain_rx(&mut rx, &mut pending, &warmed);

        // Reset hourly compile budget.
        if hour_start.elapsed() >= Duration::from_secs(3600) {
            shapes_this_hour = 0;
            hour_start = Instant::now();
        }

        // Nothing actionable right now — wait for new suspects or budget reset.
        if pending.is_empty() || shapes_this_hour >= config.max_shapes_per_hour {
            tokio::select! {
                maybe = rx.recv() => {
                    if let Some(shape) = maybe {
                        if !warmed.contains(&shape) && !pending.contains(&shape) {
                            pending.push(shape);
                        }
                    } else {
                        tracing::info!(
                            "adaptive_warmup: JIT suspect channel closed, exiting"
                        );
                        return;
                    }
                }
                () = tokio::time::sleep(Duration::from_secs(5)) => {}
            }
            continue;
        }

        // Wait until the request queue empties.
        if pool.queue_depth() > 0 {
            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }

        // Confirm idle for `quiet_secs` consecutive seconds before firing.
        let confirmed_idle =
            wait_for_quiet_window(config.quiet_secs, &pool, &mut rx, &mut pending, &warmed).await;
        if !confirmed_idle {
            continue;
        }

        // Pop and fire one shape.
        if let Some(&shape) = pending.first() {
            let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
            if pool
                .send_adaptive_warmup(shape.0, shape.1, ack_tx)
                .await
                .is_err()
            {
                tracing::warn!("adaptive_warmup: pool channel closed, exiting");
                return;
            }
            match ack_rx.await {
                Ok(Ok(compile_ms)) => {
                    pending.retain(|s| s != &shape);
                    warmed.insert(shape);
                    shapes_this_hour += 1;
                    tracing::info!(
                        batch = shape.0,
                        seq = shape.1,
                        compile_ms,
                        shapes_this_hour,
                        "adaptive_warmup_complete"
                    );
                }
                Ok(Err(ref e)) => {
                    tracing::warn!(
                        batch = shape.0,
                        seq = shape.1,
                        error = %e,
                        "adaptive_warmup_failed"
                    );
                    // Remove to avoid infinite retry in this session.
                    pending.retain(|s| s != &shape);
                }
                Err(_) => {
                    tracing::warn!(
                        batch = shape.0,
                        seq = shape.1,
                        "adaptive_warmup_ack_dropped"
                    );
                    pending.retain(|s| s != &shape);
                }
            }
        }
    }
}

/// Drains any available items from `rx` into `pending`, skipping already-warmed
/// and already-pending shapes. Non-blocking: returns as soon as `try_recv` fails.
fn drain_rx(
    rx: &mut mpsc::Receiver<(usize, usize)>,
    pending: &mut Vec<(usize, usize)>,
    warmed: &HashSet<(usize, usize)>,
) {
    while let Ok(shape) = rx.try_recv() {
        if !warmed.contains(&shape) && !pending.contains(&shape) {
            pending.push(shape);
        }
    }
}

/// Waits until the pool queue has been idle for `quiet_secs` consecutive seconds.
///
/// Continuously drains the JIT-suspect channel into `pending` while waiting.
/// Returns `true` if the quiet window was reached, `false` if the queue became
/// busy again and we should re-enter the outer idle-check loop.
async fn wait_for_quiet_window(
    quiet_secs: u64,
    pool: &EmbedPool,
    rx: &mut mpsc::Receiver<(usize, usize)>,
    pending: &mut Vec<(usize, usize)>,
    warmed: &HashSet<(usize, usize)>,
) -> bool {
    let mut consecutive_idle: u64 = 0;
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        drain_rx(rx, pending, warmed);

        if pool.queue_depth() == 0 {
            consecutive_idle += 1;
            if consecutive_idle >= quiet_secs {
                return true;
            }
        } else {
            // Queue became busy — abort this quiet-window check.
            return false;
        }
    }
}
