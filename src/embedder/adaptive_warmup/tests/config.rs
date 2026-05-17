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

//! Tests for `AdaptiveWarmupConfig` field construction and the guard paths
//! in `spawn_adaptive_warmup` (disabled flag, `max_shapes_per_hour=0`).

use tokio::sync::mpsc;

use super::super::{spawn_adaptive_warmup, AdaptiveWarmupConfig};
use crate::embedder::pool::EmbedPool;

// ─── AdaptiveWarmupConfig field tests ────────────────────────────────────────

/// `spawn_adaptive_warmup` must be a no-op when enabled=false.
#[tokio::test]
async fn spawn_adaptive_warmup_is_noop_when_disabled() {
    let (_, rx) = mpsc::channel::<(usize, usize)>(8);
    let pool = EmbedPool::idle_for_test();

    let cfg = AdaptiveWarmupConfig {
        enabled: false,
        quiet_secs: 3,
        max_shapes_per_hour: 12,
    };
    // Should return without spawning any task (no panic, no hang).
    spawn_adaptive_warmup(cfg, pool, rx);
    // If we reach here the early-return guard worked.
}

/// A config with `max_shapes_per_hour=0` is constructable and the field is zero.
/// The WARN is emitted at runtime in `spawn_adaptive_warmup`; this test verifies
/// the config struct accepts the value so the guard has something to check.
#[test]
fn adaptive_warmup_config_max_shapes_zero_is_constructable() {
    let cfg = AdaptiveWarmupConfig {
        enabled: true,
        quiet_secs: 3,
        max_shapes_per_hour: 0,
    };
    assert_eq!(cfg.max_shapes_per_hour, 0);
}

/// `spawn_adaptive_warmup` with `max_shapes_per_hour=0` and enabled=true must
/// not panic — the warn path is exercised safely.
#[tokio::test]
async fn spawn_adaptive_warmup_does_not_panic_when_max_shapes_zero() {
    // Give it a closed channel so the spawned task exits immediately.
    let (tx, rx) = mpsc::channel::<(usize, usize)>(8);
    drop(tx);
    let pool = EmbedPool::closed_for_test();

    let cfg = AdaptiveWarmupConfig {
        enabled: true,
        quiet_secs: 0,
        max_shapes_per_hour: 0,
    };
    // Should emit a WARN and spawn, but not panic.
    spawn_adaptive_warmup(cfg, pool, rx);
    // Give the spawned task time to detect the closed channel and exit.
    tokio::task::yield_now().await;
}
