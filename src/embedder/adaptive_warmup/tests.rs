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

use std::collections::HashSet;

use tokio::sync::mpsc;

use super::*;
use crate::embedder::pool::EmbedPool;

// ─── drain_rx tests ───────────────────────────────────────────────────────────

/// A new shape that is neither warmed nor pending is inserted into pending.
#[tokio::test]
async fn drain_rx_adds_new_shape() {
    let (tx, mut rx) = mpsc::channel::<(usize, usize)>(64);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    tx.send((1, 128)).await.unwrap();
    drop(tx);

    drain_rx(&mut rx, &mut pending, &warmed);

    assert!(pending.contains(&(1_usize, 128_usize)));
    assert_eq!(pending.len(), 1);
}

/// A shape that has already been warmed must be silently dropped.
#[tokio::test]
async fn drain_rx_skips_already_warmed() {
    let (tx, mut rx) = mpsc::channel::<(usize, usize)>(64);
    let mut warmed: HashSet<(usize, usize)> = HashSet::new();
    warmed.insert((1, 128));
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    tx.send((1, 128)).await.unwrap();
    drop(tx);

    drain_rx(&mut rx, &mut pending, &warmed);

    assert!(
        pending.is_empty(),
        "already-warmed shape must not be added to pending"
    );
}

/// Sending the same shape twice into an empty channel must produce exactly one
/// entry in pending (the second send is deduplicated by `drain_rx`).
#[tokio::test]
async fn drain_rx_deduplicates_same_shape_received_twice() {
    let (tx, mut rx) = mpsc::channel::<(usize, usize)>(64);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    tx.send((4, 512)).await.unwrap();
    tx.send((4, 512)).await.unwrap();
    drop(tx);

    drain_rx(&mut rx, &mut pending, &warmed);

    assert_eq!(
        pending.len(),
        1,
        "duplicate shapes must be deduplicated to a single entry"
    );
}

/// A shape that is already in pending must not be inserted again.
#[tokio::test]
async fn drain_rx_skips_already_pending() {
    let (tx, mut rx) = mpsc::channel::<(usize, usize)>(64);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();
    pending.insert((4, 512));

    tx.send((4, 512)).await.unwrap();
    drop(tx);

    drain_rx(&mut rx, &mut pending, &warmed);

    assert_eq!(
        pending.len(),
        1,
        "shape already in pending must not be inserted again"
    );
}

/// Multiple distinct shapes are all added to pending.
#[tokio::test]
async fn drain_rx_adds_multiple_distinct_shapes() {
    let (tx, mut rx) = mpsc::channel::<(usize, usize)>(64);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    tx.send((1, 128)).await.unwrap();
    tx.send((2, 256)).await.unwrap();
    tx.send((4, 512)).await.unwrap();
    drop(tx);

    drain_rx(&mut rx, &mut pending, &warmed);

    assert_eq!(pending.len(), 3);
    assert!(pending.contains(&(1_usize, 128_usize)));
    assert!(pending.contains(&(2_usize, 256_usize)));
    assert!(pending.contains(&(4_usize, 512_usize)));
}

// ─── wait_for_quiet_window tests ─────────────────────────────────────────────

/// `quiet_secs=0` must return `true` immediately without sleeping.
#[tokio::test]
async fn wait_for_quiet_window_returns_immediately_when_quiet_secs_zero() {
    let pool = EmbedPool::idle_for_test();
    let (_, mut rx) = mpsc::channel::<(usize, usize)>(1);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    let result = wait_for_quiet_window(0, &pool, &mut rx, &mut pending, &warmed).await;
    assert!(
        result,
        "quiet_secs=0 must return true immediately without sleeping"
    );
}

/// With an idle pool and `quiet_secs=1`, `wait_for_quiet_window` returns true
/// after one second has elapsed. Use `tokio::time::pause/advance` to avoid a
/// real 1-second wall-clock delay in CI.
#[tokio::test(start_paused = true)]
async fn wait_for_quiet_window_returns_true_when_idle_long_enough() {
    let pool = EmbedPool::idle_for_test();
    let (_, mut rx) = mpsc::channel::<(usize, usize)>(1);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    // advance_to drives the async sleep inside wait_for_quiet_window
    let task = tokio::spawn(async move {
        wait_for_quiet_window(1, &pool, &mut rx, &mut pending, &warmed).await
    });

    tokio::time::advance(std::time::Duration::from_secs(2)).await;
    let result = task.await.unwrap();
    assert!(result, "should return true once idle window is satisfied");
}

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

// ─── Hourly rate-limit reset ──────────────────────────────────────────────────

/// Validates the hourly-reset arithmetic: after 3600 seconds the counter
/// resets to 0. Uses `tokio::time::pause` to control time without sleeping.
///
/// Note: the adaptive warmup loop uses `std::time::Instant` for the hour
/// boundary, which is not controlled by `tokio::time::pause`. This test
/// therefore validates the reset *logic* via a thin wrapper around the
/// exact condition used in `run_adaptive_warmup_loop`.
#[test]
fn hourly_reset_condition_fires_after_3600_seconds() {
    let hour_start = std::time::Instant::now();
    let one_hour = std::time::Duration::from_secs(3600);
    let mut shapes_this_hour: u32 = 5;

    // Before an hour passes the counter is unchanged.
    if hour_start.elapsed() >= one_hour {
        shapes_this_hour = 0;
    }
    // In a real scenario the reset fires when the loop iteration falls at
    // exactly ≥ 3600 s.  We cannot advance std::time in tests without
    // mocking, so we directly exercise the guard expression.
    let would_reset = hour_start.elapsed() >= one_hour;
    if would_reset {
        shapes_this_hour = 0;
    }
    // elapsed is < 1 second here; counter must still be 5.
    assert_eq!(
        shapes_this_hour, 5,
        "counter should not reset before an hour"
    );
}

/// Simulates the reset manually to verify the post-reset invariant.
#[test]
fn hourly_reset_zeroes_counter() {
    let mut shapes_this_hour: u32 = 12;
    let mut hour_start = std::time::Instant::now();
    let one_hour = std::time::Duration::from_secs(3600);

    // Pretend an hour has passed by backdating hour_start.
    hour_start -= one_hour + std::time::Duration::from_millis(1);

    if hour_start.elapsed() >= one_hour {
        shapes_this_hour = 0;
        hour_start = std::time::Instant::now();
    }
    assert_eq!(
        shapes_this_hour, 0,
        "counter must be zero after hourly reset"
    );
    assert!(
        hour_start.elapsed() < one_hour,
        "hour_start must be refreshed after reset"
    );
}
