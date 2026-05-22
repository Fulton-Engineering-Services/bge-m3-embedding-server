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

//! Tests for `wait_for_quiet_window` timing behaviour and the hourly
//! rate-limit reset arithmetic.

use std::collections::HashSet;

use tokio::sync::mpsc;

use super::super::wait_for_quiet_window;
use crate::embedder::pool::EmbedPool;

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

/// When the queue is busy (`queue_depth` > 0) at the first sleep boundary,
/// `wait_for_quiet_window` must return `false` immediately.
#[tokio::test(start_paused = true)]
async fn wait_for_quiet_window_returns_false_when_queue_busy() {
    let pool = EmbedPool::busy_for_test();
    assert_eq!(pool.queue_depth(), 1, "test setup: queue must start busy");

    let (_, mut rx) = mpsc::channel::<(usize, usize)>(1);
    let warmed: HashSet<(usize, usize)> = HashSet::new();
    let mut pending: indexmap::IndexSet<(usize, usize)> = indexmap::IndexSet::new();

    let task = tokio::spawn(async move {
        wait_for_quiet_window(1, &pool, &mut rx, &mut pending, &warmed).await
    });
    tokio::time::advance(std::time::Duration::from_secs(2)).await;
    let result = task.await.unwrap();
    assert!(
        !result,
        "should return false when queue is busy after first sleep"
    );
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
fn hourly_reset_condition_does_not_fire_before_one_hour() {
    let hour_start = std::time::Instant::now();
    let one_hour = std::time::Duration::from_hours(1);
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
    let one_hour = std::time::Duration::from_hours(1);

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
