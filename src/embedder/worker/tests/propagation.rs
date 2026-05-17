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

//! Tests for `drain_engine_propagation`: deduplication via `warmed_local`,
//! lagged-channel recovery, already-warmed shape filtering, and empty-channel
//! early-exit.

use super::super::drain_engine_propagation;

// --- drain_engine_propagation ---

/// The same shape sent twice results in the prewarm closure being called
/// exactly once (deduplicated via `warmed_local`).
#[test]
fn drain_deduplicates_same_shape() {
    let (tx, mut rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);
    tx.send((4, 512)).unwrap();
    tx.send((4, 512)).unwrap();

    let mut warmed_local = std::collections::HashSet::new();
    let mut call_count = 0u32;

    drain_engine_propagation(&mut rx, &mut warmed_local, 0, |_shape| {
        call_count += 1;
    });

    assert_eq!(
        call_count, 1,
        "prewarm closure must be called exactly once for a duplicate shape"
    );
    assert!(warmed_local.contains(&(4_usize, 512_usize)));
}

/// A shape already in `warmed_local` before the drain must not trigger the
/// prewarm closure.
#[test]
fn drain_skips_already_warmed() {
    let (tx, mut rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);
    tx.send((1, 128)).unwrap();

    let mut warmed_local = std::collections::HashSet::new();
    warmed_local.insert((1_usize, 128_usize));
    let mut call_count = 0u32;

    drain_engine_propagation(&mut rx, &mut warmed_local, 0, |_shape| {
        call_count += 1;
    });

    assert_eq!(
        call_count, 0,
        "already-warmed shape must not trigger the prewarm closure"
    );
}

/// Filling the channel past its capacity causes a `Lagged` error; drain must
/// handle it without panicking and continue processing remaining items.
#[test]
fn drain_handles_lagged_without_panic() {
    // Capacity 4 — send 6 items to trigger lagging.
    let (tx, mut rx) = tokio::sync::broadcast::channel::<(usize, usize)>(4);
    for i in 0..6usize {
        let _ = tx.send((i, 128));
    }

    let mut warmed_local = std::collections::HashSet::new();
    // Must not panic.
    drain_engine_propagation(&mut rx, &mut warmed_local, 0, |_| {});
}

/// An empty channel causes the prewarm closure to never be called and drain
/// exits cleanly.
#[test]
fn drain_exits_on_empty() {
    let (_tx, mut rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);

    let mut warmed_local = std::collections::HashSet::new();
    let mut call_count = 0u32;

    drain_engine_propagation(&mut rx, &mut warmed_local, 0, |_shape| {
        call_count += 1;
    });

    assert_eq!(
        call_count, 0,
        "empty channel must not invoke the prewarm closure"
    );
}
