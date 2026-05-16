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

//! Tests for `drain_rx`: insertion of new shapes, deduplication, and
//! filtering of already-warmed or already-pending shapes.

use std::collections::HashSet;

use tokio::sync::mpsc;

use super::super::drain_rx;

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
