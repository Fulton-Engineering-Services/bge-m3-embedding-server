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

//! Tests for `broadcast_engine_ready` fan-out and no-subscriber resilience.

use crate::embedder::pool::EmbedPool;

/// `broadcast_engine_ready` fans out the shape to all subscribed receivers.
///
/// Creates 3 receivers from a single broadcast channel, builds a pool with
/// the sender, and verifies every receiver gets the broadcasted shape.
#[tokio::test]
async fn broadcast_engine_ready_fans_out_to_n_subscribers() {
    let (tx, mut rx1) = tokio::sync::broadcast::channel::<(usize, usize)>(32);
    let mut rx2 = tx.subscribe();
    let mut rx3 = tx.subscribe();

    let pool = EmbedPool::for_propagation_test(vec![], vec![], tx);

    pool.broadcast_engine_ready((4, 512));

    assert_eq!(
        rx1.try_recv().unwrap(),
        (4, 512),
        "receiver 1 must get the shape"
    );
    assert_eq!(
        rx2.try_recv().unwrap(),
        (4, 512),
        "receiver 2 must get the shape"
    );
    assert_eq!(
        rx3.try_recv().unwrap(),
        (4, 512),
        "receiver 3 must get the shape"
    );
}

/// `broadcast_engine_ready` does not panic when all receivers have been
/// dropped (no subscribers).
///
/// `broadcast::Sender::send` returns `Err` when there are no subscribers;
/// `broadcast_engine_ready` discards that error with `let _ = ...`.
#[tokio::test]
async fn broadcast_engine_ready_no_panic_when_no_subscribers() {
    let (tx, rx) = tokio::sync::broadcast::channel::<(usize, usize)>(32);
    // Drop the only receiver so the channel has zero subscribers.
    drop(rx);

    let pool = EmbedPool::for_propagation_test(vec![], vec![], tx);

    // Must not panic even when there are no subscribers.
    pool.broadcast_engine_ready((1, 128));
}
