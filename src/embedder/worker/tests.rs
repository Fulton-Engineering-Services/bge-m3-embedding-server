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

use super::*;

// --- is_trt_jit_oom ---

#[test]
fn is_trt_jit_oom_matches_user_allocator_error() {
    let e = anyhow::anyhow!("TRT: User allocator error during cuDNN workspace setup");
    assert!(is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_matches_no_impl_with_workspace() {
    let e = anyhow::anyhow!("Could not find any implementation for node; workspace too small");
    assert!(is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_does_not_match_no_impl_without_alloc() {
    // "Could not find any implementation" without workspace/alloc = unsupported op, not OOM
    let e = anyhow::anyhow!("Could not find any implementation for node /Reshape_3");
    assert!(!is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_does_not_match_unrelated_error() {
    let e = anyhow::anyhow!("OrtStatus: NOT_IMPLEMENTED: opset 18 not supported");
    assert!(!is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_does_not_match_empty() {
    let e = anyhow::anyhow!("");
    assert!(!is_trt_jit_oom(&e));
}

// --- embed_with_trt_retry ---

#[test]
fn embed_with_trt_retry_succeeds_without_retry() {
    let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
    let calls = std::cell::Cell::new(0u32);
    let result = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            Ok(42u64)
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(result.unwrap(), 42);
    assert_eq!(calls.get(), 1);
}

#[test]
fn embed_with_trt_retry_fires_exactly_once_on_oom() {
    let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
    let calls = std::cell::Cell::new(0u32);
    let result: anyhow::Result<u64> = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(anyhow::anyhow!("User allocator error in TRT"))
            } else {
                Ok(99)
            }
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(result.unwrap(), 99);
    assert_eq!(calls.get(), 2);
}

#[test]
fn embed_with_trt_retry_halves_workspace_on_retry() {
    let original = 2 * 1024 * 1024 * 1024usize;
    let cm = CostModel::conservative(original);
    let observed_workspace = std::cell::Cell::new(0usize);
    let calls = std::cell::Cell::new(0u32);
    let _: anyhow::Result<u64> = embed_with_trt_retry(
        |cm| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(anyhow::anyhow!("User allocator error in TRT"))
            } else {
                observed_workspace.set(cm.max_workspace_bytes);
                Ok(0)
            }
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(observed_workspace.get(), original / 2);
}

#[test]
fn embed_with_trt_retry_propagates_non_oom_error_immediately() {
    let cm = CostModel::conservative(1024);
    let calls = std::cell::Cell::new(0u32);
    let result: anyhow::Result<u64> = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("Some unrelated error"))
        },
        &cm,
        0,
        "test",
    );
    assert!(result.is_err());
    assert_eq!(calls.get(), 1, "non-OOM error must not retry");
}

#[test]
fn embed_with_trt_retry_propagates_second_failure() {
    let cm = CostModel::conservative(1024);
    let calls = std::cell::Cell::new(0u32);
    let result: anyhow::Result<u64> = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("User allocator error in TRT"))
        },
        &cm,
        0,
        "test",
    );
    assert!(result.is_err());
    assert_eq!(calls.get(), 2);
}

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
