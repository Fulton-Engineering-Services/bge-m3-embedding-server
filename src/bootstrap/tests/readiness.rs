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

use std::sync::Arc;

use super::super::readiness::run_readiness_probe;
use super::helpers::{make_test_state, test_cache_dir};

#[tokio::test]
async fn readiness_probe_fails_when_init_returns_error() {
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
    let result = run_readiness_probe(
        handle,
        state,
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("initialization failed"));
}

#[tokio::test]
async fn readiness_probe_fails_when_init_panics() {
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { panic!("worker panic") });
    let result = run_readiness_probe(
        handle,
        state,
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("panicked"));
}

#[tokio::test]
async fn readiness_probe_does_not_set_ready_when_dense_check_fails() {
    // With the serialised-probe design, readiness checks run inside the
    // spawned probe task rather than in the caller.
    // run_readiness_probe returns Ok immediately; the readiness failure
    // is logged and state.ready stays false.
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { Ok::<(), anyhow::Error>(()) });
    // disable_probe_cache=true → no override, no cache → probe spawned
    let result = run_readiness_probe(
        handle,
        Arc::clone(&state),
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    // run_readiness_probe returns Ok — the probe task was spawned.
    assert!(
        result.is_ok(),
        "run_readiness_probe should return Ok (probe spawned)"
    );
    // Give the probe task time to run the readiness check and fail.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // The pool is closed_for_test, so dense() fails; ready should stay false.
    assert!(
        !state.ready.load(std::sync::atomic::Ordering::Acquire),
        "ready must not be set when the dense readiness check fails"
    );
}

#[tokio::test]
async fn readiness_probe_does_not_set_ready_on_failure() {
    let state = make_test_state(false, 256);
    let handle = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("init failed")) });
    let _ = run_readiness_probe(
        handle,
        Arc::clone(&state),
        8192,
        2,
        0.7,
        None,
        test_cache_dir(),
        "fp16".into(),
        true,
    )
    .await;
    assert!(!state.ready.load(std::sync::atomic::Ordering::Acquire));
}
