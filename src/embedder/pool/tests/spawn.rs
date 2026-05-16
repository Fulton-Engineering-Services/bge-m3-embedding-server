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

use std::time::Duration;

use super::super::super::worker::WorkerConfig;
use super::super::EmbedPool;
use super::helpers::{bad_cache_dir, test_cost_model_handle};

#[tokio::test]
async fn spawn_propagates_leader_load_failure() {
    let (pool, init_handle) = EmbedPool::spawn(
        1,
        bad_cache_dir(),
        WorkerConfig {
            cost_model: test_cost_model_handle(),
            idle_timeout: None,
            model_variant: crate::config::ModelVariant::Fp32,
            max_seq_length: 512,
            intra_threads: 1,
            ep: crate::config::EpSelection::Cpu,
            trt_warmup_shapes: vec![],
            device_id: 0,
            gpu_count: 1,
            trt_max_workspace_bytes: None,
            gpu_mem_limit_bytes: None,
            jit_suspect_tx: None,
            engine_propagation_tx: None,
        },
    );

    let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
        .await
        .expect("init_handle should resolve quickly, not hang")
        .expect("JoinHandle should not panic");

    assert!(
        result.is_err(),
        "init should return Err on leader load failure"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("failed to load"),
        "error should mention load failure, got: {msg}"
    );

    assert_eq!(pool.loaded_worker_count(), 0);
}

/// When a GPU execution provider is selected and `BGE_M3_WORKERS` exceeds
/// `gpu_count`, the pool clamps `n` to `gpu_count` before spawning workers.
/// The clamping is synchronous — it happens inside `EmbedPool::spawn` before
/// the async init task is scheduled — so `live_worker_count()` reflects the
/// post-clamp value immediately after `spawn` returns and before the init
/// task has had any chance to run.
///
/// We use a bad cache directory so the worker fails fast; the intent is
/// to verify the clamp, not a successful model load.
#[tokio::test]
async fn gpu_ep_clamps_workers_to_gpu_count() {
    let (pool, init_handle) = EmbedPool::spawn(
        4,
        bad_cache_dir(),
        WorkerConfig {
            cost_model: test_cost_model_handle(),
            idle_timeout: None,
            model_variant: crate::config::ModelVariant::Fp16,
            max_seq_length: 128,
            intra_threads: 1,
            ep: crate::config::EpSelection::Cuda,
            trt_warmup_shapes: vec![],
            device_id: 0,
            gpu_count: 1, // 4 workers requested, 1 GPU → clamped to 1
            trt_max_workspace_bytes: None,
            gpu_mem_limit_bytes: None,
            jit_suspect_tx: None,
            engine_propagation_tx: None,
        },
    );

    // `live_workers` is initialised to the post-clamp `n` (1, not 4) before
    // the async init task is scheduled.  In the single-threaded
    // `#[tokio::test]` executor no other task runs until we `.await`, so
    // the atomic is still at its initial value here.
    assert_eq!(
        pool.live_worker_count(),
        1,
        "Cuda EP with gpu_count=1 should clamp requested workers (4) to 1"
    );

    // Drive the init task to completion (it will Err — bad cache dir).
    let _ = tokio::time::timeout(Duration::from_secs(5), init_handle).await;
}

/// Verifies that `EmbedRequest::AdaptiveWarmup` is dispatched and handled by
/// the pool without panicking.  The `with_fixed_responses` mock handles the
/// variant and replies `Ok(0)`, exercising the `AdaptiveWarmup` dispatch arm.
#[tokio::test]
async fn pool_handles_adaptive_warmup_request() {
    let pool = EmbedPool::with_fixed_responses(vec![vec![0.1f32]], vec![]);
    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    pool.send_adaptive_warmup(1, 128, ack_tx)
        .await
        .expect("send should succeed against live fixture pool");
    let result = tokio::time::timeout(Duration::from_secs(5), ack_rx)
        .await
        .expect("ack should arrive within timeout")
        .expect("ack sender should not be dropped");
    assert!(
        result.is_ok(),
        "fixture pool AdaptiveWarmup handler must reply Ok; got {result:?}"
    );
}

#[tokio::test]
async fn spawn_multi_worker_fails_fast_on_leader_failure() {
    let (pool, init_handle) = EmbedPool::spawn(
        3,
        bad_cache_dir(),
        WorkerConfig {
            cost_model: test_cost_model_handle(),
            idle_timeout: None,
            model_variant: crate::config::ModelVariant::Fp32,
            max_seq_length: 512,
            intra_threads: 1,
            ep: crate::config::EpSelection::Cpu,
            trt_warmup_shapes: vec![],
            device_id: 0,
            gpu_count: 1,
            trt_max_workspace_bytes: None,
            gpu_mem_limit_bytes: None,
            jit_suspect_tx: None,
            engine_propagation_tx: None,
        },
    );

    let result = tokio::time::timeout(Duration::from_secs(5), init_handle)
        .await
        .expect("init_handle should resolve quickly, not hang")
        .expect("JoinHandle should not panic");

    assert!(
        result.is_err(),
        "init should fail without spawning followers"
    );
    assert_eq!(pool.loaded_worker_count(), 0);
}
