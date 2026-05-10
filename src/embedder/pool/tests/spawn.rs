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
