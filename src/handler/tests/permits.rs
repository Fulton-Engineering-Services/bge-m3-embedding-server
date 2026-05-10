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

use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use axum::extract::State;
use axum::Json;
use tokio::sync::Semaphore;

use super::super::dense_embeddings;
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::models::{DenseRequest, TextInput};
use crate::state::{AppState, ProbeStatus};

// ── permit-gating ──────────────────────────────────────────────────────────

#[tokio::test]
async fn dense_embeddings_blocks_when_no_permits_available() {
    // Semaphore with 0 permits — all requests must queue.
    let permits = Arc::new(Semaphore::new(0));
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![vec![0.1f32]], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::clone(&permits),
    });
    let req = DenseRequest {
        input: TextInput(vec!["hello".into()]),
        model: None,
    };
    // Fire the request in a background task.
    let state_clone = Arc::clone(&state);
    let handle = tokio::spawn(async move { dense_embeddings(State(state_clone), Json(req)).await });
    // Give the task time to start and attempt permit acquisition.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !handle.is_finished(),
        "request should still be blocked on 0 permits"
    );
    // Release a permit — request should now complete.
    permits.add_permits(1);
    let result = tokio::time::timeout(Duration::from_secs(1), handle)
        .await
        .expect("request should complete after permit is released")
        .expect("task should not panic");
    assert!(
        result.is_ok(),
        "dense_embeddings should succeed once permitted"
    );
}

#[tokio::test]
async fn request_permits_rises_after_probe_complete() {
    // Verify the semaphore protocol: starting at N-1 permits, add_permits(1)
    // after probe completion brings it to N.
    let n: usize = 7;
    let initial = n.saturating_sub(1).max(1);
    let permits = Arc::new(Semaphore::new(initial));
    assert_eq!(
        permits.available_permits(),
        initial,
        "initial permits should be cfg_workers - 1"
    );
    permits.add_permits(1);
    assert_eq!(
        permits.available_permits(),
        n,
        "after probe completes, permits should equal cfg_workers"
    );
}

// ── worst-case budget invariant ───────────────────────────────────────────

#[test]
fn worst_case_peak_below_available_when_correctly_budgeted() {
    // Production config: 28 GB Fargate task, 7 fp16 workers, 0.7 safety.
    // With accurate model_rss_per_worker the formula must stay under the limit.
    use crate::embedder::OS_HEADROOM_BYTES;

    let available_bytes: usize = 28 * 1024 * 1024 * 1024; // 28 GB
    let cfg_workers: usize = 7;
    let model_rss_per_worker: usize = 1_080_000_000; // ~1.08 GB fp16
    let memory_safety_factor: f64 = 0.7;

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let total_workspace = available_bytes
        .saturating_sub(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_sub(OS_HEADROOM_BYTES);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let per_worker_workspace =
        ((total_workspace as f64) * memory_safety_factor / (cfg_workers as f64)) as usize;

    let worst_case_peak = cfg_workers
        .saturating_mul(per_worker_workspace)
        .saturating_add(cfg_workers.saturating_mul(model_rss_per_worker))
        .saturating_add(OS_HEADROOM_BYTES);

    assert!(
        worst_case_peak < available_bytes,
        "worst_case_peak ({} MB) must be below available ({} MB)",
        worst_case_peak / (1024 * 1024),
        available_bytes / (1024 * 1024),
    );

    #[allow(clippy::cast_precision_loss)]
    let utilization_pct = (worst_case_peak as f64 / available_bytes as f64) * 100.0;
    assert!(
        utilization_pct < 90.0,
        "utilization_pct ({utilization_pct:.1}%) must be below the 90% WARN threshold"
    );
}
