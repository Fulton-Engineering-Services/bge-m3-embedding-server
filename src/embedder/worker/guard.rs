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

//! Worker lifecycle guard, in-band JIT guard wiring, and inference outcomes.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::config::WorkerConfig;
use crate::config::EpSelection;
use crate::embedder::jit_guard::TrtJitGuard;

pub(super) struct WorkerGuard(pub Arc<AtomicUsize>);

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        let prev = self.0.fetch_sub(1, Ordering::AcqRel);
        let live_after_drop = prev.saturating_sub(1);
        if live_after_drop == 0 {
            tracing::error!("All embedding workers have exited — pool is degraded");
        } else {
            tracing::warn!(live_after_drop, "Embedding worker exited");
        }
    }
}

/// Builds the per-request in-band TRT JIT guard from the worker config and the
/// live pool-wide warmed-sequence ceiling.
///
/// Returns `None` (guard disabled) on non-TRT EPs or when
/// `BGE_M3_TRT_INBAND_JIT_GUARD=0`, so the embed call sites pass `None` and
/// skip all guard work. The ceiling is read fresh on every request so the
/// decision reflects the latest coverage extended by adaptive warmup or
/// engine propagation.
pub(super) fn build_shape_guard(config: &WorkerConfig) -> Option<TrtJitGuard> {
    if config.ep == EpSelection::TensorRt && config.trt_inband_jit_guard_enabled {
        Some(TrtJitGuard::new(
            config.trt_inband_jit_guard_seq,
            config.warmed_seq_ceiling.load(Ordering::Acquire),
        ))
    } else {
        None
    }
}

/// Emits a `WARN` describing an in-band TRT JIT guard refusal.
///
/// Greppable tag: `trt_jit_guard_refused`. A refusal means the worker
/// protected itself from a dangerous, uncovered chunk shape that could have
/// triggered a process-killing pathological autotuner allocation; the client
/// receives HTTP `503` and may retry once warmup coverage extends.
pub(super) fn log_guard_rejection<T>(
    result: &anyhow::Result<T>,
    worker_id: usize,
    route: &'static str,
) {
    if let Err(e) = result {
        tracing::warn!(
            target: "bge_m3_embedding_server::trt_warmup",
            tag = "trt_jit_guard_refused",
            worker_id,
            route,
            error = %e,
            "in-band TRT JIT guard refused a dangerous, uncovered chunk shape; \
             returning 503 instead of risking a process-killing autotuner \
             allocation (request is retriable once warmup coverage extends)"
        );
    }
}

/// Outcome of a single inference call in the worker request loop.
///
/// Used to communicate circuit-breaker and fatal-exit decisions out of the
/// nested borrow scope (where `session`/`tokenizer` live) into the outer
/// scope where `models` can be safely mutated.
#[derive(Default)]
pub(super) enum InferenceOutcome {
    /// Inference succeeded; reset the consecutive-failure counter.
    #[default]
    Ok,
    /// Inference failed; increment the consecutive-failure counter.
    Failure,
    /// Fatal TRT engine build error; worker should exit immediately.
    TrtFatal,
    /// Consecutive-failure threshold reached; unload models and reset counter.
    CircuitBreak,
    /// The in-band TRT JIT guard refused the request (dangerous, uncovered
    /// shape). The worker is healthy and deliberately protected itself, so the
    /// consecutive-failure counter is left unchanged — a refusal is neither a
    /// success nor a GPU failure and must not contribute to tripping the
    /// circuit breaker.
    Rejected,
}
