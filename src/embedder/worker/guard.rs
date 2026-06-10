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

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::config::WorkerConfig;
use super::logging::log_if_abandoned_mid_flight;
use super::propagation::log_inference_complete;
use super::trt_retry::is_trt_engine_build_fatal;
use crate::config::EpSelection;
use crate::embedder::jit_guard::{self, TrtJitGuard};
use crate::embedder::types::{EmbedStats, JitSuspectSender};

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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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

/// Per-route context passed to [`finalize_embed_route`] after inference.
pub(super) struct EmbedRouteContext<'a> {
    pub worker_id: usize,
    pub route: &'static str,
    pub consecutive_failures: u64,
    pub circuit_breaker_threshold: usize,
    pub jit_suspect_tx: Option<&'a JitSuspectSender>,
    pub engine_propagation_tx: Option<&'a tokio::sync::broadcast::Sender<(usize, usize)>>,
    pub batch_len: usize,
}

/// Emits the pre-dispatch abandonment WARN when the client disconnected
/// while the request was still queued.
pub(super) fn log_client_abandoned_before_dispatch(
    worker_id: usize,
    route: &'static str,
    batch_size: usize,
) {
    tracing::warn!(
        worker_id,
        route,
        batch_size,
        "request abandoned by client before dispatch — skipping inference"
    );
}

/// Maps inference error flags to the worker-loop [`InferenceOutcome`].
pub(super) fn classify_inference_outcome(
    trt_fatal: bool,
    guard_rejected: bool,
    is_err: bool,
    consecutive_failures: u64,
    circuit_breaker_threshold: usize,
) -> InferenceOutcome {
    if trt_fatal {
        InferenceOutcome::TrtFatal
    } else if guard_rejected {
        InferenceOutcome::Rejected
    } else if is_err {
        let next_failures = consecutive_failures + 1;
        if next_failures >= u64::try_from(circuit_breaker_threshold).unwrap_or(u64::MAX) {
            InferenceOutcome::CircuitBreak
        } else {
            InferenceOutcome::Failure
        }
    } else {
        InferenceOutcome::Ok
    }
}

/// Returns the consecutive-failure counter value after applying an outcome.
pub(super) fn next_consecutive_failures(
    outcome: InferenceOutcome,
    consecutive_failures: u64,
) -> u64 {
    match outcome {
        InferenceOutcome::Ok | InferenceOutcome::CircuitBreak => 0,
        InferenceOutcome::Failure => consecutive_failures + 1,
        InferenceOutcome::TrtFatal | InferenceOutcome::Rejected => consecutive_failures,
    }
}

/// Returns `true` when the worker loop should unload models after inference.
pub(super) fn should_unload_on_outcome(outcome: InferenceOutcome) -> bool {
    matches!(outcome, InferenceOutcome::CircuitBreak)
}

/// Shared post-inference path for dense, sparse, and dual routes.
///
/// Logs completion stats, classifies errors (TRT fatal / JIT guard / circuit
/// breaker), emits abandonment observability, and sends the oneshot reply.
pub(super) fn finalize_embed_route<T>(
    ctx: &EmbedRouteContext<'_>,
    result: anyhow::Result<(T, EmbedStats)>,
    reply: tokio::sync::oneshot::Sender<anyhow::Result<(T, EmbedStats)>>,
    inference_ms: u128,
    warmed_local: &mut HashSet<(usize, usize)>,
) -> InferenceOutcome {
    let guard_rejected = result
        .as_ref()
        .err()
        .is_some_and(jit_guard::is_trt_shape_rejected);
    let trt_fatal = result.as_ref().err().is_some_and(is_trt_engine_build_fatal);
    let is_err = result.is_err();

    if let Ok((_, ref stats)) = result {
        if let Some(shape) = log_inference_complete(
            stats,
            ctx.worker_id,
            ctx.route,
            ctx.jit_suspect_tx,
            ctx.engine_propagation_tx,
            ctx.batch_len,
        ) {
            warmed_local.insert(shape);
        }
        tracing::info!(
            worker_id = ctx.worker_id,
            chunks = stats.chunks,
            max_chunk_seq = stats.max_chunk_seq,
            total_token_positions = stats.total_token_positions,
            seq_len_min = stats.seq_len_min,
            seq_len_max = stats.seq_len_max,
            seq_len_mean = stats.seq_len_mean,
            seq_len_p95 = stats.seq_len_p95,
            tokenize_ms = stats.tokenize_ms,
            inference_ms = stats.inference_ms,
            route = ctx.route,
            "worker: embed complete"
        );
    }

    let outcome = classify_inference_outcome(
        trt_fatal,
        guard_rejected,
        is_err,
        ctx.consecutive_failures,
        ctx.circuit_breaker_threshold,
    );

    match outcome {
        InferenceOutcome::TrtFatal => {
            tracing::error!(
                worker_id = ctx.worker_id,
                route = ctx.route,
                consecutive_failures = ctx.consecutive_failures + 1,
                "trt_fatal_engine_build: unrecoverable TRT state; \
                 worker exiting to reset CUDA arena"
            );
        }
        InferenceOutcome::Rejected => {
            log_guard_rejection(&result, ctx.worker_id, ctx.route);
        }
        InferenceOutcome::CircuitBreak => {
            tracing::error!(
                worker_id = ctx.worker_id,
                route = ctx.route,
                consecutive_failures = ctx.consecutive_failures + 1,
                threshold = ctx.circuit_breaker_threshold,
                "circuit_breaker_tripped: unloading models to reset \
                 CUDA arena; worker will reload on next request"
            );
        }
        InferenceOutcome::Ok | InferenceOutcome::Failure => {}
    }

    log_if_abandoned_mid_flight(&reply, ctx.route, ctx.worker_id, &result, inference_ms);
    let _ = reply.send(result);
    outcome
}

/// Adaptive-warmup no-op compile duration for non-TRT execution providers.
pub(super) fn adaptive_warmup_non_trt_compile_ms() -> u64 {
    0
}
