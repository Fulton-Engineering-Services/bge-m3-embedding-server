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

//! Per-request dispatch for dense, sparse, dual, probe, and adaptive warmup.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;

use super::config::WorkerConfig;
use super::guard::{InferenceOutcome, log_guard_rejection};
use super::logging::log_if_abandoned_mid_flight;
use super::probe::run_probe_batch;
use super::propagation::log_inference_complete;
use super::trt_retry::{embed_with_trt_retry, is_trt_engine_build_fatal};
use crate::config::EpSelection;
use crate::embedder::dense::embed_dense;
use crate::embedder::dual::embed_both;
use crate::embedder::jit_guard::{self, TrtJitGuard};
use crate::embedder::sparse::embed_sparse;
use crate::embedder::trt_warmup::trt_prewarm;
use crate::embedder::types::EmbedRequest;

/// Result of dispatching one worker request through inference.
pub(super) struct DispatchOutcome {
    pub outcome: InferenceOutcome,
    pub skip: bool,
}

/// Runs one `EmbedRequest` against a loaded session.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn dispatch_request(
    request: EmbedRequest,
    session: &mut ort::session::Session,
    tokenizer: &mut tokenizers::Tokenizer,
    config: &WorkerConfig,
    id: usize,
    cache_dir: &Path,
    detected_sm: Option<&str>,
    warmed_local: &mut HashSet<(usize, usize)>,
    consecutive_failures: u64,
    shape_guard: Option<&TrtJitGuard>,
) -> DispatchOutcome {
    let mut inner_skip = false;
    let mut inner_outcome = InferenceOutcome::Ok;

    match request {
        EmbedRequest::Dense { texts, reply } => {
            // Pre-dispatch abandonment check: the router's hedged
            // race or the original HTTP client may have already
            // disconnected while this request sat in the worker
            // queue. ORT `session.run()` is a blocking C call that
            // cannot be interrupted mid-MatMul (see CLAUDE.md
            // "client disconnect" gotcha), so the only opportunity
            // we have to save work is BEFORE inference starts. The
            // post-inference check below is observability only.
            if reply.is_closed() {
                tracing::warn!(
                    worker_id = id,
                    route = "dense",
                    batch_size = texts.len(),
                    "request abandoned by client before dispatch — skipping inference"
                );
                inner_skip = true;
            } else {
                let t_inference = std::time::Instant::now();
                let cm_guard = config.cost_model.load();
                let result = embed_with_trt_retry(
                    |cm| {
                        embed_dense(
                            session,
                            tokenizer,
                            &texts,
                            cm,
                            config.model_variant,
                            shape_guard,
                        )
                    },
                    &cm_guard,
                    id,
                    "dense",
                )
                .map_err(|e| e.context("Dense embed error"));
                let inference_ms = t_inference.elapsed().as_millis();
                let guard_rejected = result
                    .as_ref()
                    .err()
                    .is_some_and(jit_guard::is_trt_shape_rejected);
                let trt_fatal = result.as_ref().err().is_some_and(is_trt_engine_build_fatal);
                let is_err = result.is_err();
                if let Ok((_, ref stats)) = result {
                    if let Some(shape) = log_inference_complete(
                        stats,
                        id,
                        "dense",
                        config.jit_suspect_tx.as_ref(),
                        config.engine_propagation_tx.as_ref(),
                        texts.len(),
                    ) {
                        warmed_local.insert(shape);
                    }
                    tracing::info!(
                        worker_id = id,
                        chunks = stats.chunks,
                        max_chunk_seq = stats.max_chunk_seq,
                        total_token_positions = stats.total_token_positions,
                        seq_len_min = stats.seq_len_min,
                        seq_len_max = stats.seq_len_max,
                        seq_len_mean = stats.seq_len_mean,
                        seq_len_p95 = stats.seq_len_p95,
                        tokenize_ms = stats.tokenize_ms,
                        inference_ms = stats.inference_ms,
                        "worker: dense embed complete"
                    );
                }
                if trt_fatal {
                    tracing::error!(
                        worker_id = id,
                        route = "dense",
                        consecutive_failures = consecutive_failures + 1,
                        "trt_fatal_engine_build: unrecoverable TRT state; \
                                 worker exiting to reset CUDA arena"
                    );
                    inner_outcome = InferenceOutcome::TrtFatal;
                } else if guard_rejected {
                    log_guard_rejection(&result, id, "dense");
                    inner_outcome = InferenceOutcome::Rejected;
                } else if is_err {
                    let next_failures = consecutive_failures + 1;
                    if next_failures
                        >= u64::try_from(config.circuit_breaker_threshold).unwrap_or(u64::MAX)
                    {
                        tracing::error!(
                            worker_id = id,
                            route = "dense",
                            consecutive_failures = next_failures,
                            threshold = config.circuit_breaker_threshold,
                            "circuit_breaker_tripped: unloading models to reset \
                                     CUDA arena; worker will reload on next request"
                        );
                        inner_outcome = InferenceOutcome::CircuitBreak;
                    } else {
                        inner_outcome = InferenceOutcome::Failure;
                    }
                }
                log_if_abandoned_mid_flight(&reply, "dense", id, &result, inference_ms);
                let _ = reply.send(result);
            } // end else (not abandoned)
        }
        EmbedRequest::Sparse { texts, reply } => {
            if reply.is_closed() {
                tracing::warn!(
                    worker_id = id,
                    route = "sparse",
                    batch_size = texts.len(),
                    "request abandoned by client before dispatch — skipping inference"
                );
                inner_skip = true;
            } else {
                let t_inference = std::time::Instant::now();
                let cm_guard = config.cost_model.load();
                let result = embed_with_trt_retry(
                    |cm| {
                        embed_sparse(
                            session,
                            tokenizer,
                            &texts,
                            cm,
                            config.model_variant,
                            shape_guard,
                        )
                    },
                    &cm_guard,
                    id,
                    "sparse",
                )
                .map_err(|e| e.context("Sparse embed error"));
                let inference_ms = t_inference.elapsed().as_millis();
                let guard_rejected = result
                    .as_ref()
                    .err()
                    .is_some_and(jit_guard::is_trt_shape_rejected);
                let trt_fatal = result.as_ref().err().is_some_and(is_trt_engine_build_fatal);
                let is_err = result.is_err();
                if let Ok((_, ref stats)) = result {
                    if let Some(shape) = log_inference_complete(
                        stats,
                        id,
                        "sparse",
                        config.jit_suspect_tx.as_ref(),
                        config.engine_propagation_tx.as_ref(),
                        texts.len(),
                    ) {
                        warmed_local.insert(shape);
                    }
                    tracing::info!(
                        worker_id = id,
                        chunks = stats.chunks,
                        max_chunk_seq = stats.max_chunk_seq,
                        total_token_positions = stats.total_token_positions,
                        seq_len_min = stats.seq_len_min,
                        seq_len_max = stats.seq_len_max,
                        seq_len_mean = stats.seq_len_mean,
                        seq_len_p95 = stats.seq_len_p95,
                        tokenize_ms = stats.tokenize_ms,
                        inference_ms = stats.inference_ms,
                        "worker: sparse embed complete"
                    );
                }
                if trt_fatal {
                    tracing::error!(
                        worker_id = id,
                        route = "sparse",
                        consecutive_failures = consecutive_failures + 1,
                        "trt_fatal_engine_build: unrecoverable TRT state; \
                                 worker exiting to reset CUDA arena"
                    );
                    inner_outcome = InferenceOutcome::TrtFatal;
                } else if guard_rejected {
                    log_guard_rejection(&result, id, "sparse");
                    inner_outcome = InferenceOutcome::Rejected;
                } else if is_err {
                    let next_failures = consecutive_failures + 1;
                    if next_failures
                        >= u64::try_from(config.circuit_breaker_threshold).unwrap_or(u64::MAX)
                    {
                        tracing::error!(
                            worker_id = id,
                            route = "sparse",
                            consecutive_failures = next_failures,
                            threshold = config.circuit_breaker_threshold,
                            "circuit_breaker_tripped: unloading models to reset \
                                     CUDA arena; worker will reload on next request"
                        );
                        inner_outcome = InferenceOutcome::CircuitBreak;
                    } else {
                        inner_outcome = InferenceOutcome::Failure;
                    }
                }
                log_if_abandoned_mid_flight(&reply, "sparse", id, &result, inference_ms);
                let _ = reply.send(result);
            } // end else (not abandoned)
        }
        EmbedRequest::Both { texts, reply } => {
            if reply.is_closed() {
                tracing::warn!(
                    worker_id = id,
                    route = "both",
                    batch_size = texts.len(),
                    "request abandoned by client before dispatch — skipping inference"
                );
                inner_skip = true;
            } else {
                let t_inference = std::time::Instant::now();
                let cm_guard = config.cost_model.load();
                let result = embed_with_trt_retry(
                    |cm| {
                        embed_both(
                            session,
                            tokenizer,
                            &texts,
                            cm,
                            config.model_variant,
                            shape_guard,
                        )
                    },
                    &cm_guard,
                    id,
                    "both",
                )
                .map_err(|e| e.context("Dual embed error"));
                let inference_ms = t_inference.elapsed().as_millis();
                let guard_rejected = result
                    .as_ref()
                    .err()
                    .is_some_and(jit_guard::is_trt_shape_rejected);
                let trt_fatal = result.as_ref().err().is_some_and(is_trt_engine_build_fatal);
                let is_err = result.is_err();
                if let Ok((_, ref stats)) = result {
                    if let Some(shape) = log_inference_complete(
                        stats,
                        id,
                        "both",
                        config.jit_suspect_tx.as_ref(),
                        config.engine_propagation_tx.as_ref(),
                        texts.len(),
                    ) {
                        warmed_local.insert(shape);
                    }
                    tracing::info!(
                        worker_id = id,
                        chunks = stats.chunks,
                        max_chunk_seq = stats.max_chunk_seq,
                        total_token_positions = stats.total_token_positions,
                        seq_len_min = stats.seq_len_min,
                        seq_len_max = stats.seq_len_max,
                        seq_len_mean = stats.seq_len_mean,
                        seq_len_p95 = stats.seq_len_p95,
                        tokenize_ms = stats.tokenize_ms,
                        inference_ms = stats.inference_ms,
                        "worker: both embed complete"
                    );
                }
                if trt_fatal {
                    tracing::error!(
                        worker_id = id,
                        route = "both",
                        consecutive_failures = consecutive_failures + 1,
                        "trt_fatal_engine_build: unrecoverable TRT state; \
                                 worker exiting to reset CUDA arena"
                    );
                    inner_outcome = InferenceOutcome::TrtFatal;
                } else if guard_rejected {
                    log_guard_rejection(&result, id, "both");
                    inner_outcome = InferenceOutcome::Rejected;
                } else if is_err {
                    let next_failures = consecutive_failures + 1;
                    if next_failures
                        >= u64::try_from(config.circuit_breaker_threshold).unwrap_or(u64::MAX)
                    {
                        tracing::error!(
                            worker_id = id,
                            route = "both",
                            consecutive_failures = next_failures,
                            threshold = config.circuit_breaker_threshold,
                            "circuit_breaker_tripped: unloading models to reset \
                                     CUDA arena; worker will reload on next request"
                        );
                        inner_outcome = InferenceOutcome::CircuitBreak;
                    } else {
                        inner_outcome = InferenceOutcome::Failure;
                    }
                }
                log_if_abandoned_mid_flight(&reply, "both", id, &result, inference_ms);
                let _ = reply.send(result);
            } // end else (not abandoned)
        }
        EmbedRequest::Probe { texts, reply } => {
            // Probe: tokenize once without padding, run dense inference
            // on a single flat batch at the chunk's natural max_seq.
            // Probes are internal — no client-disconnect path applies.
            let result = run_probe_batch(session, tokenizer, &texts);
            let _ = reply.send(result);
        }
        EmbedRequest::AdaptiveWarmup { batch, seq, ack } => {
            // Run trt_prewarm for a single shape so the TRT EP
            // compiles and caches the engine during an idle window.
            // On CPU/CUDA EP this is a cheap no-op (returns Ok(0)).
            let result: anyhow::Result<u64> = if config.ep == EpSelection::TensorRt {
                let shape = vec![(batch, seq)];
                let stats = trt_prewarm(session, &shape, id, cache_dir, detected_sm);
                if stats.warmed > 0 || stats.fully_cached {
                    // Newly-compiled tier now has a persisted
                    // engine plan: extend the guard ceiling so
                    // real requests at this seq are admitted.
                    config
                        .warmed_seq_ceiling
                        .fetch_max(stats.max_warmed_seq.max(seq), Ordering::AcqRel);
                    Ok(stats.total_compile_ms)
                } else {
                    Err(anyhow::anyhow!(
                        "adaptive warmup: no shapes warmed for \
                                     ({batch}, {seq}) on worker {id}"
                    ))
                }
            } else {
                Ok(0)
            };
            let _ = ack.send(result);
        }
    } // end match request

    DispatchOutcome {
        outcome: inner_outcome,
        skip: inner_skip,
    }
}
