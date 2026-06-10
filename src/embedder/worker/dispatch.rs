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
use super::guard::{
    EmbedRouteContext, InferenceOutcome, adaptive_warmup_non_trt_compile_ms, finalize_embed_route,
    log_client_abandoned_before_dispatch,
};
use super::probe::run_probe_batch;
use super::trt_retry::embed_with_trt_retry;
use crate::config::EpSelection;
use crate::embedder::dense::embed_dense;
use crate::embedder::dual::embed_both;
use crate::embedder::jit_guard::TrtJitGuard;
use crate::embedder::sparse::embed_sparse;
use crate::embedder::trt_warmup::trt_prewarm;
use crate::embedder::types::EmbedRequest;

/// Result of dispatching one worker request through inference.
pub(super) struct DispatchOutcome {
    pub outcome: InferenceOutcome,
    pub skip: bool,
}

/// Sends a model-reload error to whichever reply channel the request carries.
pub(super) fn reply_request_load_error(request: EmbedRequest, err: anyhow::Error) {
    match request {
        EmbedRequest::Dense { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        EmbedRequest::Sparse { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        EmbedRequest::Both { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        EmbedRequest::Probe { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        EmbedRequest::AdaptiveWarmup { ack, .. } => {
            let _ = ack.send(Err(err));
        }
    }
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

    let route_ctx = EmbedRouteContext {
        worker_id: id,
        route: "",
        consecutive_failures,
        circuit_breaker_threshold: config.circuit_breaker_threshold,
        jit_suspect_tx: config.jit_suspect_tx.as_ref(),
        engine_propagation_tx: config.engine_propagation_tx.as_ref(),
        batch_len: 0,
    };

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
                log_client_abandoned_before_dispatch(id, "dense", texts.len());
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
                let ctx = EmbedRouteContext {
                    route: "dense",
                    batch_len: texts.len(),
                    ..route_ctx
                };
                inner_outcome =
                    finalize_embed_route(&ctx, result, reply, inference_ms, warmed_local);
            }
        }
        EmbedRequest::Sparse { texts, reply } => {
            if reply.is_closed() {
                log_client_abandoned_before_dispatch(id, "sparse", texts.len());
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
                let ctx = EmbedRouteContext {
                    route: "sparse",
                    batch_len: texts.len(),
                    ..route_ctx
                };
                inner_outcome =
                    finalize_embed_route(&ctx, result, reply, inference_ms, warmed_local);
            }
        }
        EmbedRequest::Both { texts, reply } => {
            if reply.is_closed() {
                log_client_abandoned_before_dispatch(id, "both", texts.len());
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
                let ctx = EmbedRouteContext {
                    route: "both",
                    batch_len: texts.len(),
                    ..route_ctx
                };
                inner_outcome =
                    finalize_embed_route(&ctx, result, reply, inference_ms, warmed_local);
            }
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
                Ok(adaptive_warmup_non_trt_compile_ms())
            };
            let _ = ack.send(result);
        }
    } // end match request

    DispatchOutcome {
        outcome: inner_outcome,
        skip: inner_skip,
    }
}
