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

//! Blocking worker thread and request dispatch loop.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use tokio::runtime::Handle;
use tokio::sync::{Mutex, mpsc};
use tracing::info;

use super::config::WorkerConfig;
use super::dispatch::{DispatchOutcome, dispatch_request, reply_request_load_error};
use super::guard::{
    InferenceOutcome, WorkerGuard, build_shape_guard, next_consecutive_failures,
    should_unload_on_outcome,
};
use super::probe::probe_run_dense;
use super::propagation::drain_engine_propagation;
use super::startup::{StartupOutcome, startup_worker};
use crate::config::EpSelection;
use crate::embedder::session::{GpuSessionConfig, load_models};
use crate::embedder::trt_warmup::trt_prewarm;
use crate::embedder::types::EmbedRequest;

#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
pub(in crate::embedder) fn run_worker(
    id: usize,
    cache_dir: PathBuf,
    rx: Arc<Mutex<mpsc::Receiver<EmbedRequest>>>,
    ready_tx: mpsc::Sender<Result<usize>>,
    live_workers: Arc<AtomicUsize>,
    loaded_workers: Arc<AtomicUsize>,
    config: WorkerConfig,
) -> Result<()> {
    let _guard = WorkerGuard(Arc::clone(&live_workers));

    let rt = Handle::current();
    let StartupOutcome {
        initial_models,
        detected_sm,
    } = startup_worker(id, &cache_dir, &ready_tx, &config, &rt)?;
    let mut models: Option<(ort::session::Session, tokenizers::Tokenizer)> = Some(initial_models);

    // Tracks shapes already prewarmed by this worker via engine propagation so
    // we skip shapes we originated (which are already in our TRT profile after
    // the originating worker inserts into warmed_local before broadcasting).
    let mut warmed_local: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    // Derive per-worker broadcast receiver from the shared sender in config.
    // Each call to tx.subscribe() creates an independent receiver starting from
    // the current channel position, so workers only see shapes broadcast after
    // their own subscribe point (i.e. after model load is complete).
    let mut engine_propagation_rx = config
        .engine_propagation_tx
        .as_ref()
        .map(tokio::sync::broadcast::Sender::subscribe);

    // Per-worker consecutive-failure counter for the inference circuit breaker.
    // Incremented on every inference Err; reset to 0 on every inference Ok.
    // When it reaches `config.circuit_breaker_threshold` the worker unloads
    // its models (drops the ORT session, clears the CUDA arena) and decrements
    // `loaded_workers`. The standard idle-reload path handles model recovery on
    // the next incoming request.
    let mut consecutive_failures: u64 = 0;

    info!("Worker {id} entering request loop");
    loop {
        // Drain peer engine-ready notifications and run trt_prewarm for any
        // new shapes so the in-memory TRT profile is extended before the next
        // real request for that shape arrives (~1-3s fast disk-load vs. full JIT).
        if config.ep == EpSelection::TensorRt
            && let Some(ref mut bcast_rx) = engine_propagation_rx
            && let Some((session, _)) = models.as_mut()
        {
            let sm = detected_sm.as_deref();
            let ceiling = &config.warmed_seq_ceiling;
            drain_engine_propagation(bcast_rx, &mut warmed_local, id, |shape| {
                let started = std::time::Instant::now();
                let stats = trt_prewarm(session, &[shape], id, &cache_dir, sm);
                let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                if stats.warmed > 0 || stats.fully_cached {
                    // A peer-propagated plan now covers this shape on disk and
                    // in this worker's session: extend the guard ceiling.
                    ceiling.fetch_max(stats.max_warmed_seq.max(shape.1), Ordering::AcqRel);
                }
                tracing::info!(
                    target: "bge_m3_embedding_server::trt_shape",
                    worker_id = id,
                    chunk_batch = shape.0,
                    chunk_max_seq = shape.1,
                    elapsed_ms,
                    warmed = stats.warmed,
                    fully_cached = stats.fully_cached,
                    detected_sm = sm.unwrap_or("unfiltered"),
                    "engine_propagation_complete"
                );
            });
        }

        let msg = if let Some(timeout) = config.idle_timeout.filter(|_| models.is_some()) {
            rt.block_on(async {
                tokio::time::timeout(timeout, async { rx.lock().await.recv().await }).await
            })
        } else {
            rt.block_on(async { Ok(rx.lock().await.recv().await) })
        };

        match msg {
            Err(_elapsed) => {
                models = None;
                loaded_workers.fetch_sub(1, Ordering::AcqRel);
                tracing::info!("Worker {id} unloaded models after idle timeout");
            }
            Ok(None) => {
                if models.is_some() {
                    loaded_workers.fetch_sub(1, Ordering::AcqRel);
                }
                info!("Worker {id} channel closed, shutting down");
                break;
            }
            Ok(Some(request)) => {
                if models.is_none() {
                    tracing::info!("Worker {id} reloading models after idle...");
                    let reload_start = std::time::Instant::now();
                    match load_models(
                        &GpuSessionConfig {
                            cache_dir: &cache_dir,
                            model_variant: config.model_variant,
                            max_seq_length: config.max_seq_length,
                            intra_threads: config.intra_threads,
                            ep: config.ep,
                            device_id: config.device_id,
                            trt_max_workspace_bytes: config.trt_max_workspace_bytes,
                            gpu_mem_limit_bytes: config.gpu_mem_limit_bytes,
                        },
                        false,
                    ) {
                        Ok(mut m) => {
                            // Prime the freshly-loaded session arena so the
                            // first incoming request after idle reload doesn't
                            // pay the ~1 GiB lazy-arena-init cost. Same
                            // rationale as the startup priming in the
                            // load-models Ok arm above.
                            let prime_ids = ndarray::Array2::<i64>::zeros((1, 8));
                            let prime_mask = ndarray::Array2::<i64>::ones((1, 8));
                            if let Err(e) = probe_run_dense(&mut m.0, &prime_ids, &prime_mask) {
                                tracing::warn!(
                                    error = %e,
                                    "Worker {id} post-reload arena prime failed"
                                );
                            }
                            models = Some(m);
                            loaded_workers.fetch_add(1, Ordering::AcqRel);
                            tracing::info!(
                                elapsed_ms = reload_start.elapsed().as_millis(),
                                "Worker {id} reloaded models"
                            );
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Worker {id} failed to reload models");
                            let err = anyhow::anyhow!("Model reload failed: {e}");
                            reply_request_load_error(request, err);
                            continue;
                        }
                    }
                }

                // In-band TRT JIT admission guard, rebuilt per request from the
                // live pool-wide warmed-seq ceiling. `None` on non-TRT EPs or
                // when the guard is disabled. Passed into the embed functions
                // which refuse dangerous, uncovered chunk shapes before
                // `session.run()` (returning a TrtJitRejection → HTTP 503).
                let shape_guard = build_shape_guard(&config);

                // Flags for circuit-breaker and fatal-exit decisions; set
                // inside the borrow scope of `session`/`tokenizer` and acted
                // on AFTER that scope ends so `models` can be safely mutated.
                // `InferenceOutcome::TrtFatal` triggers a worker exit.
                // `InferenceOutcome::CircuitBreak` unloads the ORT session.
                let outcome: InferenceOutcome;
                let skip_to_next: bool;

                {
                    let (session, tokenizer) =
                        models.as_mut().expect("models loaded after reload check");

                    let DispatchOutcome {
                        outcome: inner_outcome,
                        skip: inner_skip,
                    } = dispatch_request(
                        request,
                        session,
                        tokenizer,
                        &config,
                        id,
                        &cache_dir,
                        detected_sm.as_deref(),
                        &mut warmed_local,
                        consecutive_failures,
                        shape_guard.as_ref(),
                    );
                    outcome = inner_outcome;
                    skip_to_next = inner_skip;
                } // end borrow scope — session and tokenizer dropped here

                // --- Post-inference actions (models reborrow is now safe) ---
                if skip_to_next {
                    continue;
                }
                consecutive_failures = next_consecutive_failures(outcome, consecutive_failures);
                if matches!(outcome, InferenceOutcome::TrtFatal) {
                    return Err(anyhow::anyhow!(
                        "Worker {id} hit fatal TRT engine build error; exiting"
                    ));
                }
                if should_unload_on_outcome(outcome) {
                    models = None;
                    loaded_workers.fetch_sub(1, Ordering::AcqRel);
                }
            }
        }
    }

    Ok(())
}
