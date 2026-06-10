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

//! Per-worker execution policy shared across the pool.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use arc_swap::ArcSwap;

use crate::binpack::CostModel;
use crate::config::{EpSelection, ModelVariant};
use crate::embedder::types::JitSuspectSender;

/// Execution-policy configuration shared by all workers.
///
/// `cost_model` is an `Arc<ArcSwap<CostModel>>` so all workers share a single
/// handle and the background probe can update the cost model atomically after
/// fitting.  Each worker loads the current value lock-free at the start of
/// every `session.run()` call via `config.cost_model.load()`.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Quadratic-aware workspace cost model and per-worker budget.
    ///
    /// Shared across all workers via `ArcSwap`.  The background probe updates
    /// this handle once fitted coefficients are available; workers observe the
    /// new model on their next request without any coordination or restart.
    pub cost_model: Arc<ArcSwap<CostModel>>,
    /// Duration of inactivity before workers unload their model instances.
    pub idle_timeout: Option<Duration>,
    /// ONNX model variant to load (FP32, FP16, or INT8).
    pub model_variant: ModelVariant,
    /// Maximum tokenized sequence length.
    pub max_seq_length: usize,
    /// Number of intra-op threads each ORT session may use for a single
    /// `session.run()` call. Plumbed through to `load_session` at model load
    /// time. See [`crate::config::Config::intra_threads`] for sizing guidance.
    pub intra_threads: usize,
    /// ONNX Runtime execution provider selection.
    ///
    /// Forwarded to [`load_models`] at model load time so each ORT session
    /// registers the correct EP. On macOS, `CoreML` is always used regardless
    /// of this value. See [`crate::config::EpSelection`] for details.
    pub ep: EpSelection,

    /// `(batch_size, seq_len)` shapes to pre-compile as `TensorRT` engine files
    /// during worker startup.
    ///
    /// Only applied when `ep == EpSelection::TensorRt`. Sourced from
    /// `BGE_M3_TRT_WARMUP_SHAPES` via [`crate::config::Config::trt_warmup_shapes`].
    /// With multiple workers, `EmbedPool::spawn` shards the full shape list
    /// across workers (stride partition) so each GPU compiles a disjoint subset
    /// in parallel. An empty list skips pre-warming entirely.
    pub trt_warmup_shapes: Vec<(usize, usize)>,

    /// CUDA/TRT device ID for this specific worker.
    ///
    /// Set by `EmbedPool::spawn` as `worker_index % gpu_count`. Forwarded to
    /// [`super::session::execution_providers`] so the ORT session binds to the
    /// correct GPU. Ignored on CPU EP and macOS (`CoreML` is single-device).
    pub device_id: u32,

    /// Total number of GPU devices on this instance.
    ///
    /// Propagated from [`crate::config::Config::gpu_count`]. Used by
    /// `EmbedPool::spawn` to compute per-worker `device_id` values and to
    /// clamp `BGE_M3_WORKERS` for GPU execution providers.
    pub gpu_count: usize,

    /// Optional TRT workspace size cap (bytes) forwarded to ORT's TRT EP via
    /// `with_max_workspace_size`.  `None` uses ORT's built-in default.
    /// Sourced from `BGE_M3_TRT_MAX_WORKSPACE_BYTES`.
    pub trt_max_workspace_bytes: Option<usize>,

    /// Optional CUDA device memory limit (bytes) forwarded to the CUDA EP.
    /// `None` uses ORT's built-in default.
    pub gpu_mem_limit_bytes: Option<usize>,

    /// Sender half of the JIT-suspect channel created before pool spawn.
    ///
    /// After each successful inference, if `inference_ms >= CHUNK_CACHE_HIT_THRESHOLD_MS`
    /// the worker calls `try_send((batch, seq))` so the adaptive warmup task
    /// can schedule background engine compilation.  Non-blocking: if the
    /// channel is full the message is silently dropped.  `None` when adaptive
    /// warmup is disabled.
    pub jit_suspect_tx: Option<JitSuspectSender>,

    /// Sender half of the engine propagation broadcast channel.
    ///
    /// When `Some`, after any worker writes a new TRT engine plan to EFS, the
    /// worker broadcasts the `(batch, seq)` shape to all subscribed peers so
    /// they eagerly run `trt_prewarm` (~1-3s fast disk-load) instead of paying
    /// full JIT cost on the next real request.  Each worker derives its own
    /// `Receiver` via `tx.subscribe()` at startup.  `None` when
    /// `BGE_M3_ENGINE_PROPAGATION_ENABLED=0` or when the EP is not TRT.
    pub engine_propagation_tx: Option<tokio::sync::broadcast::Sender<(usize, usize)>>,

    /// When `true`, prewarm postcondition failures cause the worker to refuse
    /// to signal ready: `run_worker` returns `Err(_)` before `ready_tx.send`,
    /// the pool's init task propagates the error, and the readiness probe in
    /// `bootstrap::readiness` triggers a hard process exit. Converts the
    /// false-positive-readiness failure mode (every worker hits TRT
    /// `Error Code 10` mid-build, postcondition logs WARN, `/health` still
    /// returns `200 ok`, real requests then 500) into an explicit startup
    /// failure that ECS retries instead of routing traffic to.
    ///
    /// Sourced from `BGE_M3_PREWARM_STRICT`; defaults to `true`. Set to
    /// `false` to preserve pre-fix behaviour (WARN only).
    pub prewarm_strict: bool,

    /// Consecutive-failure threshold for the per-worker inference circuit
    /// breaker.
    ///
    /// When a worker accumulates this many consecutive errors from an
    /// inference call (`embed_dense`, `embed_sparse`, or `embed_both`), it
    /// drops its ORT session (cleaning the CUDA arena), decrements
    /// `loaded_workers`, and waits for the next request to trigger a model
    /// reload. The counter resets to zero on any successful inference.
    ///
    /// Sourced from `BGE_M3_CIRCUIT_BREAKER_THRESHOLD`; defaults to `5`.
    pub circuit_breaker_threshold: usize,

    /// Enables the in-band TRT JIT admission guard (see [`super::jit_guard`]).
    ///
    /// When `true` (and `ep == TensorRt`), before any chunk's `session.run()`
    /// the worker refuses chunks whose sequence length is in the dangerous
    /// range and is not covered by the pool's warmed engine profile, returning
    /// an error that maps to HTTP `503` instead of risking the process-killing
    /// pathological autotuner allocation. Sourced from
    /// `BGE_M3_TRT_INBAND_JIT_GUARD`; defaults to `true`.
    pub trt_inband_jit_guard_enabled: bool,

    /// Sequence-length threshold (`guard_seq`) at/above which an *uncovered*
    /// shape is refused rather than JIT-compiled in-band. Below this value a
    /// cold in-band JIT is bounded and is allowed (letting the engine profile
    /// grow naturally). Sourced from `BGE_M3_TRT_INBAND_JIT_GUARD_SEQ`;
    /// defaults to `4096`. The guard is a no-op when `max_seq_length` is below
    /// this threshold (no chunk can reach the dangerous range).
    pub trt_inband_jit_guard_seq: usize,

    /// Pool-wide ceiling: the maximum sequence length any worker has
    /// successfully warmed (fresh compile or warm-cache hit), shared across
    /// all workers via the `Arc<AtomicUsize>`.
    ///
    /// Read by the per-request [`super::jit_guard::TrtJitGuard`]; raised via
    /// [`fetch_max`](std::sync::atomic::AtomicUsize::fetch_max) after startup
    /// prewarm, engine-propagation prewarm, and adaptive-warmup compiles. A
    /// successful warmup of a sequence tier by *any* worker means the engine
    /// plan is on the shared EFS cache and every worker can fast-load it, so a
    /// single shared ceiling is a sound pool-wide coverage signal.
    pub warmed_seq_ceiling: Arc<AtomicUsize>,

    /// **Destructive** stale-SM TRT engine cache GC flag.
    ///
    /// Only present when the `cache-gc` Cargo feature is compiled in.
    /// Defaults to `false` even with the feature on; flipping the
    /// runtime knob also requires `BGE_M3_TRT_CACHE_GC_ENABLED=1`. See
    /// [`crate::config::Config::trt_cache_gc_enabled`] for the multi-SM
    /// ASG hazard model.
    #[cfg(feature = "cache-gc")]
    pub trt_cache_gc_enabled: bool,
}
