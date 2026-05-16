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

//! Server configuration loaded from environment variables at startup.
//!
//! All settings are read once via [`Config::from_env`] and then immutable
//! for the server's lifetime. See each field's doc comment for the
//! corresponding environment variable name and default value.

use crate::binpack::CostModel;
use crate::sysinfo;
use std::env;
use std::time::Duration;
use tracing::{info, warn};

/// ONNX model variant to load.
///
/// Controlled by `BGE_M3_MODEL`. Defaults to [`ModelVariant::Fp16`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// BAAI/bge-m3 FP32 model (~2.16 GB per session).
    ///
    /// Set `BGE_M3_MODEL=fp32` to enable. Recommended for Apple Silicon `CoreML`
    /// deployments where latency is the primary constraint: the FP32 ONNX graph
    /// contains no Cast nodes, so ORT can dispatch the entire multi-head
    /// attention + FFN block as one contiguous `CoreML` subgraph to the GPU —
    /// delivering 20–61% lower latency than the MLAS CPU baseline.
    ///
    /// **Not the default.** Linux/Intel (MLAS-only) deployments should prefer
    /// [`ModelVariant::Fp16`] for lower RAM and fleet-wide embedding consistency.
    Fp32,
    /// Xenova/bge-m3 FP16 model (~1.08 GB per session). **Default.**
    /// Halves per-session memory vs FP32 (~50% reduction; ~1.08 GB vs ~2.16 GB).
    ///
    /// This is the fleet default: all Apple Silicon `LaunchAgent` deployments set
    /// `BGE_M3_MODEL=fp16` explicitly, and the server default matches so that
    /// Linux/Docker deployments produce consistent embeddings without any
    /// additional configuration.
    ///
    /// **Latency caveat (`CoreML` only).** The Xenova FP16 ONNX model contains
    /// FP16↔FP32 Cast nodes at every transformer-layer boundary. ORT's `CoreML` EP
    /// cannot fuse these into the attention/FFN subgraphs; each Cast executes on
    /// CPU and the transformer block never forms a single contiguous GPU subgraph.
    /// Result: FP16 + `CoreML` EP runs 6–10× slower than FP32 + `CoreML`. On
    /// MLAS/CPU EP (Linux, Intel), this Cast overhead is similarly present but
    /// the MLAS FP16 penalty (~6–9×) is the accepted trade-off for lower RAM and
    /// fleet consistency. Use `BGE_M3_MODEL=fp32` on Apple Silicon to recover
    /// `CoreML` GPU acceleration.
    Fp16,
    /// Xenova/bge-m3 INT8 quantized model (~568 MB per session).
    /// Weights-only quantization; ORT dequantizes to f32 internally.
    /// Reduces peak memory by ~74% per worker vs FP32.
    ///
    /// Embedding quality validated: dense cosine similarity ≥ 0.963 vs FP32
    /// reference across a 184-text corpus — suitable for ANN search and semantic
    /// ranking. Avoid for applications requiring ranking precision within very
    /// small similarity margins (< 0.05 apart).
    ///
    /// **Use with MLAS (CPU EP) only.** `DequantizeLinear` nodes fragment the
    /// `CoreML` execution plan identically to FP16 Cast nodes; INT8 + `CoreML` EP
    /// runs 42–79% slower than INT8 + MLAS with no GPU benefit.
    Int8,
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp32 => f.write_str("fp32"),
            Self::Fp16 => f.write_str("fp16"),
            Self::Int8 => f.write_str("int8"),
        }
    }
}

/// VRAM workspace budget used for GPU EPs when `BGE_M3_GPU_VRAM_BUDGET_BYTES` is unset.
///
/// 10 GiB is a conservative ceiling for NVIDIA GPUs with ≥ 16 GiB VRAM (A10G, L4, H100 80GB).
/// Override with `BGE_M3_GPU_VRAM_BUDGET_BYTES` for GPUs with less VRAM.
const DEFAULT_GPU_VRAM_BUDGET_BYTES: usize = 10 * 1024 * 1024 * 1024;

/// ONNX Runtime execution provider selection.
///
/// Controlled by `BGE_M3_EP`. Defaults to [`EpSelection::Cpu`].
///
/// On macOS the `CoreML` EP is always used regardless of this setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpSelection {
    /// CPU inference via MLAS (default). Works everywhere, no GPU required.
    Cpu,
    /// NVIDIA CUDA execution provider (requires `cuda` feature and a CUDA ORT build).
    ///
    /// Set `BGE_M3_EP=cuda` to enable. `BGE_M3_WORKERS` is clamped to
    /// `BGE_M3_GPU_COUNT` so each worker is pinned to a distinct CUDA device.
    /// Set `BGE_M3_GPU_COUNT` to match the number of GPUs on the instance for
    /// maximum parallel inference throughput.
    Cuda,
    /// NVIDIA `TensorRT` execution provider (requires `tensorrt` feature and a TRT ORT build).
    ///
    /// Set `BGE_M3_EP=tensorrt` to enable. Falls back to CUDA for ops TRT cannot
    /// handle. `BGE_M3_WORKERS` is clamped to `BGE_M3_GPU_COUNT`; each worker
    /// compiles its own per-GPU TRT shard of the warmup shapes during startup,
    /// enabling parallel engine compilation across GPUs.
    TensorRt,
}

impl std::fmt::Display for EpSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda => f.write_str("cuda"),
            Self::TensorRt => f.write_str("tensorrt"),
        }
    }
}

/// Maximum sequence length supported by the model architecture.
/// BGE-M3's positional embedding table extends to 8192; this is the hard upper
/// bound used to validate `BGE_M3_MAX_SEQ_LENGTH`.
pub const MODEL_MAX_SEQ: usize = 8192;

/// Runtime configuration loaded from environment variables.
///
/// All fields are read once at startup via [`Config::from_env`]. Changes to
/// environment variables after startup have no effect.
pub struct Config {
    /// Path to the directory where ONNX model files are cached.
    ///
    /// Set with `BGE_M3_CACHE_DIR`. Defaults to `/cache`.
    pub cache_dir: String,
    /// TCP bind address for the HTTP server.
    ///
    /// Set with `BGE_M3_BIND`. Defaults to `0.0.0.0:8081`.
    /// The `0.0.0.0` default is intentional for Docker container deployments.
    pub bind_addr: String,
    /// Number of embedding worker threads to spawn.
    ///
    /// Set with `BGE_M3_WORKERS`. Defaults to `2`. Minimum effective value is `1`.
    /// Each worker loads its own model instance.
    pub workers: usize,
    /// Number of intra-op threads each ORT session may use for a single
    /// `session.run()` call (matmul / attention kernels).
    ///
    /// Set with `BGE_M3_INTRA_THREADS`. Defaults to `1`. Minimum effective
    /// value is `1`.
    ///
    /// The default of `1` preserves predictable per-worker RSS (the workspace
    /// probe and quadratic cost model are calibrated against single-threaded
    /// MLAS runs). Raise this on under-utilized hosts where `BGE_M3_WORKERS *
    /// intra_threads <= num_cpus`: e.g. on an 8 vCPU task with `workers=2`,
    /// setting `intra_threads=4` lets each worker fan out to four cores during
    /// inference, taking CPU utilization from ~25% to ~100% under load. Going
    /// above `floor(num_cpus / workers)` causes thread oversubscription and
    /// hurts throughput.
    ///
    /// Re-run the startup probe (do not pin coefficients) after changing this
    /// value so the cost model captures any new scratch-buffer overhead.
    pub intra_threads: usize,
    /// Maximum number of input texts accepted in a single request.
    ///
    /// Set with `BGE_M3_MAX_BATCH`. Defaults to `256`. Minimum effective value is `1`.
    pub max_batch: usize,
    /// Maximum sequence length (tokens) for a single text.
    ///
    /// Set with `BGE_M3_MAX_SEQ_LENGTH`. Defaults to `8192` (BGE-M3's published max).
    /// Range: `[1, 8192]`. Set lower to reduce memory footprint on constrained hardware.
    ///
    /// The tokenizer will silently truncate any input exceeding this length.
    /// The probe and bin-packer use this as the upper bound when computing
    /// workspace costs.
    pub max_seq_length: usize,
    /// Duration of inactivity after which workers unload their model instances from memory.
    ///
    /// Set with `BGE_M3_IDLE_TIMEOUT_SECS`. Defaults to `300` (5 minutes).
    /// Set to `0` to disable idle unloading entirely.
    ///
    /// When unloaded, models are automatically reloaded on the next incoming request.
    /// The reload blocks the request until complete (~5–10 s from `CoreML` compiled
    /// cache; ~15–30 s cold).
    pub idle_timeout: Option<Duration>,
    /// ONNX model variant to load.
    ///
    /// Set with `BGE_M3_MODEL`. Accepts `"fp32"`, `"fp16"`, or `"int8"`.
    /// Defaults to `"fp16"` for fleet-wide embedding consistency and reduced RAM
    /// on Linux/Intel deployments. Set `BGE_M3_MODEL=fp32` on Apple Silicon to
    /// recover `CoreML` GPU acceleration. See [`ModelVariant`] for per-variant
    /// performance and memory trade-offs.
    pub model_variant: ModelVariant,

    // --- auto-budget and cost-model knobs ---
    /// Fraction of estimated available workspace to actually use per worker.
    ///
    /// Set with `BGE_M3_MEMORY_SAFETY_FACTOR`. Defaults to `0.7` (30% headroom
    /// for ORT arena fragmentation and spike overhead not captured by the probe).
    /// Range: `0.1..=1.0`.
    pub memory_safety_factor: f64,

    /// If `Some`, skip the startup probe and use this cost model directly.
    ///
    /// Populated when:
    /// - `BGE_M3_DISABLE_AUTO_BUDGET=1` is set (uses conservative defaults), or
    /// - `BGE_M3_TOKEN_BUDGET` is set (translates the legacy token count to a
    ///   `max_workspace_bytes` using conservative `a`/`b` coefficients), or
    /// - `BGE_M3_COST_MODEL_A` and `BGE_M3_COST_MODEL_B` are both set with
    ///   `BGE_M3_AVAILABLE_MEMORY_BYTES` (full explicit override).
    pub cost_model_override: Option<CostModel>,
    /// Interval (seconds) between periodic heartbeat log events.
    ///
    /// Set with `BGE_M3_HEARTBEAT_SECS`. Defaults to `60`.
    /// Set to `0` to disable heartbeat logging entirely.
    ///
    /// Heartbeat events log RSS, live/loaded worker counts, queue depth,
    /// available request permits, and current probe status — useful for
    /// detecting slow memory leaks or queue saturation between requests.
    pub heartbeat_secs: u64,

    /// ONNX Runtime execution provider to use.
    ///
    /// Set with `BGE_M3_EP`. Accepts `"cpu"`, `"cuda"`, or `"tensorrt"`.
    /// Defaults to `"cpu"`. On macOS, `CoreML` is always used regardless of
    /// this setting. Requires the corresponding Cargo feature (`cuda` or
    /// `tensorrt`) to be enabled at build time for GPU EPs.
    ///
    /// When set to `"cuda"` or `"tensorrt"`, the host-RAM probe is bypassed
    /// in favour of the VRAM budget, and `BGE_M3_WORKERS` is clamped to
    /// `BGE_M3_GPU_COUNT` in `EmbedPool::spawn`.
    pub ep: EpSelection,

    /// Number of GPU devices available on this instance.
    ///
    /// Set with `BGE_M3_GPU_COUNT`. When a GPU execution provider (`cuda` or
    /// `tensorrt`) is active, `BGE_M3_WORKERS` is clamped to this value in
    /// `EmbedPool::spawn` so each worker is pinned to a distinct CUDA device
    /// (`device_id = worker_index % gpu_count`).
    ///
    /// Auto-detected on Linux from `/proc/driver/nvidia/gpus/` entry count.
    /// Defaults to `1` on macOS (`CoreML` is always single-device) and on
    /// Linux when the driver proc path is absent. Override explicitly on
    /// multi-GPU ECS instances: `BGE_M3_GPU_COUNT=8`.
    pub gpu_count: usize,

    /// VRAM workspace ceiling (bytes) when a GPU execution provider is active.
    ///
    /// Set with `BGE_M3_GPU_VRAM_BUDGET_BYTES`. Ignored when `ep == Cpu`.
    /// Defaults to `None`, which causes the server to use 10 GiB as the
    /// ceiling (suitable for GPUs with ≥ 16 GiB VRAM such as A10G / L4).
    /// Lower this on GPUs with less VRAM (e.g. `8589934592` for 8 GiB).
    pub gpu_vram_budget_bytes: Option<usize>,

    /// TRT EP workspace size cap in bytes.
    ///
    /// Set with `BGE_M3_TRT_MAX_WORKSPACE_BYTES`. When `None`, TRT uses its
    /// default "as large as possible" workspace — which can OOM on saturated
    /// VRAM. Set to a value that leaves room for resident model weights and
    /// cached engine plans (e.g. 4 GiB = `4294967296` on an L40S with 4 workers).
    pub trt_max_workspace_bytes: Option<usize>,

    /// CUDA EP device-level memory limit in bytes.
    ///
    /// Set with `BGE_M3_GPU_MEM_LIMIT_BYTES`. When `None`, CUDA EP uses all
    /// available device memory. Symmetric to `trt_max_workspace_bytes`.
    pub gpu_mem_limit_bytes: Option<usize>,

    /// Enable the in-process adaptive background warmup loop.
    ///
    /// Set with `BGE_M3_ADAPTIVE_WARMUP_ENABLED=1`. When enabled, the server
    /// detects unseen `(batch, seq)` shapes during live inference and compiles
    /// TRT engines for them during idle windows.
    pub adaptive_warmup_enabled: bool,

    /// Seconds the server must be idle (`queue_depth == 0`, all workers free)
    /// before the adaptive warmup loop fires a shape.
    ///
    /// Set with `BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS`. Default: 3.
    pub adaptive_warmup_quiet_secs: u64,

    /// Maximum number of shapes the adaptive warmup loop may compile per hour.
    ///
    /// Set with `BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR`. Default: 12.
    /// Prevents pathological traffic patterns from compiling indefinitely.
    pub adaptive_warmup_max_shapes_per_hour: u32,

    /// List of `(batch_size, seq_len)` shapes to pre-compile as `TensorRT` engine
    /// files during worker startup.
    ///
    /// Set with `BGE_M3_TRT_WARMUP_SHAPES` as a comma-separated list of `BxL`
    /// tokens (e.g. `"1x128,4x512,16x2048,32x8192"`). Only used when
    /// `ep == EpSelection::TensorRt`. Invalid tokens are skipped with a `WARN`.
    /// An empty or all-invalid value falls back to the default set. Operators
    /// can shrink the grid for local development (e.g. `"1x128"`) — the env
    /// var override path is the canonical way to keep cold-start tractable on
    /// workstations.
    ///
    /// Default: a 2D `{1, 4, 16, 32} × {128, 512, 2048, 8192}` grid composed
    /// in batch-major order so the smallest batches finish first and the
    /// most common router shapes (`chunks = 1–2 × max_chunk_seq up to ~6000`)
    /// hit a pre-compiled engine on the very first real request. The expensive
    /// `_ × 8192` shapes compile last (~30–170 s each) — total cold-cache
    /// compile budget is roughly 6–12 minutes on first deploy. Subsequent
    /// starts on the same EC2 instance reuse cached engine files (seconds).
    ///
    /// Each shape may take 30–170 s to compile on the first run; the worker
    /// signals ready only after all shapes finish, so `/health` returns `503`
    /// during this window.
    pub trt_warmup_shapes: Vec<(usize, usize)>,

    /// Exit cleanly after `TensorRT` engine compilation and cache flush.
    ///
    /// Set with `BGE_M3_WARMUP_ONLY`. Default `false`.
    ///
    /// When `true` the server initialises the model and ORT session exactly as
    /// normal — loading ONNX weights, configuring the TRT EP, and running the
    /// pre-warm shape compilation via the existing warmup path. After all
    /// engines have been compiled and fsynced to the EFS cache the process
    /// logs a single `INFO` line and calls `process::exit(0)`. No TCP listener
    /// is bound; the HTTP server never starts.
    ///
    /// Primary use-case: ECS init container that pre-populates the shared EFS
    /// engine cache before the main container starts, so the main container
    /// always sees a warm cache and skips the 6–12 minute cold-compile window.
    ///
    /// A `WARN` is logged if this flag is set with `BGE_M3_EP` other than
    /// `tensorrt` — warmup-only on CPU is a no-op (there is nothing to compile)
    /// but the server still exits 0 cleanly rather than erroring.
    pub warmup_only: bool,
}

impl Config {
    /// Creates a [`Config`] by reading environment variables.
    ///
    /// Unrecognized or missing variables fall back to their defaults.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_lookup(|key| env::var(key).ok())
    }

    #[allow(clippy::too_many_lines)]
    /// Creates a [`Config`] by resolving each setting through `lookup`.
    ///
    /// `lookup` receives an env-var name and returns its value if set, or
    /// `None` to fall back to the default for that setting. Used by
    /// [`Config::from_env`] with the real environment and in tests with a
    /// closure over a `HashMap`.
    pub(crate) fn from_lookup<F: Fn(&str) -> Option<String>>(lookup: F) -> Self {
        let workers = lookup("BGE_M3_WORKERS")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);

        let intra_threads = lookup("BGE_M3_INTRA_THREADS")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);

        let max_batch = lookup("BGE_M3_MAX_BATCH")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            .max(1);

        let max_seq_length = {
            let raw = lookup("BGE_M3_MAX_SEQ_LENGTH")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(MODEL_MAX_SEQ);
            if raw == 0 || raw > MODEL_MAX_SEQ {
                warn!(
                    requested = raw,
                    clamped = MODEL_MAX_SEQ,
                    "BGE_M3_MAX_SEQ_LENGTH out of range [1, {MODEL_MAX_SEQ}]; clamping"
                );
                MODEL_MAX_SEQ
            } else {
                raw
            }
        };

        let idle_timeout_secs = lookup("BGE_M3_IDLE_TIMEOUT_SECS")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        let idle_timeout = (idle_timeout_secs > 0).then(|| Duration::from_secs(idle_timeout_secs));

        let model_variant = match lookup("BGE_M3_MODEL").as_deref() {
            Some("fp32") => ModelVariant::Fp32,
            Some("int8") => ModelVariant::Int8,
            _ => ModelVariant::Fp16,
        };

        let memory_safety_factor = {
            let raw = lookup("BGE_M3_MEMORY_SAFETY_FACTOR")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.7);
            raw.clamp(0.1, 1.0)
        };

        // --- cost model override resolution ---
        // Priority:
        //  1. BGE_M3_DISABLE_AUTO_BUDGET → conservative defaults
        //  2. BGE_M3_TOKEN_BUDGET (legacy) → translates to max_workspace_bytes
        //  3. BGE_M3_COST_MODEL_A + BGE_M3_COST_MODEL_B + BGE_M3_AVAILABLE_MEMORY_BYTES
        //  4. None → probe at startup

        let cost_model_override = resolve_cost_model_override(&lookup, max_seq_length);

        // --- legacy BGE_M3_ONNX_BATCH_SIZE deprecation ---
        if lookup("BGE_M3_ONNX_BATCH_SIZE").is_some() {
            warn!(
                "BGE_M3_ONNX_BATCH_SIZE is deprecated and will be removed in a future release. \
                 The server now uses a quadratic-aware cost model and auto-budget probe. \
                 Set BGE_M3_TOKEN_BUDGET to pin a specific workspace ceiling, or remove the \
                 variable to enable fully automatic tuning."
            );
        }

        let heartbeat_secs = lookup("BGE_M3_HEARTBEAT_SECS")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(60);

        let ep = match lookup("BGE_M3_EP").as_deref() {
            Some("cuda") => EpSelection::Cuda,
            Some("tensorrt") => EpSelection::TensorRt,
            _ => EpSelection::Cpu,
        };

        let gpu_vram_budget_bytes =
            lookup("BGE_M3_GPU_VRAM_BUDGET_BYTES").and_then(|v| v.parse::<usize>().ok());

        let trt_max_workspace_bytes = lookup("BGE_M3_TRT_MAX_WORKSPACE_BYTES").and_then(|v| {
            v.parse::<usize>()
                .inspect_err(|e| {
                    tracing::warn!(
                        raw = %v,
                        error = %e,
                        "BGE_M3_TRT_MAX_WORKSPACE_BYTES parse failed — TRT workspace cap disabled"
                    );
                })
                .ok()
        });

        let gpu_mem_limit_bytes = lookup("BGE_M3_GPU_MEM_LIMIT_BYTES").and_then(|v| {
            v.parse::<usize>()
                .inspect_err(|e| {
                    tracing::warn!(
                        raw = %v,
                        error = %e,
                        "BGE_M3_GPU_MEM_LIMIT_BYTES parse failed — CUDA memory limit disabled"
                    );
                })
                .ok()
        });

        let adaptive_warmup_enabled = lookup("BGE_M3_ADAPTIVE_WARMUP_ENABLED")
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"));

        let adaptive_warmup_quiet_secs = lookup("BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(3);

        let adaptive_warmup_max_shapes_per_hour =
            lookup("BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(12);

        // When a GPU EP is active, the host-RAM probe is meaningless — VRAM is
        // the constraint. Override the cost model unconditionally so the probe
        // is skipped and the VRAM budget drives bin-packing instead.
        let cost_model_override = if ep == EpSelection::Cpu {
            cost_model_override
        } else {
            let vram_budget = gpu_vram_budget_bytes.unwrap_or(DEFAULT_GPU_VRAM_BUDGET_BYTES);
            info!(
                ep = %ep,
                vram_budget_bytes = vram_budget,
                "GPU execution provider selected — bypassing host-RAM probe; \
                 using VRAM budget as the workspace ceiling"
            );
            Some(CostModel::conservative(vram_budget))
        };

        let gpu_count = sysinfo::detect_gpu_count(
            lookup("BGE_M3_GPU_COUNT").and_then(|v| v.parse::<usize>().ok()),
        );

        let trt_warmup_shapes = parse_trt_warmup_shapes(lookup("BGE_M3_TRT_WARMUP_SHAPES"));

        let warmup_only = lookup("BGE_M3_WARMUP_ONLY")
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"));

        if warmup_only && ep != EpSelection::TensorRt {
            warn!(
                ep = %ep,
                "BGE_M3_WARMUP_ONLY=1 is set but BGE_M3_EP is not tensorrt — \
                 warmup-only is a no-op on non-TRT EPs (nothing to compile); \
                 the server will exit 0 without performing any engine compilation"
            );
        }

        Self {
            cache_dir: lookup("BGE_M3_CACHE_DIR").unwrap_or_else(|| "/cache".to_string()),
            bind_addr: lookup("BGE_M3_BIND").unwrap_or_else(|| "0.0.0.0:8081".to_string()),
            workers,
            intra_threads,
            max_batch,
            max_seq_length,
            idle_timeout,
            model_variant,
            memory_safety_factor,
            cost_model_override,
            heartbeat_secs,
            ep,
            gpu_vram_budget_bytes,
            trt_max_workspace_bytes,
            gpu_mem_limit_bytes,
            adaptive_warmup_enabled,
            adaptive_warmup_quiet_secs,
            adaptive_warmup_max_shapes_per_hour,
            gpu_count,
            trt_warmup_shapes,
            warmup_only,
        }
    }
}

/// Default TRT warmup shapes: a 2D `{1, 4, 16, 32} × {128, 512, 2048, 8192}`
/// grid composed in batch-major order.
///
/// Batch is the outer dimension so the smallest batches (which dominate
/// real router traffic — `chunks = 1–2 × max_chunk_seq`) are fully compiled
/// first; larger batches that are mostly observed during bulk re-indexing
/// fill in afterwards. Within each batch the sequence dimension grows
/// monotonically so the cheap `_ × 128` shape comes before the expensive
/// `_ × 8192` shape — operators watching `/health` see progress quickly.
///
/// Production observation (2026-05 codekeeper investigation): real router
/// traffic shows `chunks = 1–2 × max_chunk_seq` up to ~6 000 token positions.
/// Previously-unseen shapes triggered in-band engine compilation in the middle
/// of a real request, producing 56–356 s `inference_ms` values. This 16-shape
/// grid covers the full realistic shape space so every router request hits
/// a pre-compiled engine.
const DEFAULT_TRT_WARMUP_SHAPES: &[(usize, usize)] = &[
    (1, 128),
    (1, 512),
    (1, 2048),
    (1, 8192),
    (4, 128),
    (4, 512),
    (4, 2048),
    (4, 8192),
    (16, 128),
    (16, 512),
    (16, 2048),
    (16, 8192),
    (32, 128),
    (32, 512),
    (32, 2048),
    (32, 8192),
];

/// Parses `BGE_M3_TRT_WARMUP_SHAPES` from its raw env-var value.
///
/// Accepts a comma-separated list of `BxL` tokens (e.g. `"1x128,1x512"`).
/// Invalid tokens are skipped with a `WARN`. Returns the default shape set
/// when `raw` is `None`, empty, or all tokens are invalid.
pub(crate) fn parse_trt_warmup_shapes(raw: Option<String>) -> Vec<(usize, usize)> {
    let Some(val) = raw else {
        return DEFAULT_TRT_WARMUP_SHAPES.to_vec();
    };
    if val.trim().is_empty() {
        return DEFAULT_TRT_WARMUP_SHAPES.to_vec();
    }
    let parsed: Vec<(usize, usize)> = val
        .split(',')
        .filter_map(|token| {
            let token = token.trim();
            let mut parts = token.splitn(2, 'x');
            let batch = parts.next()?.parse::<usize>().ok()?;
            let seq = parts.next()?.parse::<usize>().ok()?;
            Some((batch, seq))
        })
        .collect();

    if parsed.is_empty() {
        warn!(
            raw = %val,
            "BGE_M3_TRT_WARMUP_SHAPES contained no valid BxL tokens; \
             falling back to default warmup shapes"
        );
        DEFAULT_TRT_WARMUP_SHAPES.to_vec()
    } else {
        parsed
    }
}

/// Resolves an optional `CostModel` from env vars that explicitly override auto-tuning.
///
/// Returns `None` when the server should run the startup probe.
//
// cast_precision_loss: token_budget and max_seq_length are small integers (≤ 8192)
//   that are well within f64 mantissa range; cost-per-position is an estimate.
// cast_possible_truncation / cast_sign_loss: the workspace result is always positive
//   (products of positive coefficients and non-negative token counts), and fractional
//   bytes are intentionally floored when converting back to usize.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn resolve_cost_model_override<F: Fn(&str) -> Option<String>>(
    lookup: &F,
    max_seq_length: usize,
) -> Option<CostModel> {
    // 1. BGE_M3_DISABLE_AUTO_BUDGET — skip probe, use conservative defaults.
    //    max_workspace_bytes comes from BGE_M3_AVAILABLE_MEMORY_BYTES if set,
    //    otherwise uses the built-in default (2 GiB).
    if lookup("BGE_M3_DISABLE_AUTO_BUDGET")
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
    {
        let max_workspace = lookup("BGE_M3_AVAILABLE_MEMORY_BYTES")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(CostModel::DEFAULT_MAX_WORKSPACE);
        return Some(CostModel::conservative(max_workspace));
    }

    // 2. BGE_M3_TOKEN_BUDGET — legacy token-count ceiling.
    //    Translates: max_workspace = token_budget × cost_per_token
    //    using conservative coefficients at the configured max_seq_length.
    if let Some(token_budget) = lookup("BGE_M3_TOKEN_BUDGET").and_then(|v| v.parse::<usize>().ok())
    {
        // cost_per_position at max_seq = a + b * max_seq
        let cost_per_position =
            CostModel::CONSERVATIVE_A + CostModel::CONSERVATIVE_B * max_seq_length as f64;
        let max_workspace = (token_budget as f64 * cost_per_position) as usize;
        return Some(CostModel {
            a: CostModel::CONSERVATIVE_A,
            b: CostModel::CONSERVATIVE_B,
            max_workspace_bytes: max_workspace,
        });
    }

    // 3. Explicit coefficient override — requires A, B, AND available memory.
    if let (Some(a_str), Some(b_str)) =
        (lookup("BGE_M3_COST_MODEL_A"), lookup("BGE_M3_COST_MODEL_B"))
    {
        if let (Ok(a), Ok(b)) = (a_str.parse::<f64>(), b_str.parse::<f64>()) {
            let max_workspace = lookup("BGE_M3_AVAILABLE_MEMORY_BYTES")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(CostModel::DEFAULT_MAX_WORKSPACE);
            return Some(CostModel {
                a,
                b,
                max_workspace_bytes: max_workspace,
            });
        }
    }

    // 4. No override — run the startup probe.
    None
}

#[cfg(test)]
mod tests;
