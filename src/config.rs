use crate::binpack::CostModel;
use std::env;
use std::time::Duration;
use tracing::warn;

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
}

impl Config {
    /// Creates a [`Config`] by reading environment variables.
    ///
    /// Unrecognized or missing variables fall back to their defaults.
    pub fn from_env() -> Self {
        Self::from_lookup(|key| env::var(key).ok())
    }

    #[allow(clippy::too_many_lines)]
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
        }
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
