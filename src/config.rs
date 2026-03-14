use std::env;
use std::time::Duration;

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
    /// Maximum number of input texts accepted in a single request.
    ///
    /// Set with `BGE_M3_MAX_BATCH`. Defaults to `256`. Minimum effective value is `1`.
    pub max_batch: usize,
    /// Maximum number of texts submitted per ONNX `session.run()` call.
    ///
    /// Set with `BGE_M3_ONNX_BATCH_SIZE`. On macOS the `CoreML` execution provider
    /// uses `MLProgram` with `FastPrediction` specialisation, which pre-allocates
    /// the full intermediate-tensor workspace at model-compilation time. BGE-M3
    /// (24 transformer layers, 16 attention heads, 1 024 hidden) produces a
    /// peak workspace of roughly `batch × 720 MB` per layer for a 512-token
    /// sequence; submitting 50 texts at once can require 35 GB, triggering
    /// Jetsam OOM kills. Chunking to 8 texts per call caps the peak at ~5.6 GB.
    ///
    /// On other platforms (MLAS/CPU EP) no pre-allocation occurs, so larger
    /// values improve throughput with no stability risk.
    ///
    /// Defaults to `8` on macOS, `256` elsewhere. Minimum effective value is `1`.
    pub onnx_batch_size: usize,
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
}

impl Config {
    /// Creates a [`Config`] by reading environment variables.
    ///
    /// Unrecognized or missing variables fall back to their defaults.
    pub fn from_env() -> Self {
        Self::from_lookup(|key| env::var(key).ok())
    }

    fn from_lookup<F: Fn(&str) -> Option<String>>(lookup: F) -> Self {
        let workers = lookup("BGE_M3_WORKERS")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);

        let max_batch = lookup("BGE_M3_MAX_BATCH")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            .max(1);

        // Platform-specific default: CoreML on macOS pre-allocates the full
        // intermediate-tensor workspace, so a small default avoids OOM kills.
        let onnx_batch_size_default: usize = if cfg!(target_os = "macos") { 8 } else { 256 };
        let onnx_batch_size = lookup("BGE_M3_ONNX_BATCH_SIZE")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(onnx_batch_size_default)
            .max(1);

        let idle_timeout_secs = lookup("BGE_M3_IDLE_TIMEOUT_SECS")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        let idle_timeout = (idle_timeout_secs > 0).then(|| Duration::from_secs(idle_timeout_secs));

        let model_variant = match lookup("BGE_M3_MODEL").as_deref() {
            Some("fp32") => ModelVariant::Fp32,
            Some("int8") => ModelVariant::Int8,
            _ => ModelVariant::Fp16,
        };

        Self {
            cache_dir: lookup("BGE_M3_CACHE_DIR").unwrap_or_else(|| "/cache".to_string()),
            bind_addr: lookup("BGE_M3_BIND").unwrap_or_else(|| "0.0.0.0:8081".to_string()),
            workers,
            max_batch,
            onnx_batch_size,
            idle_timeout,
            model_variant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup_from<'a>(map: &'a HashMap<&'a str, &'a str>) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| map.get(key).map(|&v| v.to_string())
    }

    #[test]
    fn defaults_without_env_vars() {
        let map = HashMap::new();
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.cache_dir, "/cache");
        assert_eq!(cfg.bind_addr, "0.0.0.0:8081");
        assert_eq!(cfg.workers, 2);
        assert_eq!(cfg.max_batch, 256);
        assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(300)));
        assert_eq!(cfg.model_variant, ModelVariant::Fp16);
        // Platform-specific: macOS=8 (CoreML OOM guard), other=256
        let expected_onnx: usize = if cfg!(target_os = "macos") { 8 } else { 256 };
        assert_eq!(cfg.onnx_batch_size, expected_onnx);
    }

    #[test]
    fn workers_clamps_to_minimum_1() {
        let map = HashMap::from([("BGE_M3_WORKERS", "0")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.workers, 1);
    }

    #[test]
    fn max_batch_clamps_to_minimum_1() {
        let map = HashMap::from([("BGE_M3_MAX_BATCH", "0")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.max_batch, 1);
    }

    #[test]
    fn custom_values_are_applied() {
        let map = HashMap::from([
            ("BGE_M3_CACHE_DIR", "/tmp/models"),
            ("BGE_M3_BIND", "127.0.0.1:9090"),
            ("BGE_M3_WORKERS", "4"),
            ("BGE_M3_MAX_BATCH", "128"),
            ("BGE_M3_IDLE_TIMEOUT_SECS", "600"),
        ]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.cache_dir, "/tmp/models");
        assert_eq!(cfg.bind_addr, "127.0.0.1:9090");
        assert_eq!(cfg.workers, 4);
        assert_eq!(cfg.max_batch, 128);
        assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(600)));
    }

    #[test]
    fn idle_timeout_defaults_to_5_minutes() {
        let map = HashMap::new();
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(300)));
    }

    #[test]
    fn idle_timeout_disabled_when_zero() {
        let map = HashMap::from([("BGE_M3_IDLE_TIMEOUT_SECS", "0")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.idle_timeout, None);
    }

    #[test]
    fn idle_timeout_custom_value() {
        let map = HashMap::from([("BGE_M3_IDLE_TIMEOUT_SECS", "60")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn onnx_batch_size_uses_platform_default() {
        let map = HashMap::new();
        let cfg = Config::from_lookup(lookup_from(&map));

        // macOS defaults to 8 (CoreML FastPrediction OOM guard).
        // All other platforms default to 256.
        let expected: usize = if cfg!(target_os = "macos") { 8 } else { 256 };
        assert_eq!(cfg.onnx_batch_size, expected);
    }

    #[test]
    fn onnx_batch_size_custom_value() {
        let map = HashMap::from([("BGE_M3_ONNX_BATCH_SIZE", "16")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.onnx_batch_size, 16);
    }

    #[test]
    fn onnx_batch_size_clamps_to_minimum_1() {
        let map = HashMap::from([("BGE_M3_ONNX_BATCH_SIZE", "0")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.onnx_batch_size, 1);
    }

    #[test]
    fn model_variant_defaults_to_fp16() {
        let map = HashMap::new();
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.model_variant, ModelVariant::Fp16);
    }

    #[test]
    fn model_variant_fp32_when_set() {
        let map = HashMap::from([("BGE_M3_MODEL", "fp32")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.model_variant, ModelVariant::Fp32);
    }

    #[test]
    fn model_variant_int8_when_set() {
        let map = HashMap::from([("BGE_M3_MODEL", "int8")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.model_variant, ModelVariant::Int8);
    }

    #[test]
    fn model_variant_unknown_value_falls_back_to_fp16() {
        let map = HashMap::from([("BGE_M3_MODEL", "invalid")]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.model_variant, ModelVariant::Fp16);
    }
}
