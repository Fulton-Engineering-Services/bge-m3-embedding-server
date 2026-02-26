use std::env;
use std::time::Duration;

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
    /// Duration of inactivity after which workers unload their model instances from memory.
    ///
    /// Set with `BGE_M3_IDLE_TIMEOUT_SECS`. Defaults to `300` (5 minutes).
    /// Set to `0` to disable idle unloading entirely.
    ///
    /// When unloaded, models are automatically reloaded on the next incoming request.
    /// The reload blocks the request until complete (~10–30 s from cache).
    pub idle_timeout: Option<Duration>,
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

        let idle_timeout_secs = lookup("BGE_M3_IDLE_TIMEOUT_SECS")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        let idle_timeout = (idle_timeout_secs > 0).then(|| Duration::from_secs(idle_timeout_secs));

        Self {
            cache_dir: lookup("BGE_M3_CACHE_DIR").unwrap_or_else(|| "/cache".to_string()),
            bind_addr: lookup("BGE_M3_BIND").unwrap_or_else(|| "0.0.0.0:8081".to_string()),
            workers,
            max_batch,
            idle_timeout,
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
}
