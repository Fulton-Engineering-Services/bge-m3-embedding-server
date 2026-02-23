use std::env;

pub struct Config {
    pub cache_dir: String,
    /// Bind address. Defaults to `0.0.0.0:8081` — intentional for Docker
    /// container deployments where all interfaces must be reachable (SEC-5).
    pub bind_addr: String,
    pub workers: usize,
    pub max_batch: usize,
}

impl Config {
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

        Self {
            cache_dir: lookup("BGE_M3_CACHE_DIR").unwrap_or_else(|| "/cache".to_string()),
            bind_addr: lookup("BGE_M3_BIND").unwrap_or_else(|| "0.0.0.0:8081".to_string()),
            workers,
            max_batch,
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
        ]);
        let cfg = Config::from_lookup(lookup_from(&map));

        assert_eq!(cfg.cache_dir, "/tmp/models");
        assert_eq!(cfg.bind_addr, "127.0.0.1:9090");
        assert_eq!(cfg.workers, 4);
        assert_eq!(cfg.max_batch, 128);
    }
}
