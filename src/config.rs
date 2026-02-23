use std::env;

pub struct Config {
    pub cache_dir: String,
    pub bind_addr: String,
    pub workers: usize,
    pub max_batch: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let workers = env::var("BGE_M3_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);

        let max_batch = env::var("BGE_M3_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            .max(1);

        Self {
            cache_dir: env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/cache".to_string()),
            bind_addr: env::var("BGE_M3_BIND").unwrap_or_else(|_| "0.0.0.0:8081".to_string()),
            workers,
            max_batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_without_env_vars() {
        // Remove all relevant env vars to test defaults
        env::remove_var("BGE_M3_CACHE_DIR");
        env::remove_var("BGE_M3_BIND");
        env::remove_var("BGE_M3_WORKERS");
        env::remove_var("BGE_M3_MAX_BATCH");

        let cfg = Config::from_env();

        assert_eq!(cfg.cache_dir, "/cache");
        assert_eq!(cfg.bind_addr, "0.0.0.0:8081");
        assert_eq!(cfg.workers, 2);
        assert_eq!(cfg.max_batch, 256);
    }

    #[test]
    fn workers_clamps_to_minimum_1() {
        env::remove_var("BGE_M3_CACHE_DIR");
        env::remove_var("BGE_M3_BIND");
        env::remove_var("BGE_M3_MAX_BATCH");
        env::set_var("BGE_M3_WORKERS", "0");

        let cfg = Config::from_env();

        assert_eq!(cfg.workers, 1);

        env::remove_var("BGE_M3_WORKERS");
    }
}
