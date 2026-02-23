use std::env;

pub struct Config {
    pub cache_dir: String,
    pub bind_addr: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            cache_dir: env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/cache".to_string()),
            bind_addr: env::var("BGE_M3_BIND").unwrap_or_else(|_| "0.0.0.0:8081".to_string()),
        }
    }
}
