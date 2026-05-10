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

//! Persistent cache of fitted probe coefficients on the EFS volume.
//!
//! The cache key is `{server_version, model, max_seq, arch}`. When the
//! fingerprint matches the current server's configuration, the probe is
//! skipped and the cached `(a, b)` are used immediately.

use std::path::Path;

use tracing::{info, warn};

#[derive(serde::Serialize, serde::Deserialize)]
struct ProbeCache {
    schema_version: u32,
    server_version: String,
    model: String,
    max_seq: usize,
    arch: String,
    fitted_at_unix: u64,
    a: f64,
    b: f64,
}

/// Attempts to load cached probe coefficients from `{cache_dir}/probe-coefficients.json`.
///
/// Returns `Some((a, b))` when a valid, fingerprint-matching cache file exists.
/// Returns `None` when the file is absent, unreadable, or the fingerprint does
/// not match the current `(server_version, model_variant, max_seq, arch)`.
pub(crate) fn try_load_probe_cache(
    cache_dir: &Path,
    model_variant: &str,
    max_seq: usize,
) -> Option<(f64, f64)> {
    let path = cache_dir.join("probe-coefficients.json");
    let raw = std::fs::read_to_string(&path).ok()?;
    let cache: ProbeCache = serde_json::from_str(&raw).ok()?;

    let current_version = env!("CARGO_PKG_VERSION");
    let current_arch = std::env::consts::ARCH;

    if cache.schema_version != 1
        || cache.server_version != current_version
        || cache.model != model_variant
        || cache.max_seq != max_seq
        || cache.arch != current_arch
    {
        info!(
            cached_version = %cache.server_version,
            current_version,
            cached_model = %cache.model,
            model_variant,
            cached_max_seq = cache.max_seq,
            max_seq,
            cached_arch = %cache.arch,
            current_arch,
            "Probe cache fingerprint mismatch; will re-probe"
        );
        return None;
    }

    if cache.a <= 0.0 || cache.b <= 0.0 {
        warn!("Probe cache has non-positive coefficients; ignoring");
        return None;
    }

    info!(
        a = cache.a,
        b = cache.b,
        fitted_at_unix = cache.fitted_at_unix,
        "Probe cache hit — skipping startup probe"
    );
    Some((cache.a, cache.b))
}

/// Saves fitted probe coefficients to `{cache_dir}/probe-coefficients.json`
/// via an atomic temp-file + rename.
///
/// Errors are logged and silently ignored — a cache write failure must never
/// abort the server.
pub(crate) fn save_probe_cache(
    cache_dir: &Path,
    model_variant: &str,
    max_seq: usize,
    a: f64,
    b: f64,
) {
    let fitted_at_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let cache = ProbeCache {
        schema_version: 1,
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        model: model_variant.to_string(),
        max_seq,
        arch: std::env::consts::ARCH.to_string(),
        fitted_at_unix,
        a,
        b,
    };

    let json = match serde_json::to_string_pretty(&cache) {
        Ok(j) => j,
        Err(e) => {
            warn!(error = %e, "Failed to serialize probe cache; skipping write");
            return;
        }
    };

    let final_path = cache_dir.join("probe-coefficients.json");
    let tmp_path = cache_dir.join("probe-coefficients.json.tmp");

    if let Err(e) = std::fs::write(&tmp_path, &json) {
        warn!(error = %e, path = %tmp_path.display(), "Failed to write probe cache temp file");
        return;
    }

    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        warn!(error = %e, "Failed to atomically rename probe cache file");
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }

    info!(
        path = %final_path.display(),
        a,
        b,
        "Probe coefficients cached to EFS"
    );
}
