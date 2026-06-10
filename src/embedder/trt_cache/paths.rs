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

//! Cache directory paths and startup snapshot type.

use std::path::{Path, PathBuf};

/// Subdirectory under `BGE_M3_CACHE_DIR` where ORT/TRT writes engine plan files.
pub(crate) const TRT_ENGINE_SUBDIR: &str = "trt-engines";

/// Subdirectory under `BGE_M3_CACHE_DIR` where TRT writes the timing cache.
pub(crate) const TRT_TIMING_SUBDIR: &str = "trt-timing";

/// Summary of the TRT engine cache state, produced at startup so the operator
/// can see in `CloudWatch` whether the persistent volume is actually being
/// reused between container restarts.
#[derive(Debug, Clone)]
pub(crate) struct TrtCacheInfo {
    /// Absolute cache directory path (stable; no per-container ephemera).
    pub path: PathBuf,
    /// Number of `.engine` files found in the cache directory before this
    /// container started doing any work. Zero means a cold cache.
    pub engine_count: usize,
    /// Number of `.profile` files (TRT input-shape profiles emitted alongside
    /// each `.engine`). Reported for diagnostic completeness; the count is
    /// expected to be `engine_count` once warmup completes.
    pub profile_count: usize,
}

/// Returns the canonical TRT engine-cache directory for a given root cache.
///
/// Path is stable across container restarts as long as `cache_dir` is mounted
/// at the same location — i.e. for ECS this means a persistent EFS access
/// point or a host-bind mount. There is **no PID, hostname, or container
/// identifier in the path** — that was already true before this change, but
/// is now centralised here so future callers cannot reintroduce per-container
/// ephemera.
pub(crate) fn engine_cache_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(TRT_ENGINE_SUBDIR)
}

/// Returns the canonical TRT timing-cache file path for a given root cache.
///
/// The timing cache is a single file, not a directory of per-shape files.
pub(crate) fn timing_cache_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(TRT_TIMING_SUBDIR)
}
