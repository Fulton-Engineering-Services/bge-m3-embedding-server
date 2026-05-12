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

//! `TensorRT` engine cache path construction, inspection, and durability.
//!
//! Why a dedicated module? Investigation of the 2026-05 codekeeper outage
//! showed two consecutive cold starts producing identical 172 s recompile
//! times for `1×8192` — the `{cache_dir}/trt-engines/` directory on EFS was
//! NOT being reused between container restarts. The most plausible root cause
//! is that ECS SIGKILL on OOM (`exitCode 137`) interrupts the container before
//! the kernel's writeback timer flushes buffered writes to EFS. The EFS inode
//! lists the engine files, but their data blocks are zero-length or partial,
//! so ORT/TRT silently treats them as cache misses and rebuilds from scratch.
//!
//! This module:
//! 1. Constructs the cache directory path (stable, no per-container ephemera).
//! 2. Inspects the directory at startup so the operator-visible INFO log shows
//!    "found N cached engines" or "empty (will compile)" — without this we
//!    cannot tell from `CloudWatch` whether the EFS mount is actually persisting.
//! 3. Exposes an explicit `fsync_cache_dir` that flushes both file data and
//!    directory metadata to disk, called after each successful TRT engine
//!    compile so an OOM-kill never strands a partially-written engine plan.
//!
//! TRT plan files embed `(GPU compute capability, CUDA version, TRT version,
//! ONNX model SHA, builder config)`. Within a homogeneous ASG (same instance
//! family, same AMI) these are stable, so the cache is reusable per-EC2-host.
//! ASGs that mix instance families (T4 → A10G) will see expected cache misses
//! when a task lands on a different GPU architecture.
//!
//! Several items below are referenced only from `session.rs` under
//! `#[cfg(all(not(target_os = "macos"), feature = "tensorrt"))]` — the CPU /
//! macOS build legitimately never calls them. The `#[allow(dead_code)]`
//! attribute below silences the resulting unused-warning under those builds;
//! the unit tests in this file keep the items exercised on every CI target.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Subdirectory under `BGE_M3_CACHE_DIR` where ORT/TRT writes engine plan files.
///
/// Documented in `CLAUDE.md`. Kept here as a constant so callers cannot drift.
pub(crate) const TRT_ENGINE_SUBDIR: &str = "trt-engines";

/// Subdirectory under `BGE_M3_CACHE_DIR` where TRT writes the timing cache.
///
/// The timing cache is a separate persistence layer from the engine cache:
/// it stores per-tactic kernel timings so the TRT builder can skip the tactic
/// selection step on rebuild. Sharing it across shapes within a single
/// container produces meaningful speedup even when the engine cache is cold.
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

/// Creates the engine cache directory if missing and returns a snapshot of
/// its current contents so the operator-visible startup log can report
/// whether the persistent cache is being reused or whether the container
/// must compile every shape from scratch.
///
/// Logs a `WARN` (but does not error) if the directory cannot be created —
/// TRT will fail in a more diagnostic way on the next compile attempt, and
/// surfacing the error here would mask CPU-EP startup paths that share this
/// code path. The intent is operator visibility in `CloudWatch`, not a hard
/// gate.
pub(crate) fn ensure_and_inspect(cache_dir: &Path) -> TrtCacheInfo {
    let path = engine_cache_path(cache_dir);
    if let Err(e) = std::fs::create_dir_all(&path) {
        tracing::warn!(
            cache_path = %path.display(),
            error = %e,
            "TensorRT engine cache directory could not be created; \
             engine caching may be unavailable for this container"
        );
        return TrtCacheInfo {
            path,
            engine_count: 0,
            profile_count: 0,
        };
    }

    let (engine_count, profile_count) = count_cache_entries(&path);
    TrtCacheInfo {
        path,
        engine_count,
        profile_count,
    }
}

/// Counts `.engine` and `.profile` files in `dir`. Returns `(0, 0)` if the
/// directory cannot be read (also covers "not yet created" cases).
fn count_cache_entries(dir: &Path) -> (usize, usize) {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return (0, 0);
    };
    let mut engines = 0usize;
    let mut profiles = 0usize;
    for entry in read_dir.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let lossy = name.to_string_lossy();
        if lossy.ends_with(".engine") {
            engines += 1;
        } else if lossy.ends_with(".profile") {
            profiles += 1;
        }
    }
    (engines, profiles)
}

/// Emits a single INFO log line describing the TRT cache state at startup.
///
/// Operators reading `CloudWatch` should be able to tell at a glance whether
/// the cache is being reused. The message wording is stable and grep-friendly:
/// `trt cache: ...`.
pub(crate) fn log_cache_state(info: &TrtCacheInfo) {
    if info.engine_count == 0 {
        tracing::info!(
            cache_path = %info.path.display(),
            engine_count = 0,
            "trt cache: empty (will compile)"
        );
    } else {
        tracing::info!(
            cache_path = %info.path.display(),
            engine_count = info.engine_count,
            profile_count = info.profile_count,
            "trt cache: found cached engines"
        );
    }
}

/// Counts `.engine` files in `dir` without allocating a sorted list.
///
/// Returns `0` when the directory does not exist or cannot be read (covers both
/// cold-cache and permission-error cases). Used by `trt_warmup` to decide
/// whether to run the coverage-check fast path before attempting a full
/// prewarm sweep.
pub(super) fn count_engine_files(dir: &Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |rd| {
        rd.flatten()
            .filter(|e| {
                e.file_type().is_ok_and(|t| t.is_file())
                    && e.file_name().to_string_lossy().ends_with(".engine")
            })
            .count()
    })
}

/// Returns sorted basenames of regular files ending in `.engine` under `engine_dir`.
///
/// Used for operator-visible logs before TRT prewarm; filenames are **not** a
/// reliable `(batch, seq)` key for dynamic models (see `CLAUDE.md`).
pub(crate) fn engine_basenames_in_dir_sorted(engine_dir: &Path) -> std::io::Result<Vec<String>> {
    let read_dir = std::fs::read_dir(engine_dir)?;
    let mut basenames: Vec<String> = read_dir
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.ends_with(".engine").then_some(name)
        })
        .collect();
    basenames.sort();
    Ok(basenames)
}

/// Lists basenames of existing `.engine` files immediately before TRT prewarm.
///
/// ONNX Runtime's `TensorRT` EP names engine caches from the fused subgraph id
/// and precision (`TensorrtExecutionProvider_TRTKernel_…_fp16_smXX.engine`), not
/// from literal `(batch, seq)` dimensions — dynamic shapes are carried in the
/// companion `.profile`. Operators can grep `CloudWatch` for `trt prewarm: cache
/// engine basenames` to correlate disk state with compile times; this is not
/// used for skip logic (see `CLAUDE.md`).
pub(crate) fn log_engine_basenames_before_prewarm(engine_dir: &Path) {
    const MAX_LIST: usize = 64;
    let basenames = match engine_basenames_in_dir_sorted(engine_dir) {
        Ok(v) => v,
        Err(e) => {
            tracing::info!(
                cache_path = %engine_dir.display(),
                error = %e,
                "trt prewarm: could not read engine cache directory for basename listing"
            );
            return;
        }
    };

    let total = basenames.len();
    let mut listed = basenames;
    let truncated = total > MAX_LIST;
    if truncated {
        listed.truncate(MAX_LIST);
    }

    if total == 0 {
        tracing::info!(
            cache_path = %engine_dir.display(),
            engine_basename_count = 0,
            "trt prewarm: cache engine basenames (none on disk yet)"
        );
    } else {
        tracing::info!(
            cache_path = %engine_dir.display(),
            engine_basename_count = total,
            truncated,
            engine_basenames = ?listed,
            "trt prewarm: cache engine basenames (for operator correlation; not shape-specific)"
        );
    }
}

/// Flushes every regular file in `dir` plus the directory's own metadata to
/// disk, so an unexpected SIGKILL (ECS OOM, host failure) cannot strand a
/// partially-written engine plan in the page cache.
///
/// On Linux, `File::sync_all` issues `fsync(2)`. Calling it on a directory
/// handle flushes the directory inode (name → inode mapping); calling it on
/// each file flushes that file's data blocks. Both are required: directory
/// fsync alone is not enough to guarantee file data is durable on EFS or any
/// POSIX-compliant filesystem.
///
/// Errors on individual files are logged at `WARN` but do not abort the
/// sweep — a single broken sidecar file should not prevent the rest of the
/// cache from being durable.
///
/// On non-Linux targets this is a no-op (the TRT EP itself is Linux-only).
#[cfg(target_os = "linux")]
pub(crate) fn fsync_cache_dir(dir: &Path) {
    use std::fs::File;
    let read_dir = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                cache_path = %dir.display(),
                error = %e,
                "trt cache: could not enumerate directory for fsync; \
                 engine plan files may not be durable on the next OOM"
            );
            return;
        }
    };

    let mut synced = 0usize;
    for entry in read_dir.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        match File::open(&path).and_then(|f| f.sync_all()) {
            Ok(()) => synced += 1,
            Err(e) => {
                tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "trt cache: fsync(file) failed; engine plan may not be durable"
                );
            }
        }
    }

    match File::open(dir).and_then(|d| d.sync_all()) {
        Ok(()) => {
            tracing::debug!(
                cache_path = %dir.display(),
                files_synced = synced,
                "trt cache: directory fsynced"
            );
        }
        Err(e) => {
            tracing::warn!(
                cache_path = %dir.display(),
                error = %e,
                "trt cache: fsync(directory) failed; name → inode mapping \
                 may not be durable on the next OOM"
            );
        }
    }
}

/// Non-Linux no-op stub so callers do not need to gate every call site.
#[cfg(not(target_os = "linux"))]
pub(crate) fn fsync_cache_dir(_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn engine_cache_path_is_stable_subdirectory() {
        let root = Path::new("/tmp/example-cache");
        assert_eq!(
            engine_cache_path(root),
            PathBuf::from("/tmp/example-cache/trt-engines")
        );
    }

    #[test]
    fn timing_cache_path_is_stable_subdirectory() {
        let root = Path::new("/tmp/example-cache");
        assert_eq!(
            timing_cache_path(root),
            PathBuf::from("/tmp/example-cache/trt-timing")
        );
    }

    #[test]
    fn engine_cache_path_has_no_per_container_ephemera() {
        // Regression guard: the cache path must not embed PID, hostname,
        // container ID, or any other per-container suffix. Two calls with
        // the same root must yield byte-identical paths.
        let root = Path::new("/cache");
        let a = engine_cache_path(root);
        let b = engine_cache_path(root);
        assert_eq!(a, b);
        let s = a.to_string_lossy();
        assert!(
            !s.contains(&std::process::id().to_string()),
            "cache path should not embed PID, got: {s}"
        );
    }

    #[test]
    fn ensure_and_inspect_creates_missing_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let cache_root = tmp.path();
        let info = ensure_and_inspect(cache_root);
        assert_eq!(info.path, cache_root.join(TRT_ENGINE_SUBDIR));
        assert!(info.path.is_dir(), "trt-engines/ should have been created");
        assert_eq!(info.engine_count, 0);
        assert_eq!(info.profile_count, 0);
    }

    #[test]
    fn ensure_and_inspect_counts_engine_and_profile_files() {
        let tmp = TempDir::new().expect("tempdir");
        let cache_root = tmp.path();
        let engine_dir = cache_root.join(TRT_ENGINE_SUBDIR);
        fs::create_dir_all(&engine_dir).unwrap();
        fs::write(engine_dir.join("foo.engine"), b"x").unwrap();
        fs::write(engine_dir.join("bar.engine"), b"y").unwrap();
        fs::write(engine_dir.join("foo.profile"), b"z").unwrap();
        fs::write(engine_dir.join("notes.txt"), b"ignored").unwrap();

        let info = ensure_and_inspect(cache_root);
        assert_eq!(info.engine_count, 2);
        assert_eq!(info.profile_count, 1);
    }

    #[test]
    fn count_cache_entries_returns_zero_for_missing_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("does-not-exist");
        assert_eq!(count_cache_entries(&missing), (0, 0));
    }

    /// fsync sweep must succeed on a populated directory and leave file
    /// contents intact. We can't assert the kernel actually flushed to disk
    /// without root-level tooling, but exercising the syscall paths catches
    /// permission / handle bugs that would silently degrade durability.
    #[cfg(target_os = "linux")]
    #[test]
    fn fsync_cache_dir_walks_populated_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.engine"), b"plan-a").unwrap();
        fs::write(dir.join("b.engine"), b"plan-b").unwrap();

        // No panic, no error log — and contents preserved afterwards.
        fsync_cache_dir(&dir);

        assert_eq!(fs::read(dir.join("a.engine")).unwrap(), b"plan-a");
        assert_eq!(fs::read(dir.join("b.engine")).unwrap(), b"plan-b");
    }

    /// On Linux, fsync of an empty directory must not panic. The TRT cache
    /// directory is empty between `ensure_and_inspect` and the first
    /// engine compile — the sweep must be safe to call in that window.
    #[cfg(target_os = "linux")]
    #[test]
    fn fsync_cache_dir_handles_empty_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        fsync_cache_dir(&dir);

        // Directory still exists and is still empty.
        let entries: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert_eq!(entries.len(), 0);
    }

    /// On Linux, fsync of a missing directory must not panic (logs a WARN).
    #[cfg(target_os = "linux")]
    #[test]
    fn fsync_cache_dir_missing_directory_does_not_panic() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("never-created");

        fsync_cache_dir(&missing);
    }

    #[test]
    fn count_engine_files_returns_zero_for_missing_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("does-not-exist");
        assert_eq!(count_engine_files(&missing), 0);
    }

    #[test]
    fn count_engine_files_counts_only_engine_extension() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.engine"), b"plan-a").unwrap();
        fs::write(dir.join("b.engine"), b"plan-b").unwrap();
        fs::write(dir.join("c.profile"), b"profile-c").unwrap();
        fs::write(dir.join("d.txt"), b"ignored").unwrap();

        assert_eq!(count_engine_files(&dir), 2);
    }

    #[test]
    fn engine_basenames_in_dir_sorted_empty_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        assert!(super::engine_basenames_in_dir_sorted(&dir)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn engine_basenames_in_dir_sorted_collects_and_sorts() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_graph_z_999_fp16_sm89.engine"),
            b"a",
        )
        .unwrap();
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_graph_a_111_fp16_sm89.engine"),
            b"b",
        )
        .unwrap();
        fs::write(dir.join("notes.txt"), b"x").unwrap();

        let names = super::engine_basenames_in_dir_sorted(&dir).unwrap();
        assert_eq!(
            names,
            vec![
                "TensorrtExecutionProvider_TRTKernel_graph_a_111_fp16_sm89.engine",
                "TensorrtExecutionProvider_TRTKernel_graph_z_999_fp16_sm89.engine",
            ]
        );
    }

    #[test]
    fn engine_basenames_in_dir_sorted_missing_directory_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("nope");
        assert!(super::engine_basenames_in_dir_sorted(&missing).is_err());
    }
}
