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
/// In addition to the count snapshot, this function performs a one-shot
/// write/read/delete probe of a sentinel file (`.write_probe`) under the
/// cache directory so operators can definitively rule in/out filesystem
/// permission and EFS-AP issues at next boot. The probe emits one of:
///
/// * `INFO "trt cache: write probe succeeded"` (greppable; carries
///   `cache_path`, `bytes_written`, `bytes_read_back`)
/// * `ERROR "trt cache: write probe failed"` (greppable; carries
///   `cache_path`, `phase`, `error`) — `phase` is `create` / `write` /
///   `read` / `mismatch` / `unlink` so operators can see exactly which
///   syscall in the round-trip blocked.
///
/// The probe is best-effort: a failed probe does not abort startup — TRT
/// will surface the same problem (loudly, now that `error_on_failure` is
/// set on the EP dispatch) on the first engine compile if writes are
/// genuinely blocked.
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

    run_write_probe(&path);

    let (engine_count, profile_count) = count_cache_entries(&path);
    TrtCacheInfo {
        path,
        engine_count,
        profile_count,
    }
}

/// One-shot write/read/delete sentinel probe under `dir`.
///
/// The probe rules in/out the "EFS access point POSIX uid mapping blocks
/// regular `creat(2) + write(2) + unlink(2)`" hypothesis at next boot.
/// It writes a 9-byte sentinel `b"trt-probe"` to `<dir>/.write_probe`,
/// reads it back, verifies the round-trip, and deletes the file. Each
/// distinct failure mode (`create`, `write`, `read`, `mismatch`, `unlink`)
/// fires a tagged `ERROR` so operators can disambiguate without
/// instrumenting the filesystem from outside.
///
/// The probe path is fixed (`.write_probe`) so it is greppable and never
/// collides with TRT's own filenames (which all begin with
/// `TensorrtExecutionProvider_TRTKernel_`). Hidden by leading dot so
/// `count_cache_entries` and `count_engine_files` ignore it without any
/// extra filtering.
fn run_write_probe(dir: &Path) {
    use std::io::{Read, Write};

    const PROBE_NAME: &str = ".write_probe";
    const PROBE_DATA: &[u8] = b"trt-probe";

    let probe_path = dir.join(PROBE_NAME);

    // Best-effort cleanup of any stale probe file from a previous boot —
    // this is not the failure mode we are testing, so silently ignore the
    // `NotFound` case and let the create call below surface real problems.
    let _ = std::fs::remove_file(&probe_path);

    let create = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe_path);
    let mut file = match create {
        Ok(f) => f,
        Err(e) => {
            tracing::error!(
                cache_path = %dir.display(),
                phase = "create",
                error = %e,
                "trt cache: write probe failed"
            );
            return;
        }
    };

    if let Err(e) = file.write_all(PROBE_DATA) {
        tracing::error!(
            cache_path = %dir.display(),
            phase = "write",
            error = %e,
            "trt cache: write probe failed"
        );
        let _ = std::fs::remove_file(&probe_path);
        return;
    }
    drop(file);

    let mut buf = Vec::with_capacity(PROBE_DATA.len());
    let mut reader = match std::fs::File::open(&probe_path) {
        Ok(f) => f,
        Err(e) => {
            tracing::error!(
                cache_path = %dir.display(),
                phase = "read",
                error = %e,
                "trt cache: write probe failed"
            );
            let _ = std::fs::remove_file(&probe_path);
            return;
        }
    };
    if let Err(e) = reader.read_to_end(&mut buf) {
        tracing::error!(
            cache_path = %dir.display(),
            phase = "read",
            error = %e,
            "trt cache: write probe failed"
        );
        let _ = std::fs::remove_file(&probe_path);
        return;
    }
    drop(reader);

    if buf != PROBE_DATA {
        tracing::error!(
            cache_path = %dir.display(),
            phase = "mismatch",
            bytes_written = PROBE_DATA.len(),
            bytes_read_back = buf.len(),
            "trt cache: write probe failed"
        );
        let _ = std::fs::remove_file(&probe_path);
        return;
    }

    if let Err(e) = std::fs::remove_file(&probe_path) {
        // Read+write succeeded but unlink didn't — still emit the success
        // INFO so the success-path counter is accurate, then a separate
        // ERROR documenting the unlink failure (the directory will
        // accumulate `.write_probe` files across restarts otherwise).
        tracing::info!(
            cache_path = %dir.display(),
            bytes_written = PROBE_DATA.len(),
            bytes_read_back = buf.len(),
            "trt cache: write probe succeeded"
        );
        tracing::error!(
            cache_path = %dir.display(),
            phase = "unlink",
            error = %e,
            "trt cache: write probe failed"
        );
        return;
    }

    tracing::info!(
        cache_path = %dir.display(),
        bytes_written = PROBE_DATA.len(),
        bytes_read_back = buf.len(),
        "trt cache: write probe succeeded"
    );
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

/// Returns `true` when `name` is a TRT engine plan basename whose `_smXX`
/// suffix matches the requested SM exactly.
///
/// ORT names every TRT engine plan with a `_smXX.engine` suffix tied to the
/// GPU compute capability that built it (`sm75` = T4, `sm86` = A10G, `sm89` =
/// L40S/L4, `sm120` = Blackwell). The match must be **strict** — `sm12` must
/// not match `sm120.engine`, or a B200 worker would happily believe a Hopper
/// plan is usable. We accomplish this by anchoring on the leading underscore:
/// the suffix tested is `"_{sm}.engine"`, so `_sm12.engine` and `_sm120.engine`
/// occupy disjoint string sets.
///
/// Pure, no I/O — easy to unit-test.
#[must_use]
pub(crate) fn matches_sm_suffix(name: &str, sm: &str) -> bool {
    let suffix = format!("_{sm}.engine");
    name.ends_with(&suffix)
}

/// Returns full paths of `.engine` files under `engine_dir` that match the
/// requested SM, or all `.engine` files when `sm` is `None`.
///
/// Single source of truth for engine enumeration: every other function in
/// this module that needs to enumerate engine plans (count, basenames,
/// operator log line) delegates here. Centralising the filter is the design
/// constraint behind the SM-aware refactor — a future caller that grew its
/// own `read_dir` loop would silently regress the heterogeneous-SM safety
/// invariant.
///
/// Failure modes (directory missing or unreadable) collapse to an empty
/// `Vec`, mirroring the legacy `count_engine_files` behaviour and the
/// "operator-visible, not load-bearing" stance of this module.
pub(super) fn engine_files_for_sm(engine_dir: &Path, sm: Option<&str>) -> Vec<PathBuf> {
    let Ok(read_dir) = std::fs::read_dir(engine_dir) else {
        return Vec::new();
    };
    read_dir
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".engine") {
                return None;
            }
            match sm {
                Some(target) if !matches_sm_suffix(&name, target) => None,
                _ => Some(e.path()),
            }
        })
        .collect()
}

/// Counts `.engine` files in `dir` that match the requested SM.
///
/// Pass `Some("sm120")` to count only Blackwell plans; pass `None` to count
/// every `.engine` file regardless of suffix (legacy behaviour, also what
/// the wrapper [`count_engine_files`] does). Returns `0` when the directory
/// does not exist or cannot be read.
///
/// Crate-visible (not `pub(super)`) because the warmup-only postcondition
/// in `lib.rs` calls it directly to apply the same SM filter as the
/// per-worker prewarm path.
pub(crate) fn count_engine_files_for_sm(dir: &Path, sm: Option<&str>) -> usize {
    engine_files_for_sm(dir, sm).len()
}

/// Counts every `.engine` file in `dir` regardless of `_smXX` suffix.
///
/// Backwards-compatible wrapper around [`count_engine_files_for_sm`] with
/// `sm = None`. Retained because the operator-visible startup cache log
/// (`trt cache: found cached engines` in [`log_cache_state`]) reports the
/// total disk footprint, not the SM-filtered subset. Callers that drive
/// the prewarm postcondition use the SM-aware variant directly.
pub(super) fn count_engine_files(dir: &Path) -> usize {
    count_engine_files_for_sm(dir, None)
}

/// Returns sorted basenames of `.engine` files under `engine_dir` that match
/// the requested SM, or all `.engine` basenames when `sm` is `None`.
///
/// Used for operator-visible logs before TRT prewarm; filenames are **not**
/// a reliable `(batch, seq)` key for dynamic models (see `CLAUDE.md`).
pub(crate) fn engine_basenames_for_sm(
    engine_dir: &Path,
    sm: Option<&str>,
) -> std::io::Result<Vec<String>> {
    // We re-implement the read instead of delegating to `engine_files_for_sm`
    // because the latter swallows I/O errors (returns empty on missing dir),
    // whereas this function needs to propagate them so the operator log can
    // surface "could not read engine cache directory for basename listing".
    let read_dir = std::fs::read_dir(engine_dir)?;
    let mut basenames: Vec<String> = read_dir
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".engine") {
                return None;
            }
            match sm {
                Some(target) if !matches_sm_suffix(&name, target) => None,
                _ => Some(name),
            }
        })
        .collect();
    basenames.sort();
    Ok(basenames)
}

/// Returns sorted basenames of every `.engine` file under `engine_dir`,
/// regardless of `_smXX` suffix.
///
/// Backwards-compatible wrapper around [`engine_basenames_for_sm`] with
/// `sm = None`. Retained for the unit tests below; production prewarm log
/// emission goes through [`log_engine_basenames_before_prewarm_for_sm`].
pub(crate) fn engine_basenames_in_dir_sorted(engine_dir: &Path) -> std::io::Result<Vec<String>> {
    engine_basenames_for_sm(engine_dir, None)
}

/// Lists basenames of `.engine` files immediately before TRT prewarm.
///
/// When `sm` is `Some("smXX")` only basenames whose `_smXX.engine` suffix
/// matches are listed (and counted); when `None`, every `.engine` file is
/// listed. Emits a sibling `engine_basename_total_count` field so operators
/// reading `CloudWatch` can see the heterogeneous-SM picture at a glance —
/// e.g. `matching=0, total=3` is the exact failure mode behind the
/// 2026-05-16 codekeeper outage on Blackwell with stale L40S plans.
///
/// ONNX Runtime's `TensorRT` EP names engine caches from the fused subgraph
/// id and precision (`TensorrtExecutionProvider_TRTKernel_…_fp16_smXX.engine`),
/// not from literal `(batch, seq)` dimensions — dynamic shapes are carried
/// in the companion `.profile`. Operators can grep `CloudWatch` for
/// `trt prewarm: cache engine basenames` to correlate disk state with
/// compile times; this is not used for skip logic (see `CLAUDE.md`).
pub(crate) fn log_engine_basenames_before_prewarm_for_sm(engine_dir: &Path, sm: Option<&str>) {
    const MAX_LIST: usize = 64;
    let basenames = match engine_basenames_for_sm(engine_dir, sm) {
        Ok(v) => v,
        Err(e) => {
            tracing::info!(
                cache_path = %engine_dir.display(),
                detected_sm = sm.unwrap_or("unfiltered"),
                error = %e,
                "trt prewarm: could not read engine cache directory for basename listing"
            );
            return;
        }
    };

    let matching_total = basenames.len();
    let mut listed = basenames;
    let truncated = matching_total > MAX_LIST;
    if truncated {
        listed.truncate(MAX_LIST);
    }
    // Also surface the unfiltered total so heterogeneous-SM situations are
    // obvious at a glance. When `sm` is `None` this equals `matching_total`.
    let unfiltered_total = if sm.is_some() {
        engine_basenames_for_sm(engine_dir, None).map_or(matching_total, |v| v.len())
    } else {
        matching_total
    };

    if matching_total == 0 {
        tracing::info!(
            cache_path = %engine_dir.display(),
            detected_sm = sm.unwrap_or("unfiltered"),
            engine_basename_count = 0,
            engine_basename_total_count = unfiltered_total,
            "trt prewarm: cache engine basenames (none for this SM)"
        );
    } else {
        tracing::info!(
            cache_path = %engine_dir.display(),
            detected_sm = sm.unwrap_or("unfiltered"),
            engine_basename_count = matching_total,
            engine_basename_total_count = unfiltered_total,
            truncated,
            engine_basenames = ?listed,
            "trt prewarm: cache engine basenames (for operator correlation; not shape-specific)"
        );
    }
}

/// Backwards-compatible wrapper for the unfiltered prewarm basename log.
///
/// Delegates to [`log_engine_basenames_before_prewarm_for_sm`] with
/// `sm = None` — equivalent to today's behaviour for callers that have not
/// yet been updated to pass the worker's detected SM.
pub(crate) fn log_engine_basenames_before_prewarm(engine_dir: &Path) {
    log_engine_basenames_before_prewarm_for_sm(engine_dir, None);
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

    /// Models the production failure shape from the 2026-05 codekeeper
    /// outage: the engine cache directory exists (created via
    /// `ensure_and_inspect` on every container start) but the prewarm
    /// loop emitted "compile success" without TRT actually writing any
    /// `.engine` files. `ensure_and_inspect` must return
    /// `engine_count = 0` on this state so the warmup-only postcondition
    /// in `lib.rs` can fail the deployment loudly.
    #[test]
    fn ensure_and_inspect_surfaces_empty_cache_after_supposed_writes() {
        let tmp = TempDir::new().expect("tempdir");
        let cache_root = tmp.path();

        // Pretend the prewarm loop ran: directory exists, but the
        // expected `.engine` files were never persisted.
        let info = ensure_and_inspect(cache_root);
        assert!(info.path.is_dir(), "cache dir must exist");
        assert_eq!(
            info.engine_count, 0,
            "an empty cache directory after prewarm must report \
             engine_count=0 — this is the postcondition signal"
        );

        // Add a non-`.engine` file to confirm we don't mis-count
        // sidecar artifacts as engines.
        fs::write(info.path.join("notes.txt"), b"not an engine").unwrap();
        let reinspected = ensure_and_inspect(cache_root);
        assert_eq!(reinspected.engine_count, 0);
    }

    /// After a real successful prewarm sweep, `ensure_and_inspect` must
    /// see the persisted `.engine` files. This is the positive control
    /// for the test above: same directory shape, same caller, but with
    /// the artifacts actually present.
    #[test]
    fn ensure_and_inspect_counts_engines_after_real_writes() {
        let tmp = TempDir::new().expect("tempdir");
        let cache_root = tmp.path();
        let initial = ensure_and_inspect(cache_root);
        assert_eq!(initial.engine_count, 0);

        // Simulate the TRT EP writing a plan file during prewarm.
        let engine_dir = initial.path.clone();
        fs::write(
            engine_dir.join("TensorrtExecutionProvider_TRTKernel_real_sm89.engine"),
            b"plan",
        )
        .unwrap();
        fs::write(
            engine_dir.join("TensorrtExecutionProvider_TRTKernel_real_sm89.profile"),
            b"profile",
        )
        .unwrap();

        let post = ensure_and_inspect(cache_root);
        assert_eq!(post.engine_count, 1);
        assert_eq!(post.profile_count, 1);
    }

    /// On a writable directory, the probe runs to completion and leaves
    /// no `.write_probe` artifact behind for `count_cache_entries` to
    /// stumble over.
    #[test]
    fn write_probe_leaves_no_artifact_on_success() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        run_write_probe(&dir);

        // Probe file must be gone (otherwise it would accumulate across
        // restarts and pollute the engine_count signal).
        assert!(
            !dir.join(".write_probe").exists(),
            "write probe must clean up its sentinel on success"
        );
    }

    /// `ensure_and_inspect` must NOT count the transient `.write_probe`
    /// sentinel as either an engine or a profile, even if the probe is
    /// running in parallel with other work. This is enforced by the
    /// dot-prefixed name and the `.engine` / `.profile` suffix filter in
    /// `count_cache_entries`. We assert it directly here so a future
    /// refactor that changes the suffix filter would catch it.
    #[test]
    fn write_probe_artifact_is_never_counted_as_engine() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        // Simulate a stuck probe file (e.g. from a prior crash).
        fs::write(dir.join(".write_probe"), b"trt-probe").unwrap();
        // Plus a real engine file to anchor the assertion.
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_a_b_sm89.engine"),
            b"plan",
        )
        .unwrap();

        let (engines, profiles) = count_cache_entries(&dir);
        assert_eq!(
            engines, 1,
            "stale .write_probe must not be miscounted as an engine"
        );
        assert_eq!(
            profiles, 0,
            ".write_probe must not be miscounted as a profile either"
        );
    }

    /// Stale probe files left over from a prior crashed boot must be
    /// cleaned up by the next probe run, not multiplied. Documents the
    /// idempotence contract of `run_write_probe`.
    #[test]
    fn write_probe_overwrites_stale_artifact_idempotently() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        // Simulate a stale probe file from a previous crashed boot.
        fs::write(dir.join(".write_probe"), b"stale-data").unwrap();

        run_write_probe(&dir);

        assert!(
            !dir.join(".write_probe").exists(),
            "stale probe must be cleaned up after a successful run"
        );
    }

    // ─── matches_sm_suffix (strict suffix match) ──────────────────────────

    /// Exact-match positives for every SM in the supported fleet.
    #[test]
    fn matches_sm_suffix_accepts_exact_match() {
        assert!(matches_sm_suffix(
            "TensorrtExecutionProvider_TRTKernel_graph_a_fp16_sm89.engine",
            "sm89"
        ));
        assert!(matches_sm_suffix("foo_sm75.engine", "sm75"));
        assert!(matches_sm_suffix("foo_sm86.engine", "sm86"));
        assert!(matches_sm_suffix("foo_sm120.engine", "sm120"));
    }

    /// CRITICAL regression guard: requesting `sm12` MUST NOT match a
    /// `_sm120.engine` plan. The 2026-05-16 outage hinged on accidentally
    /// counting Blackwell plans as "the cache is warm for sm12" or vice
    /// versa. Anchoring on `_sm{XX}.engine` is the only safe match shape.
    #[test]
    fn matches_sm_suffix_rejects_prefix_collision() {
        assert!(!matches_sm_suffix("foo_sm120.engine", "sm12"));
        assert!(!matches_sm_suffix("foo_sm89.engine", "sm8"));
        assert!(!matches_sm_suffix("foo_sm890.engine", "sm89"));
        assert!(!matches_sm_suffix("foo_sm121.engine", "sm12"));
    }

    /// Different SMs in the cache must not cross-match.
    #[test]
    fn matches_sm_suffix_rejects_different_sm() {
        assert!(!matches_sm_suffix("foo_sm89.engine", "sm120"));
        assert!(!matches_sm_suffix("foo_sm120.engine", "sm89"));
        assert!(!matches_sm_suffix("foo_sm75.engine", "sm86"));
    }

    /// `.profile` sidecars share the prefix but the function must not
    /// accept them — only `.engine` files are valid TRT plans.
    #[test]
    fn matches_sm_suffix_rejects_non_engine_extension() {
        assert!(!matches_sm_suffix("foo_sm89.profile", "sm89"));
        assert!(!matches_sm_suffix("foo_sm89.txt", "sm89"));
    }

    /// Files without the leading underscore (unlikely but defensive)
    /// must not match — the underscore is what disambiguates `_sm12` from
    /// `_sm120`.
    #[test]
    fn matches_sm_suffix_requires_leading_underscore() {
        assert!(!matches_sm_suffix("sm89.engine", "sm89"));
        assert!(!matches_sm_suffix("xsm89.engine", "sm89"));
    }

    // ─── engine_files_for_sm (heterogeneous-cache filtering) ──────────────

    /// **The exact production scenario** from the 2026-05-16 codekeeper outage.
    /// Cache contains plans for three SMs (`sm86`, `sm89`, `sm120`); a worker
    /// on a Blackwell GPU (`sm120`) asks for its own SM and must see exactly
    /// one entry — not three. The pre-fix bookkeeping counted all three and
    /// reported `cache_hit:true, engine_count_before:3` even though only the
    /// `sm120` plan was usable.
    #[test]
    fn engine_files_for_sm_isolates_blackwell_from_heterogeneous_cache() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        // The exact production filename shape per CloudWatch evidence.
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_subgraph_fp16_sm86.engine"),
            b"a10g",
        )
        .unwrap();
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_subgraph_fp16_sm89.engine"),
            b"l40s",
        )
        .unwrap();
        fs::write(
            dir.join("TensorrtExecutionProvider_TRTKernel_subgraph_fp16_sm120.engine"),
            b"blackwell",
        )
        .unwrap();

        let sm120 = engine_files_for_sm(&dir, Some("sm120"));
        assert_eq!(sm120.len(), 1, "sm120 worker must see exactly 1 plan");
        let sm89 = engine_files_for_sm(&dir, Some("sm89"));
        assert_eq!(sm89.len(), 1, "sm89 worker must see exactly 1 plan");
        let unfiltered = engine_files_for_sm(&dir, None);
        assert_eq!(unfiltered.len(), 3, "None passthrough must see all 3");

        // Sibling SM check: an SM that is not present on disk must return
        // empty — this is the fresh-deploy / stale-cache failure mode the
        // postcondition is meant to surface as `cache_hit:false`.
        let sm75 = engine_files_for_sm(&dir, Some("sm75"));
        assert!(sm75.is_empty(), "sm75 (no plan) must return empty");
    }

    /// `None` SM acts as a passthrough (today's behaviour), counting every
    /// `.engine` file regardless of suffix. Pinned so a future refactor
    /// cannot accidentally introduce a filter-on-None regression.
    #[test]
    fn engine_files_for_sm_none_passthrough_matches_legacy() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a_sm89.engine"), b"x").unwrap();
        fs::write(dir.join("b_sm120.engine"), b"y").unwrap();
        fs::write(dir.join("c.profile"), b"z").unwrap();
        fs::write(dir.join("d.txt"), b"w").unwrap();

        let all = engine_files_for_sm(&dir, None);
        assert_eq!(
            all.len(),
            2,
            "None must return every .engine, ignoring .profile/.txt"
        );
        assert_eq!(count_engine_files(&dir), 2);
        assert_eq!(count_engine_files_for_sm(&dir, None), 2);
    }

    /// Missing cache directory collapses to an empty Vec — the same shape
    /// as the legacy `count_engine_files` fallback. Required so the prewarm
    /// path can call this on a cold-boot worker without error handling.
    #[test]
    fn engine_files_for_sm_missing_directory_returns_empty() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("never-created");
        assert!(engine_files_for_sm(&missing, Some("sm120")).is_empty());
        assert!(engine_files_for_sm(&missing, None).is_empty());
        assert_eq!(count_engine_files_for_sm(&missing, Some("sm120")), 0);
    }

    /// Empty cache directory returns empty for every SM.
    #[test]
    fn engine_files_for_sm_empty_directory_returns_empty() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        assert!(engine_files_for_sm(&dir, Some("sm120")).is_empty());
        assert!(engine_files_for_sm(&dir, None).is_empty());
    }

    /// `.profile` sibling files (TRT input-shape profiles) MUST be ignored
    /// by every engine enumerator. We retain them on disk for diagnostic
    /// purposes (counted under `profile_count` in [`TrtCacheInfo`]) but
    /// they are never engine plans and must not be returned here.
    #[test]
    fn engine_files_for_sm_ignores_profile_sidecars() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("foo_sm120.engine"), b"plan").unwrap();
        fs::write(dir.join("foo_sm120.profile"), b"shapes").unwrap();

        let sm120 = engine_files_for_sm(&dir, Some("sm120"));
        assert_eq!(sm120.len(), 1);
        assert!(sm120[0].to_string_lossy().ends_with(".engine"));
    }

    /// Subdirectories and non-file entries must be skipped. Defensive
    /// against future cache layouts that add per-shape subdirs.
    #[test]
    fn engine_files_for_sm_skips_non_files() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("subdir_sm120.engine")).unwrap();
        fs::write(dir.join("real_sm120.engine"), b"plan").unwrap();

        let files = engine_files_for_sm(&dir, Some("sm120"));
        assert_eq!(files.len(), 1, "subdir named *.engine must not be counted");
    }

    // ─── engine_basenames_for_sm (sorted, SM-aware) ───────────────────────

    /// SM filter returns only matching basenames, sorted. The unfiltered
    /// case returns every `.engine` basename, sorted (backward compat with
    /// [`engine_basenames_in_dir_sorted`]).
    #[test]
    fn engine_basenames_for_sm_sorted_and_filtered() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("z_sm120.engine"), b"").unwrap();
        fs::write(dir.join("a_sm120.engine"), b"").unwrap();
        fs::write(dir.join("m_sm89.engine"), b"").unwrap();

        let sm120 = engine_basenames_for_sm(&dir, Some("sm120")).unwrap();
        assert_eq!(
            sm120,
            vec!["a_sm120.engine".to_string(), "z_sm120.engine".to_string()]
        );
        let unfiltered = engine_basenames_for_sm(&dir, None).unwrap();
        assert_eq!(
            unfiltered,
            vec![
                "a_sm120.engine".to_string(),
                "m_sm89.engine".to_string(),
                "z_sm120.engine".to_string(),
            ]
        );
    }

    /// Missing directory propagates as `Err` (so the operator log can
    /// surface the I/O error path). Pinned because `engine_files_for_sm`
    /// swallows it — the two have different error-handling contracts.
    #[test]
    fn engine_basenames_for_sm_missing_directory_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("nope");
        assert!(engine_basenames_for_sm(&missing, Some("sm120")).is_err());
        assert!(engine_basenames_for_sm(&missing, None).is_err());
    }

    /// On Linux only: a directory with `0o555` (read+execute, no write)
    /// must produce a probe failure rather than panicking. This is the
    /// closest standalone unit-test analogue of the EFS-AP-blocks-write
    /// scenario described in the production gotcha. The probe should
    /// log an ERROR and return without aborting startup.
    #[cfg(target_os = "linux")]
    #[test]
    fn write_probe_failure_does_not_panic_on_readonly_dir() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().join("trt-engines");
        fs::create_dir_all(&dir).unwrap();

        // Drop write permission before running the probe.
        let mut perms = fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&dir, perms).unwrap();

        // Probe must return cleanly (logs an ERROR with phase=create).
        run_write_probe(&dir);

        // Restore write permission so TempDir's cleanup can proceed.
        let mut perms = fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dir, perms).unwrap();
    }
}
