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

//! Startup cache inspection and EFS write-probe.

use std::path::Path;

use super::paths::{TrtCacheInfo, engine_cache_path};

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
pub(super) fn run_write_probe(dir: &Path) {
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
pub(super) fn count_cache_entries(dir: &Path) -> (usize, usize) {
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
