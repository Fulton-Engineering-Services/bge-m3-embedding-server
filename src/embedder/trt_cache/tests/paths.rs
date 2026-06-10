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

use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

use super::super::inspect::{count_cache_entries, ensure_and_inspect, run_write_probe};
use super::super::paths::{TRT_ENGINE_SUBDIR, TrtCacheInfo, engine_cache_path, timing_cache_path};

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
/// Models the production failure shape where the engine cache directory
/// exists (created via
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
