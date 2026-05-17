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

//! Filesystem-backed tests for the `count_engine_files` snapshot contract
//! that both the per-shape delta WARN and the per-worker prewarm
//! postcondition rely on.

use super::super::{
    prewarm_persistence_postcondition_failed, prewarm_persistence_suspicious_undercount,
};

// ─── engine count snapshot wiring (filesystem-backed) ─────────────────

/// `count_engine_files` is the mechanism behind both the per-shape
/// delta WARN and the per-worker prewarm postcondition. Verify it
/// reflects engine-file writes the way the prewarm path does:
/// before-compile snapshot + on-disk write + after-compile snapshot
/// must produce a delta of `+1`. This pins down the contract that the
/// silent-persistence detector relies on.
#[test]
fn count_engine_files_reflects_post_write_delta() {
    use super::super::trt_cache::count_engine_files;
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();

    let before = count_engine_files(&dir);
    assert_eq!(before, 0, "fresh tempdir should have no engines");

    fs::write(
        dir.join("TensorrtExecutionProvider_TRTKernel_graph_a_111_fp16_sm89.engine"),
        b"plan-bytes",
    )
    .unwrap();

    let after = count_engine_files(&dir);
    assert_eq!(after, 1, "after a single engine write, count must be 1");
    let delta = i64::try_from(after).unwrap() - i64::try_from(before).unwrap();
    assert_eq!(delta, 1, "delta must be +1");
}

/// When `session.run()` returns `Ok(_)` but TRT EP did NOT write an engine
/// file (the production defect signal), `count_engine_files` returns 0.
/// The postcondition helper must flag it.
///
/// Note: the postcondition now takes `engine_count_after` directly rather
/// than a computed delta — `after == 0` is the actionable condition.
#[test]
fn count_engine_files_zero_after_triggers_postcondition() {
    use super::super::trt_cache::count_engine_files;
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    std::fs::create_dir_all(&dir).unwrap();

    // Fake "compile success without persistence": the directory is
    // never written to, even though the prewarm aggregator believes
    // a fresh compile happened.
    let after = count_engine_files(&dir);
    assert_eq!(after, 0);

    assert!(
        prewarm_persistence_postcondition_failed(1, after),
        "fresh_compiles=1 with after=0 must trigger the postcondition"
    );
}

/// Profile-update case: the engine file already existed before this shape
/// was compiled (from a previous shape's cold compile), and TRT EP rewrote
/// it in-place.  `after == before == 1`, delta == 0.
///
/// The postcondition must NOT fire — the file is still there.
#[test]
fn count_engine_files_profile_update_passes_postcondition() {
    use super::super::trt_cache::count_engine_files;
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();

    // Write an engine file to simulate the state after the first compile.
    fs::write(
        dir.join("TensorrtExecutionProvider_TRTKernel_a_sm89.engine"),
        b"plan-v1",
    )
    .unwrap();

    // Simulate TRT EP "rewriting in-place": overwrite with updated profile.
    fs::write(
        dir.join("TensorrtExecutionProvider_TRTKernel_a_sm89.engine"),
        b"plan-v2-extended-profile",
    )
    .unwrap();

    let after = count_engine_files(&dir);
    assert_eq!(after, 1, "in-place rewrite must not change the file count");

    // 15 more fresh compiles happened (shapes 2-16 of a 16-shape shard),
    // each reusing/rewriting the same file.  after==1 must pass.
    assert!(
        !prewarm_persistence_postcondition_failed(15, after),
        "fresh_compiles=15 with after=1 must pass (profile-update case)"
    );
    assert!(
        !prewarm_persistence_suspicious_undercount(15, after),
        "WARN must be silent for profile-update case"
    );
}

/// Ensure the postcondition is satisfied when the directory is
/// populated between `_before` and `_after` snapshots.
#[test]
fn count_engine_files_positive_after_passes_postcondition() {
    use super::super::trt_cache::count_engine_files;
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();

    fs::write(
        dir.join("TensorrtExecutionProvider_TRTKernel_a_sm89.engine"),
        b"plan",
    )
    .unwrap();
    let after = count_engine_files(&dir);

    assert!(
        !prewarm_persistence_postcondition_failed(1, after),
        "a single fresh compile that wrote one engine must pass"
    );
}
