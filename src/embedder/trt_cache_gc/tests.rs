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

//! Unit tests for the (feature-gated) stale-SM cache GC.
//!
//! Each test constructs a temp directory containing a synthetic mix of
//! engine plans across SM tags plus benign non-engine files, calls
//! `gc_stale_sm_plans`, and asserts both the on-disk residue and the
//! returned counts. The whole `tests` module is gated by the parent
//! `#[cfg(feature = "cache-gc")]` declaration in `embedder.rs`.

use std::fs;
use std::path::Path;

use tempfile::TempDir;

use super::{ENGINE_SIDE_SUFFIXES, GcStats, gc_stale_sm_plans};

/// Convenience helper: write a byte-string file under `dir/name` and return
/// its size for `bytes_freed` cross-checks.
fn write_file(dir: &Path, name: &str, contents: &[u8]) -> u64 {
    fs::write(dir.join(name), contents).expect("write fixture file");
    contents.len() as u64
}

/// Returns the sorted list of file basenames remaining under `dir`.
fn list_basenames_sorted(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .expect("read_dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn deletes_only_engines_for_other_sms_and_keeps_sidecars_aligned() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path();

    let _ = write_file(dir, "eng1_sm86.engine", b"sm86-plan");
    let bytes_sm89 = write_file(dir, "eng2_sm89.engine", b"sm89-plan-larger");
    let _ = write_file(dir, "eng3_sm120.engine", b"sm120-plan");
    let _ = write_file(dir, "non_engine_file.txt", b"this is not an engine");
    // Sidecar for the SM=120 engine must be preserved.
    let _ = write_file(dir, "eng3_sm120.engine.profile", b"sm120-profile");

    let stats = gc_stale_sm_plans(dir, "sm120");

    assert_eq!(
        stats.plans_deleted, 2,
        "expected sm86+sm89 plans to be deleted, got {stats:?}"
    );
    assert!(
        stats.bytes_freed >= bytes_sm89,
        "bytes_freed should include at least the sm89 plan size, got {stats:?}"
    );
    let mut observed = stats.other_sms_observed.clone();
    observed.sort();
    assert_eq!(observed, vec!["sm86".to_string(), "sm89".to_string()]);

    let remaining = list_basenames_sorted(dir);
    assert_eq!(
        remaining,
        vec![
            "eng3_sm120.engine".to_string(),
            "eng3_sm120.engine.profile".to_string(),
            "non_engine_file.txt".to_string(),
        ],
        "non-engine file, sm120 engine, and its sidecar must survive: {remaining:?}"
    );
}

#[test]
fn deletes_aligned_sidecars_for_stale_engines() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path();

    // SM89 engine plus all known sidecar artifacts the TRT EP writes
    // alongside it.
    let plan_bytes = write_file(dir, "eng_sm89.engine", b"sm89-plan");
    let mut sidecar_bytes_total: u64 = 0;
    for suffix in ENGINE_SIDE_SUFFIXES {
        let name = format!("eng_sm89.engine{suffix}");
        sidecar_bytes_total += write_file(dir, &name, b"sidecar");
    }
    let expected_bytes = plan_bytes + sidecar_bytes_total;

    // A current-SM engine that must not be touched.
    let _ = write_file(dir, "eng_sm120.engine", b"sm120-plan");
    let _ = write_file(dir, "eng_sm120.engine.profile", b"sm120-profile");

    let stats = gc_stale_sm_plans(dir, "sm120");

    assert_eq!(stats.plans_deleted, 1, "exactly one stale plan deleted");
    assert_eq!(stats.other_sms_observed, vec!["sm89".to_string()]);
    assert!(
        stats.bytes_freed >= expected_bytes,
        "bytes_freed ({}) must include plan ({}) + sidecar ({}) sizes; got {stats:?}",
        stats.bytes_freed,
        plan_bytes,
        sidecar_bytes_total,
    );

    let remaining = list_basenames_sorted(dir);
    // All sm89-tagged artifacts (plan + every aligned sidecar) must be gone.
    assert!(
        remaining.iter().all(|n| !n.contains("sm89")),
        "no sm89 artifacts should survive: {remaining:?}"
    );
    // Current-SM artifacts must survive.
    assert!(remaining.iter().any(|n| n == "eng_sm120.engine"));
    assert!(remaining.iter().any(|n| n == "eng_sm120.engine.profile"));
}

#[test]
fn empty_directory_is_zero_op() {
    let tmp = TempDir::new().expect("tempdir");
    let stats = gc_stale_sm_plans(tmp.path(), "sm120");
    assert_eq!(stats.plans_deleted, 0);
    assert_eq!(stats.bytes_freed, 0);
    assert!(stats.other_sms_observed.is_empty());
}

#[test]
fn missing_directory_is_silent_no_op() {
    let tmp = TempDir::new().expect("tempdir");
    let missing = tmp.path().join("never-existed");
    let stats = gc_stale_sm_plans(&missing, "sm120");
    assert_eq!(stats.plans_deleted, 0);
    assert_eq!(stats.bytes_freed, 0);
    assert!(stats.other_sms_observed.is_empty());
}

#[test]
fn unrecognized_basename_without_smxx_suffix_is_never_deleted() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path();

    // No `_smXX` token at all — engine plan with an unusual basename.
    let _ = write_file(dir, "weird_no_sm_tag.engine", b"plan-no-sm");
    // Plan whose tag is NOT a recognized `_smXX` pattern (e.g. trailing
    // version qualifier embedded by a future TRT version) — also must be
    // preserved.
    let _ = write_file(dir, "engine_sm89-rev3.engine", b"plan-with-rev");
    // And a clean current-SM plan as a positive control.
    let _ = write_file(dir, "good_sm120.engine", b"sm120");

    let stats = gc_stale_sm_plans(dir, "sm120");

    assert_eq!(
        stats.plans_deleted, 0,
        "no plan without a recognized other-SM tag should be deleted: {stats:?}"
    );
    assert!(stats.other_sms_observed.is_empty());

    let remaining = list_basenames_sorted(dir);
    assert_eq!(
        remaining.len(),
        3,
        "all three plans must survive: {remaining:?}"
    );
}

#[test]
fn ignores_subdirectories_does_not_recurse() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path();

    let _ = write_file(dir, "eng_sm120.engine", b"sm120");

    // A nested directory that itself contains a stale-SM plan must not
    // be traversed. We deliberately do not recurse (TRT writes engines
    // flat in the trt-engines directory).
    let nested = dir.join("nested");
    fs::create_dir(&nested).expect("mkdir");
    let _ = write_file(&nested, "stale_sm89.engine", b"sm89");

    let stats = gc_stale_sm_plans(dir, "sm120");
    assert_eq!(stats.plans_deleted, 0);
    assert!(stats.other_sms_observed.is_empty());

    assert!(
        nested.join("stale_sm89.engine").exists(),
        "nested stale_sm89.engine must not be touched (GC does not recurse)"
    );
}

#[test]
fn current_sm_only_is_zero_op_idempotent() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path();

    let _ = write_file(dir, "a_sm120.engine", b"a");
    let _ = write_file(dir, "b_sm120.engine", b"b");
    let _ = write_file(dir, "b_sm120.engine.profile", b"profile");

    let first = gc_stale_sm_plans(dir, "sm120");
    let second = gc_stale_sm_plans(dir, "sm120");

    assert_eq!(first.plans_deleted, 0);
    assert_eq!(second.plans_deleted, 0);
    assert_eq!(first.bytes_freed, 0);
    assert_eq!(second.bytes_freed, 0);
    assert!(first.other_sms_observed.is_empty());
    assert!(second.other_sms_observed.is_empty());

    let remaining = list_basenames_sorted(dir);
    assert_eq!(
        remaining,
        vec![
            "a_sm120.engine".to_string(),
            "b_sm120.engine".to_string(),
            "b_sm120.engine.profile".to_string(),
        ]
    );
}

#[test]
fn handles_multiple_stale_sms_independently() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path();

    let _ = write_file(dir, "a_sm75.engine", b"sm75");
    let _ = write_file(dir, "b_sm86.engine", b"sm86");
    let _ = write_file(dir, "c_sm89.engine", b"sm89");
    let _ = write_file(dir, "d_sm120.engine", b"sm120");

    let stats = gc_stale_sm_plans(dir, "sm89");

    assert_eq!(stats.plans_deleted, 3);
    let mut observed = stats.other_sms_observed.clone();
    observed.sort();
    assert_eq!(
        observed,
        vec!["sm120".to_string(), "sm75".to_string(), "sm86".to_string()]
    );

    let remaining = list_basenames_sorted(dir);
    assert_eq!(remaining, vec!["c_sm89.engine".to_string()]);
}

#[test]
fn gcstats_default_is_zeroed() {
    let s = GcStats::default();
    assert_eq!(s.plans_deleted, 0);
    assert_eq!(s.bytes_freed, 0);
    assert!(s.other_sms_observed.is_empty());
}

/// Non-readable directory: `gc_stale_sm_plans` must not panic and must
/// return zero-stats (the `read_dir` call silently returns an empty
/// iterator when it cannot enumerate, so no plan is deleted).
#[cfg(unix)]
#[test]
fn gc_stale_sm_plans_on_unreadable_dir_does_not_panic() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = TempDir::new().expect("tempdir");
    let cache_dir = tmp.path().to_path_buf();

    std::fs::set_permissions(&cache_dir, std::fs::Permissions::from_mode(0o000))
        .expect("chmod 000");

    let stats = gc_stale_sm_plans(&cache_dir, "sm89");

    // Restore permissions so TempDir cleanup does not fail.
    std::fs::set_permissions(&cache_dir, std::fs::Permissions::from_mode(0o755))
        .expect("chmod 755");

    assert_eq!(
        stats.plans_deleted, 0,
        "unreadable dir must yield zero deletions"
    );
    assert_eq!(
        stats.bytes_freed, 0,
        "unreadable dir must yield zero bytes freed"
    );
}

/// Symlink directory: GC must treat a symlink masquerading as the cache
/// dir gracefully (the resolved target is enumerated normally if it's a
/// directory). The contract is "no panic, no surprise crash"; either
/// outcome (deleted via link, or skipped) is acceptable.
#[cfg(target_family = "unix")]
#[test]
fn symlinked_cache_dir_does_not_panic() {
    use std::os::unix::fs::symlink;

    let tmp = TempDir::new().expect("tempdir");
    let real = tmp.path().join("real");
    fs::create_dir(&real).expect("mkdir");
    let _ = write_file(&real, "x_sm89.engine", b"sm89-stale");
    let _ = write_file(&real, "y_sm120.engine", b"sm120-current");

    let link = tmp.path().join("link");
    symlink(&real, &link).expect("symlink");

    let stats = gc_stale_sm_plans(&link, "sm120");
    assert!(stats.plans_deleted <= 1);
}
