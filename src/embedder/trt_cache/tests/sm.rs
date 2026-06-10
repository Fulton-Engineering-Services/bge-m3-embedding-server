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
use std::path::Path;
use tempfile::TempDir;

use super::super::enumerate::{
    count_engine_files, count_engine_files_for_sm, engine_basenames_for_sm,
    engine_basenames_in_dir_sorted, engine_files_for_sm, matches_sm_suffix,
};
use super::super::inspect::run_write_probe;

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
    assert!(engine_basenames_in_dir_sorted(&dir).unwrap().is_empty());
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

    let names = engine_basenames_in_dir_sorted(&dir).unwrap();
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
    assert!(engine_basenames_in_dir_sorted(&missing).is_err());
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
/// `_sm120.engine` plan. A prior heterogeneous-cache bug hinged on
/// accidentally counting Blackwell plans as "the cache is warm for sm12" or vice
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

/// **The exact heterogeneous-cache scenario** observed in production.
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
