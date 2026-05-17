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

//! Tests for `prewarm_persistence_postcondition_failed` and
//! `prewarm_persistence_suspicious_undercount`.

use super::super::{
    SUSPICIOUS_UNDERCOUNT_MIN_FRESH, prewarm_persistence_postcondition_failed,
    prewarm_persistence_suspicious_undercount,
};

// ─── prewarm_persistence_postcondition_failed ─────────────────────────
//
// Signatures: (fresh_compiles: usize, engine_count_after: usize) -> bool
// Logic:      fresh_compiles > 0 && engine_count_after == 0
//
// The postcondition is keyed on `engine_count_after == 0`, NOT on
// `engine_count_delta <= 0`.  ORT's TRT EP writes one profile-based
// `.engine` file and rewrites it in-place as the profile range expands
// to cover more shapes — `delta == 0` at steady state is healthy.
// Only `after == 0` (no engine file at all) is a persistence failure.

/// Exact production defect signal: fresh compiles occurred but the engine
/// cache directory is empty (after == 0) → flag as ERROR.
#[test]
fn postcondition_flags_fresh_compiles_with_zero_engine_count_after() {
    assert!(prewarm_persistence_postcondition_failed(16, 0));
    assert!(prewarm_persistence_postcondition_failed(1, 0));
    // Production-scale: 1215 compile-success events, 0 engines on disk.
    assert!(prewarm_persistence_postcondition_failed(1215, 0));
}

/// Healthy first-deploy cold cache: 16 fresh compiles, engine file present
/// → accept.  The exact count of engine files does not matter; >= 1 passes.
#[test]
fn postcondition_accepts_fresh_compiles_with_engine_files_present() {
    assert!(!prewarm_persistence_postcondition_failed(16, 16));
    assert!(!prewarm_persistence_postcondition_failed(16, 1));
}

/// Healthy partial-shard compile: 4 fresh compiles, engine file present
/// → accept.
#[test]
fn postcondition_accepts_partial_shard_with_engine_files_present() {
    assert!(!prewarm_persistence_postcondition_failed(4, 4));
}

/// Healthy warm-cache fast path: 0 fresh compiles, 0 engine files on disk
/// → accept.  Cache hits only must NOT be flagged.
#[test]
fn postcondition_accepts_warm_cache_with_no_compiles() {
    assert!(!prewarm_persistence_postcondition_failed(0, 0));
}

/// Healthy: 0 fresh compiles but engine files already present (a sibling
/// worker on the same EFS-shared cache wrote engines before we ran).
#[test]
fn postcondition_accepts_zero_compiles_with_engine_files_present() {
    assert!(!prewarm_persistence_postcondition_failed(0, 4));
}

/// KEY profile-update case (the false-positive the old delta-based rule
/// produced): N fresh compiles occurred, engine file count stayed at 1
/// because TRT EP rewrote the same file in-place for each new shape.
/// Must NOT trip the ERROR postcondition.
///
/// Confirmed production evidence: every `engine compiled, cached, and
/// fsynced` log showed `before=1, after=1, increased=0` on workers 0/1/3.
#[test]
fn postcondition_accepts_profile_update_case_after_one() {
    for fresh in [1_usize, 2, 4, 15, 16, 1215] {
        assert!(
            !prewarm_persistence_postcondition_failed(fresh, 1),
            "fresh={fresh}, after=1: profile-update case must not trip ERROR"
        );
    }
}

/// Boundary: `fresh=0, after=0` → no compiles, no files → accept (no work
/// done on this shard; warm-cache skip path).
#[test]
fn postcondition_boundary_fresh_zero_after_zero_passes() {
    assert!(!prewarm_persistence_postcondition_failed(0, 0));
}

/// Boundary: `fresh=1, after=1` → one compile, file present → accept.
#[test]
fn postcondition_boundary_fresh_one_after_one_passes() {
    assert!(!prewarm_persistence_postcondition_failed(1, 1));
}

/// Boundary: `fresh=2, after=1` → two compiles but only one engine file
/// (profile-update rewrite) → accept.
#[test]
fn postcondition_boundary_fresh_two_after_one_passes() {
    assert!(!prewarm_persistence_postcondition_failed(2, 1));
}

/// Boundary: `fresh=1, after=0` → one compile, no engine file → fail.
/// This is the minimal form of the production defect.
#[test]
fn postcondition_boundary_fresh_one_after_zero_fails() {
    assert!(prewarm_persistence_postcondition_failed(1, 0));
}

/// Any `fresh > 0` with `after > 0` must pass, regardless of the exact
/// counts (covers engine reuse, multi-file, etc).
#[test]
fn postcondition_passes_whenever_engine_files_exist() {
    for fresh in [1_usize, 2, 4, 16, 128, 1215] {
        for after in [1_usize, 2, 16, 1215] {
            assert!(
                !prewarm_persistence_postcondition_failed(fresh, after),
                "fresh={fresh}, after={after}: any after > 0 must pass"
            );
        }
    }
}

// ─── prewarm_persistence_suspicious_undercount ────────────────────────
//
// Signatures: (fresh_compiles: usize, engine_count_after: usize) -> bool
//
// Current behaviour: always returns `false` when `engine_count_after > 0`.
// TRT EP profile-based in-place engine rewrite means delta == 0 at
// steady state; flagging that would produce false-positive WARNs on every
// shape after the first compile.  The only actionable anomaly (after == 0)
// is already covered by the ERROR predicate.

/// Profile-update case: fresh compiles but engine file count stays at 1
/// (TRT EP in-place rewrite).  WARN must be silent — this is healthy.
#[test]
fn suspicious_undercount_silent_for_profile_update_case() {
    for fresh in [1_usize, 2, 4, 10, 15, 16, 1215] {
        assert!(
            !prewarm_persistence_suspicious_undercount(fresh, 1),
            "fresh={fresh}, after=1: profile-update case must not trip WARN"
        );
    }
}

/// WARN must be silent whenever engine files exist, regardless of how many
/// fresh compiles were recorded.
#[test]
fn suspicious_undercount_silent_when_engine_files_present() {
    for &(fresh, after) in &[
        (2_usize, 1_usize),
        (4, 2),
        (10, 1),
        (16, 7),
        (16, 8),
        (16, 16),
        (1215, 1),
        (1215, 5),
        (1215, 700),
        (1215, 1215),
    ] {
        assert!(
            !prewarm_persistence_suspicious_undercount(fresh, after),
            "WARN must be silent for fresh={fresh}, after={after} (engine files present)"
        );
    }
}

/// WARN must NOT double-fire on top of the ERROR.  When `after == 0` and
/// `fresh > 0`, the ERROR postcondition fires; the WARN suppresses itself.
#[test]
fn suspicious_undercount_suppressed_when_error_fires() {
    assert!(prewarm_persistence_postcondition_failed(1215, 0));
    assert!(
        !prewarm_persistence_suspicious_undercount(1215, 0),
        "WARN must not fire when ERROR is already covering the same case"
    );
    assert!(prewarm_persistence_postcondition_failed(4, 0));
    assert!(!prewarm_persistence_suspicious_undercount(4, 0));
}

/// WARN is silent for trivial shards (`fresh < MIN_FRESH`).
#[test]
fn suspicious_undercount_silent_below_min_fresh_threshold() {
    // `fresh = 0` is the warm-cache case — never flag.
    assert!(!prewarm_persistence_suspicious_undercount(0, 0));
    assert!(!prewarm_persistence_suspicious_undercount(0, 4));
    // `fresh = 1` is below the min-fresh floor — never flag.
    assert!(!prewarm_persistence_suspicious_undercount(1, 1));
    // Compile-time pin: changing the floor below 2 would make
    // single-compile boundary cases noisy.
    const { assert!(SUSPICIOUS_UNDERCOUNT_MIN_FRESH >= 2) }
}

/// Robustness: large `after` values (e.g. an unusually large engine cache)
/// must not panic and must remain silent.
#[test]
fn suspicious_undercount_silent_for_large_engine_counts() {
    assert!(!prewarm_persistence_suspicious_undercount(4, usize::MAX));
    assert!(!prewarm_persistence_suspicious_undercount(
        usize::MAX,
        usize::MAX
    ));
}
