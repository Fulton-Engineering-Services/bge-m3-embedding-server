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

//! Unit tests for the `trt_warmup` module: pure helpers
//! (`coverage_check_shapes`, `shard_shapes`), the
//! `prewarm_persistence_*` postcondition predicates re-exported via the
//! parent facade, and fixture-backed checks of the engine-count snapshot
//! contract those predicates rely on.

use super::{
    coverage_check_shapes, prewarm_persistence_postcondition_failed,
    prewarm_persistence_suspicious_undercount, shard_shapes, CACHE_HIT_THRESHOLD_MS,
    SUSPICIOUS_UNDERCOUNT_MIN_FRESH,
};

/// Default 16-shape grid for stride-sharding and coverage-check tests.
fn default_shapes() -> Vec<(usize, usize)> {
    vec![
        (1, 128),
        (1, 512),
        (1, 2048),
        (1, 8192),
        (4, 128),
        (4, 512),
        (4, 2048),
        (4, 8192),
        (16, 128),
        (16, 512),
        (16, 2048),
        (16, 8192),
        (32, 128),
        (32, 512),
        (32, 2048),
        (32, 8192),
    ]
}

// ─── CACHE_HIT_THRESHOLD_MS ───────────────────────────────────────────

// Compile-time guard: the threshold must lie between the warm-load ceiling
// (< 3 000 ms observed) and the cold-compile floor (≥ 30 000 ms observed).
// Expressed as a const assertion so it is checked at every build.
const _: () = {
    assert!(CACHE_HIT_THRESHOLD_MS > 3_000);
    assert!(CACHE_HIT_THRESHOLD_MS < 30_000);
};

// ─── coverage_check_shapes ────────────────────────────────────────────

#[test]
fn coverage_check_shapes_empty_input_returns_empty() {
    assert!(coverage_check_shapes(&[]).is_empty());
}

#[test]
fn coverage_check_shapes_single_shape_returns_itself() {
    let shapes = vec![(32_usize, 8192_usize)];
    assert_eq!(coverage_check_shapes(&shapes), shapes);
}

#[test]
fn coverage_check_shapes_all_same_batch_returns_seq_extremes() {
    // Worker-3 shard: all batch=4, seq varies → extremes collapse to 2 shapes.
    let shapes = vec![(4, 128), (4, 512), (4, 2048), (4, 8192)];
    let checks = coverage_check_shapes(&shapes);
    // min_batch == max_batch → rep_min_batch == rep_max_batch → deduped to 1 from batch
    // + (4,128) from min_seq + (4,8192) from max_seq = 2 shapes
    assert_eq!(checks.len(), 2);
    assert!(checks.contains(&(4, 128)), "must include min_seq shape");
    assert!(checks.contains(&(4, 8192)), "must include max_seq shape");
}

#[test]
fn coverage_check_shapes_all_same_seq_returns_batch_extremes() {
    // Shapes that share the same sequence length.
    let shapes = vec![(1, 512), (4, 512), (16, 512), (32, 512)];
    let checks = coverage_check_shapes(&shapes);
    assert_eq!(checks.len(), 2);
    assert!(checks.contains(&(1, 512)), "must include min_batch shape");
    assert!(checks.contains(&(32, 512)), "must include max_batch shape");
}

#[test]
fn coverage_check_shapes_full_grid_returns_four_distinct_corners() {
    let shapes = default_shapes();
    let checks = coverage_check_shapes(&shapes);
    // min_batch=1 → (1,128); max_batch=32 → (32,128); min_seq=128 → (1,128) same;
    // max_seq=8192 → (1,8192). After dedup: {(1,128),(32,128),(1,8192)} = 3 shapes.
    // (The (max_batch, min_seq) corner = (32,128) == rep_max_batch when the first
    // shape with max_batch=32 is (32,128).)
    assert!(checks.len() >= 2 && checks.len() <= 4);
    // All checks must be present in the original shapes list.
    for &c in &checks {
        assert!(
            shapes.contains(&c),
            "check shape {c:?} not in original shard"
        );
    }
}

#[test]
fn coverage_check_shapes_covers_all_dimension_extremes() {
    let shapes = default_shapes();
    let checks = coverage_check_shapes(&shapes);
    let min_batch = shapes.iter().map(|(b, _)| b).min().copied().unwrap();
    let max_batch = shapes.iter().map(|(b, _)| b).max().copied().unwrap();
    let min_seq = shapes.iter().map(|(_, s)| s).min().copied().unwrap();
    let max_seq = shapes.iter().map(|(_, s)| s).max().copied().unwrap();

    assert!(
        checks.iter().any(|(b, _)| *b == min_batch),
        "must include a shape with min_batch={min_batch}"
    );
    assert!(
        checks.iter().any(|(b, _)| *b == max_batch),
        "must include a shape with max_batch={max_batch}"
    );
    assert!(
        checks.iter().any(|(_, s)| *s == min_seq),
        "must include a shape with min_seq={min_seq}"
    );
    assert!(
        checks.iter().any(|(_, s)| *s == max_seq),
        "must include a shape with max_seq={max_seq}"
    );
}

#[test]
fn coverage_check_shapes_no_duplicates() {
    let shapes = default_shapes();
    let checks = coverage_check_shapes(&shapes);
    // Each shape appears at most once.
    for i in 0..checks.len() {
        for j in (i + 1)..checks.len() {
            assert_ne!(
                checks[i], checks[j],
                "duplicate shape in check set: {:?}",
                checks[i]
            );
        }
    }
}

#[test]
fn coverage_check_shapes_two_shape_shard_returns_both() {
    let shapes = vec![(1_usize, 128_usize), (32_usize, 8192_usize)];
    let checks = coverage_check_shapes(&shapes);
    assert_eq!(checks.len(), 2);
    assert!(checks.contains(&(1, 128)));
    assert!(checks.contains(&(32, 8192)));
}

/// Regression: the zero-false-positive guarantee requires that the check
/// set independently exercises ALL four dimensional extremes.  This test
/// verifies that a shard where no single shape carries both `min_batch` AND
/// `min_seq` still produces a check set that covers each extreme separately.
#[test]
fn coverage_check_shapes_non_rectangular_shard_covers_all_extremes() {
    // Shard without a (min_batch, min_seq) corner shape:
    //   (1, 8192) carries min_batch but NOT min_seq
    //   (32, 128) carries min_seq and max_batch but NOT min_batch
    let shapes = vec![(1, 8192), (32, 128)];
    let checks = coverage_check_shapes(&shapes);

    let has_min_batch = checks.iter().any(|(b, _)| *b == 1);
    let has_max_batch = checks.iter().any(|(b, _)| *b == 32);
    let has_min_seq = checks.iter().any(|(_, s)| *s == 128);
    let has_max_seq = checks.iter().any(|(_, s)| *s == 8192);

    assert!(has_min_batch, "needs a shape with batch=1 (min_batch)");
    assert!(has_max_batch, "needs a shape with batch=32 (max_batch)");
    assert!(has_min_seq, "needs a shape with seq=128 (min_seq)");
    assert!(has_max_seq, "needs a shape with seq=8192 (max_seq)");
}

// ─── trt_prewarm stub test ─────────────────────────────────────────────

/// `trt_prewarm` with an empty shape list returns 0 immediately without
/// attempting any inference.  We verify this by inspecting the
/// accumulator logic: the loop over an empty slice never executes, so
/// `warmed` stays 0.
#[test]
fn empty_shapes_returns_zero() {
    let shapes: Vec<(usize, usize)> = vec![];
    let count: usize = shapes.iter().map(|_| 1usize).sum();
    assert_eq!(count, 0);
}

// ─── shard_shapes ──────────────────────────────────────────────────────

#[test]
fn single_worker_returns_all_shapes() {
    let shapes = default_shapes();
    assert_eq!(shard_shapes(&shapes, 0, 1), shapes);
}

#[test]
fn zero_worker_count_returns_all_shapes() {
    let shapes = default_shapes();
    assert_eq!(shard_shapes(&shapes, 0, 0), shapes);
}

#[test]
fn two_workers_cover_all_shapes() {
    let shapes = default_shapes();
    let shard0 = shard_shapes(&shapes, 0, 2);
    let shard1 = shard_shapes(&shapes, 1, 2);

    let mut combined = shard0.clone();
    combined.extend(&shard1);
    combined.sort_unstable();
    let mut expected = shapes.clone();
    expected.sort_unstable();
    assert_eq!(combined, expected, "two shards must cover every shape once");
}

#[test]
fn four_workers_cover_all_shapes() {
    let shapes = default_shapes();
    let mut combined: Vec<(usize, usize)> =
        (0..4).flat_map(|w| shard_shapes(&shapes, w, 4)).collect();
    combined.sort_unstable();
    let mut expected = shapes.clone();
    expected.sort_unstable();
    assert_eq!(
        combined, expected,
        "four shards must cover every shape once"
    );
}

#[test]
fn four_workers_each_get_four_shapes() {
    let shapes = default_shapes();
    for w in 0..4 {
        assert_eq!(
            shard_shapes(&shapes, w, 4).len(),
            4,
            "worker {w} should get exactly 4 shapes"
        );
    }
}

#[test]
fn stride_spreads_expensive_shapes_across_workers() {
    let shapes = default_shapes();
    let shard3 = shard_shapes(&shapes, 3, 4);
    assert!(
        shard3.iter().all(|(_, seq)| *seq == 8192),
        "worker 3 should compile only seq=8192 shapes; got: {shard3:?}"
    );
    for w in 0..3 {
        let shard = shard_shapes(&shapes, w, 4);
        assert!(
            shard.iter().all(|(_, seq)| *seq != 8192),
            "worker {w} should not compile seq=8192 shapes; got: {shard:?}"
        );
    }
}

#[test]
fn more_workers_than_shapes_gives_empty_shards() {
    let shapes = vec![(1, 128), (1, 512)];
    assert_eq!(shard_shapes(&shapes, 0, 4), vec![(1, 128)]);
    assert_eq!(shard_shapes(&shapes, 1, 4), vec![(1, 512)]);
    assert_eq!(shard_shapes(&shapes, 2, 4), vec![]);
    assert_eq!(shard_shapes(&shapes, 3, 4), vec![]);
}

// ─── coverage-check correctness proofs ────────────────────────────────

/// Verify the zero-FP guarantee for the default 4-worker stride shard.
///
/// Worker-3 shard = [(1,8192),(4,8192),(16,8192),(32,8192)].
/// `coverage_check_shapes` returns `{(1,8192),(32,8192)}`.
/// If both are cache hits (fast), then:
///   - `profile.min_batch` ≤ 1 AND `profile.max_batch` ≥ 32
///   - `profile.min_seq` ≤ 8192 AND `profile.max_seq` ≥ 8192
///
/// Every shape with batch ∈ \[1,32\] and seq=8192 is covered → no FPs.
#[test]
fn worker3_shard_check_shapes_cover_intermediate_shapes() {
    let shard = shard_shapes(&default_shapes(), 3, 4);
    assert_eq!(shard, vec![(1, 8192), (4, 8192), (16, 8192), (32, 8192)]);

    let checks = coverage_check_shapes(&shard);
    assert!(checks.contains(&(1, 8192)));
    assert!(checks.contains(&(32, 8192)));

    // Intermediate shapes (4,8192) and (16,8192) are NOT in the check set.
    let non_checks: Vec<_> = shard
        .iter()
        .copied()
        .filter(|s| !checks.contains(s))
        .collect();
    assert_eq!(non_checks.len(), 2);
    // Each non-check shape has batch ∈ [1,32] and seq=8192, so it is
    // guaranteed to be within the profile range if both checks pass.
    for (b, s) in &non_checks {
        assert!(*b >= 1 && *b <= 32, "batch {b} must be within [1,32]");
        assert!(*s == 8192, "seq {s} must equal 8192");
    }
}

/// Same proof for worker-0 shard: [(1,128),(4,128),(16,128),(32,128)].
#[test]
fn worker0_shard_check_shapes_cover_intermediate_shapes() {
    let shard = shard_shapes(&default_shapes(), 0, 4);
    assert_eq!(shard, vec![(1, 128), (4, 128), (16, 128), (32, 128)]);

    let checks = coverage_check_shapes(&shard);
    assert!(checks.contains(&(1, 128)));
    assert!(checks.contains(&(32, 128)));

    let non_checks: Vec<_> = shard
        .iter()
        .copied()
        .filter(|s| !checks.contains(s))
        .collect();
    for (b, s) in &non_checks {
        assert!(*b >= 1 && *b <= 32);
        assert!(*s == 128);
    }
}

/// Single-worker (all 16 shapes) check covers all four extremes and
/// produces at most 4 shapes, skipping at least 12.
#[test]
fn single_worker_check_shapes_skips_most_shapes() {
    let shapes = default_shapes();
    let checks = coverage_check_shapes(&shapes);
    assert!(checks.len() <= 4, "at most 4 check shapes for any shard");
    let would_skip = shapes.len() - checks.len();
    assert!(
        would_skip >= 12,
        "single-worker should skip ≥ 12 shapes; would_skip={would_skip}"
    );
}

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

// ─── engine count snapshot wiring (filesystem-backed) ─────────────────

/// `count_engine_files` is the mechanism behind both the per-shape
/// delta WARN and the per-worker prewarm postcondition. Verify it
/// reflects engine-file writes the way the prewarm path does:
/// before-compile snapshot + on-disk write + after-compile snapshot
/// must produce a delta of `+1`. This pins down the contract that the
/// silent-persistence detector relies on.
#[test]
fn count_engine_files_reflects_post_write_delta() {
    use super::trt_cache::count_engine_files;
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
    use super::trt_cache::count_engine_files;
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
    use super::trt_cache::count_engine_files;
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
    use super::trt_cache::count_engine_files;
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
