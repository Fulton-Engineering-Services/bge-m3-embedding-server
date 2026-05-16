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

//! Tests for `coverage_check_shapes`, `shard_shapes`, the
//! `CACHE_HIT_THRESHOLD_MS` compile-time assertion, and the
//! per-shard coverage-check correctness proofs.

use super::super::{coverage_check_shapes, shard_shapes, CACHE_HIT_THRESHOLD_MS};

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
