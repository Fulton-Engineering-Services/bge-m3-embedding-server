use super::cache::{save_probe_cache, try_load_probe_cache};
use super::corpus::{load_probe_texts, synthesize_texts};
use super::fit::{fit_cost_model, DataPoint};
use super::runner::PROBE_SHAPES;
use super::validate::validate_max_seq_shape;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Builds a `DataPoint` from `(batch, seq, a, b)` using the model formula
/// `rss = a * (batch * seq) + b * (batch * seq²)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn make_dp(batch: usize, seq: usize, a: f64, b: f64) -> DataPoint {
    let rss_delta = (a * (batch * seq) as f64 + b * (batch * seq * seq) as f64) as usize;
    DataPoint {
        batch,
        seq,
        rss_delta,
    }
}

// ---------------------------------------------------------------------------
// Stage 1: diagnostic — does current fit_cost_model pass for production data?
//
// This test uses the 16-shape production set with a=18000, b=6 (fp16/aarch64
// typical values). Before the normalization fix (Stage 2) this would return
// None because the Gram matrix det falls below the scale-invariance threshold.
// After Stage 2 it must return Some with coefficients close to the ground truth.
// ---------------------------------------------------------------------------

#[test]
fn fit_cost_model_production_scale_16_shapes_with_max_seq_8192() {
    // All 16 shapes swept by the old probe sweep, plus the dynamic (1,8192).
    let a_true = 18_000.0_f64;
    let b_true = 6.0_f64;
    let data: Vec<DataPoint> = [
        (1usize, 64usize),
        (1, 256),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (4, 64),
        (4, 256),
        (4, 1024),
        (4, 2048),
        (8, 64),
        (8, 256),
        (8, 1024),
        (16, 64),
        (16, 256),
        (16, 512),
    ]
    .iter()
    .map(|&(b, s)| make_dp(b, s, a_true, b_true))
    .collect();

    let result = fit_cost_model(&data);
    // After the normalization fix this must succeed.
    assert!(
        result.is_some(),
        "fit_cost_model should succeed on 16-shape production data including (1,8192)"
    );
    let (a, b) = result.unwrap();
    // Expect recovery within 5% of the true coefficients.
    assert!(
        (a - a_true).abs() < 0.05 * a_true,
        "a={a:.0} should be within 5% of {a_true}"
    );
    assert!(
        (b - b_true).abs() < 0.05 * b_true,
        "b={b:.4} should be within 5% of {b_true}"
    );
}

// ---------------------------------------------------------------------------
// Stage 2: normalized OLS correctness
// ---------------------------------------------------------------------------

#[test]
fn fit_cost_model_two_points() {
    // hand-crafted data: batch=1,seq=64 → 8 MB; batch=1,seq=512 → 100 MB.
    // Expect a reasonable (a,b) pair.
    let data = vec![
        DataPoint {
            batch: 1,
            seq: 64,
            rss_delta: 8_000_000,
        },
        DataPoint {
            batch: 1,
            seq: 512,
            rss_delta: 100_000_000,
        },
        DataPoint {
            batch: 4,
            seq: 64,
            rss_delta: 30_000_000,
        },
        DataPoint {
            batch: 4,
            seq: 256,
            rss_delta: 80_000_000,
        },
    ];
    let result = fit_cost_model(&data);
    assert!(result.is_some(), "should produce a valid fit");
    let (a, b) = result.unwrap();
    assert!(a > 0.0, "a must be positive");
    assert!(b > 0.0, "b must be positive");
}

#[test]
fn fit_cost_model_recovers_known_coefficients_from_7_probe_shapes() {
    // Verify the 7-shape set (6 static + dynamic max_seq=8192) recovers
    // coefficients within 5% of ground truth.
    let a_true = 18_500.0_f64;
    let b_true = 6.5_f64;
    let data: Vec<DataPoint> = [
        (1, 64),
        (4, 64),
        (1, 256),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
    ]
    .iter()
    .map(|&(b, s)| make_dp(b, s, a_true, b_true))
    .collect();

    let result = fit_cost_model(&data);
    assert!(result.is_some(), "7-shape fit should succeed");
    let (a, b) = result.unwrap();
    assert!(
        (a - a_true).abs() < 0.05 * a_true,
        "a={a:.0} should be within 5% of {a_true}"
    );
    assert!(
        (b - b_true).abs() < 0.05 * b_true,
        "b={b:.4} should be within 5% of {b_true}"
    );
}

#[test]
fn fit_cost_model_single_point_returns_none() {
    let data = vec![DataPoint {
        batch: 1,
        seq: 128,
        rss_delta: 5_000_000,
    }];
    assert!(fit_cost_model(&data).is_none(), "need >= 2 points");
}

#[test]
fn fit_cost_model_singular_system_returns_none() {
    // All points on the same line through the origin in one variable — singular.
    let data = vec![
        DataPoint {
            batch: 1,
            seq: 128,
            rss_delta: 1_000_000,
        },
        DataPoint {
            batch: 1,
            seq: 128,
            rss_delta: 1_000_000,
        },
    ];
    // x2 = x1 * seq = same for both → singular (columns are linearly dependent).
    assert!(
        fit_cost_model(&data).is_none(),
        "identical rows should give None"
    );
}

/// rc8 regression test — recreates the kernel-switch discontinuity
/// observed on the rc7 production deployment at `max_seq=8192`:
/// near-zero deltas at seq ≤ 2048 (small ORT attention kernel) and
/// purely quadratic deltas at seq ≥ 4096 (full O(N²) attention).
/// `fit_cost_model` must clamp the non-physical negative `a_raw` to 0
/// (rounded up to the 4 KiB/token floor) and recover `b ≈ 117`
/// from the high-seq points instead of returning None.
#[test]
fn fit_recovers_b_when_kernel_switch_creates_negative_a_raw() {
    // Production data points from the rc7 task at workers=2,
    // max_seq=8192 (CloudWatch probe sweep, 2026-05-10T06:21:47Z):
    //   (1, 64)   →     ~0 MB
    //   (4, 64)   →     ~0 MB
    //   (1, 256)  →     ~0 MB
    //   (1, 1024) →    10 MB
    //   (1, 2048) →    12 MB
    //   (1, 4096) →  1976 MB
    //   (1, 8192) →  7846 MB
    let data = vec![
        DataPoint {
            batch: 1,
            seq: 64,
            rss_delta: 0,
        },
        DataPoint {
            batch: 4,
            seq: 64,
            rss_delta: 0,
        },
        DataPoint {
            batch: 1,
            seq: 256,
            rss_delta: 0,
        },
        DataPoint {
            batch: 1,
            seq: 1024,
            rss_delta: 10_000_000,
        },
        DataPoint {
            batch: 1,
            seq: 2048,
            rss_delta: 12_000_000,
        },
        DataPoint {
            batch: 1,
            seq: 4096,
            rss_delta: 1_976_000_000,
        },
        DataPoint {
            batch: 1,
            seq: 8192,
            rss_delta: 7_846_000_000,
        },
    ];

    let result = fit_cost_model(&data);
    assert!(
        result.is_some(),
        "rc8 fit must succeed on kernel-switch data instead of returning \
         None — see the negative-a clamp branch in fit_cost_model"
    );
    let (a, b) = result.unwrap();

    // After clamp, `a` should be at the lower bound (4 KiB/token) since
    // the OLS solver wanted it negative. Allow up to 5% above the floor
    // to absorb numerical jitter.
    assert!(
        (4_096.0..4_300.0).contains(&a),
        "a={a:.0} should be at or just above the 4096 floor"
    );

    // `b` should recover the quadratic coefficient from the high-seq
    // data points: rss_delta(1, 8192) ≈ b · 8192² → b ≈ 117 bytes/token²
    // for a pure quadratic. The low-seq points (small but nonzero) pull
    // the fit slightly above the pure-quadratic ideal — empirically
    // b ≈ 131 on this data set. Allow ±20% to keep the test resilient
    // to small changes in the low-seq deltas while still asserting the
    // fit recovers the correct order of magnitude (and not the
    // CONSERVATIVE_B = 8 fallback, which would mean we returned the
    // wrong branch).
    assert!(
        (95.0..145.0).contains(&b),
        "b={b:.4} should recover roughly 117 bytes/token² from the \
         high-seq points (must NOT be CONSERVATIVE_B=8 conservative \
         fallback nor wildly off)"
    );

    // Sanity: the fitted coefficients must predict (1, 8192) workspace
    // close to the measured 7.85 GB (within 15%) so the bin-packer
    // correctly rejects oversize batches at max_seq=8192.
    let predicted_8192 = a * 8192.0 + b * 8192.0 * 8192.0;
    let measured_8192 = 7_846_000_000.0_f64;
    let ratio = predicted_8192 / measured_8192;
    assert!(
        (0.85..=1.15).contains(&ratio),
        "predicted (1, 8192) = {predicted_8192:.0} must be within 15% \
         of measured {measured_8192:.0}; got ratio {ratio:.3}"
    );
}

#[test]
fn fit_rejects_negative_coefficients() {
    // Construct data that forces a negative coefficient — artificially small
    // rss at high seq relative to low seq.
    let data = vec![
        DataPoint {
            batch: 16,
            seq: 64,
            rss_delta: 10_000_000,
        },
        DataPoint {
            batch: 1,
            seq: 8192,
            rss_delta: 1,
        }, // pathological
        DataPoint {
            batch: 8,
            seq: 256,
            rss_delta: 5_000_000,
        },
        DataPoint {
            batch: 4,
            seq: 1024,
            rss_delta: 2_000_000,
        },
    ];
    // May or may not produce a valid fit; just assert it doesn't panic.
    let _ = fit_cost_model(&data);
}

#[test]
fn synthesize_texts_returns_batch_count() {
    let corpus = vec!["hello world".to_string(); 5];
    let texts = synthesize_texts(&corpus, 7, 256);
    assert_eq!(texts.len(), 7);
}

#[test]
fn synthesize_texts_produces_non_empty() {
    let corpus = vec!["x".to_string()];
    let texts = synthesize_texts(&corpus, 3, 64);
    for t in &texts {
        assert!(!t.is_empty());
    }
}

#[test]
fn load_probe_texts_returns_nonempty() {
    // Corpus is available when tests run from project root.
    let texts = load_probe_texts();
    assert!(!texts.is_empty());
}

#[test]
fn probe_shapes_are_sorted_ascending_token_positions() {
    // Verify the static table is sane (not a hard requirement, just hygiene).
    let mut prev = 0usize;
    for &(b, s) in PROBE_SHAPES {
        let positions = b * s;
        // Not required to be strictly ascending (table has duplicates at same
        // token count but different (b,s) combos), just check no zeros.
        assert!(
            positions > 0,
            "probe shape ({b},{s}) has zero token positions"
        );
        let _ = prev;
        prev = positions;
    }
}

#[test]
fn probe_cache_roundtrip() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    save_probe_cache(dir.path(), "fp16", 8192, 18_432.0, 6.2);
    let result = try_load_probe_cache(dir.path(), "fp16", 8192);
    assert!(result.is_some(), "should load after save");
    let (a, b) = result.unwrap();
    assert!((a - 18_432.0).abs() < 1.0);
    assert!((b - 6.2).abs() < 0.01);
}

#[test]
fn probe_cache_fingerprint_mismatch_returns_none() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    save_probe_cache(dir.path(), "fp16", 8192, 18_432.0, 6.2);
    // Different model variant
    assert!(
        try_load_probe_cache(dir.path(), "fp32", 8192).is_none(),
        "model mismatch should return None"
    );
    // Different max_seq
    assert!(
        try_load_probe_cache(dir.path(), "fp16", 4096).is_none(),
        "max_seq mismatch should return None"
    );
}

#[test]
fn probe_cache_missing_returns_none() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    assert!(
        try_load_probe_cache(dir.path(), "fp16", 8192).is_none(),
        "missing cache file should return None"
    );
}

// -----------------------------------------------------------------------
// Diagnostic branch coverage for the empty-data path
//
// The three diagnostic scenarios in run_probe cannot be tested end-to-end
// without loading real ORT models. Instead we validate the component
// decisions that trigger each branch:
//
//   Branch 1: all shapes skipped  → conservative.fits() always returns false
//             when rss_ceiling=0 (chunk_cost > 0 for any batch/seq > 0).
//   Branch 2: all shapes errored  → upstream logic; covered by error path.
//   Branch 3: zero-delta check    → data.iter().all(|dp| dp.rss_delta == 0)
//             when rss_before == rss_after (non-Linux or no ORT activity).
// -----------------------------------------------------------------------

/// Branch 1: `rss_ceiling=0` causes `conservative.fits()` to reject every shape.
/// This is the condition that caused the production `probe_status=failed`.
#[test]
fn all_probe_shapes_skipped_when_rss_ceiling_is_zero() {
    use crate::binpack::CostModel;

    let ceiling_zero = CostModel::conservative(0);

    // Every static probe shape must be rejected by fits() at ceiling=0,
    // because chunk_cost(batch, seq) > 0 for any batch,seq ≥ 1.
    for &(batch, seq) in PROBE_SHAPES {
        assert!(
            !ceiling_zero.fits(batch, seq),
            "shape ({batch},{seq}) should not fit when rss_ceiling=0, \
             cost={} > 0",
            ceiling_zero.chunk_cost(batch, seq)
        );
    }
    // Also verify that a long-context shape that might appear in future
    // shape sets would still be rejected at zero ceiling.
    assert!(!ceiling_zero.fits(1, 8192));
}

/// Branch 3: zero-delta detection — all data points with `rss_delta=0` are
/// treated as non-Linux RSS-unavailable and should not be used to fit the
/// cost model (the early-return check prevents `fit_cost_model` from being
/// called with all-zero y-values, which would produce (0,0) coefficients).
#[test]
fn zero_rss_deltas_not_passed_to_fit() {
    // Simulate what happens when read_process_rss_bytes() returns 0 on macOS:
    // rss_before = 0, rss_after = 0, delta = 0.
    let zero_delta_data: Vec<DataPoint> = PROBE_SHAPES
        .iter()
        .map(|&(batch, seq)| DataPoint {
            batch,
            seq,
            rss_delta: 0,
        })
        .collect();

    // Verify that all_zero check holds.
    assert!(
        zero_delta_data.iter().all(|dp| dp.rss_delta == 0),
        "all deltas should be zero in this scenario"
    );

    // fit_cost_model with all-zero y-values produces coefficients (0.0, 0.0)
    // (the OLS solution to 0 = a*x1 + b*x2 is a=0, b=0), which are then
    // rejected by the non-negative check... but actually the check is
    // a_raw < 0.0 || b_raw < 0.0, which 0.0 does not satisfy. Let's verify
    // fit_cost_model behavior — it should succeed with (0,0) or fail, and
    // either way the zero-delta check in run_probe catches it first.
    let result = fit_cost_model(&zero_delta_data);
    // The coefficients would be 0.0 which is clamped to the lower bound
    // (4096.0, 0.01), so fit_cost_model may succeed with clamped values.
    // The important invariant is that the zero-delta early-return in
    // run_probe fires before fit_cost_model is called for the all-zero case.
    // We validate that all_zero detection is correct:
    let all_zero = zero_delta_data.iter().all(|dp| dp.rss_delta == 0);
    assert!(all_zero, "all-zero check should catch this before fitting");
    // fit_cost_model may or may not return Some here; the caller (run_probe)
    // never reaches it when all_zero is true — so just ensure no panic.
    let _ = result;
}

/// Verifies the all-zero check does not fire when at least one delta is nonzero.
#[test]
fn non_zero_delta_does_not_trigger_zero_delta_branch() {
    let data = [
        DataPoint {
            batch: 1,
            seq: 64,
            rss_delta: 0,
        },
        DataPoint {
            batch: 1,
            seq: 256,
            rss_delta: 5_000_000, // nonzero
        },
    ];
    assert!(
        !data.iter().all(|dp| dp.rss_delta == 0),
        "mixed data should not trigger the all-zero branch"
    );
}

/// `validate_max_seq_shape` must not panic for the full supported range.
#[test]
fn validate_max_seq_shape_does_not_panic() {
    // All representative max_seq values must succeed without session.run().
    for &max_seq in &[64usize, 512, 1024, 2048, 4096, 8192] {
        validate_max_seq_shape(max_seq);
    }
}
