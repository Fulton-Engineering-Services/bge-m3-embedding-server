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

use super::super::fit::{fit_cost_model, DataPoint};
use super::helpers::make_dp;

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
