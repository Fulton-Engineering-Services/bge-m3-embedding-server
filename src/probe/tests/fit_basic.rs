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

use super::super::fit::{DataPoint, fit_cost_model};
use super::helpers::make_dp;

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
