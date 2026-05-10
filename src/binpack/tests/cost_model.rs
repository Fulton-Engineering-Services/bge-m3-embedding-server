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

use super::super::CostModel;
use super::helpers::model;
use proptest::prelude::*;

// ── cost_model unit tests ──────────────────────────────────────────────

#[test]
fn chunk_cost_pure_linear() {
    let cm = model(100.0, 0.0, usize::MAX);
    // cost(4, 128) = 100 * 4 * 128 + 0 = 51_200
    assert_eq!(cm.chunk_cost(4, 128), 51_200);
}

#[test]
fn chunk_cost_pure_quadratic() {
    let cm = model(0.0, 1.0, usize::MAX);
    // cost(2, 64) = 0 + 1 * 2 * 64 * 64 = 8_192
    assert_eq!(cm.chunk_cost(2, 64), 8_192);
}

#[test]
fn fits_returns_false_when_over_budget() {
    let cm = model(1.0, 0.0, 100);
    assert!(cm.fits(1, 50));
    assert!(!cm.fits(3, 50)); // 150 > 100
}

#[test]
fn conservative_defaults_at_16x512_is_reasonable() {
    let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
    // (16, 512) must fit (this is the old static budget's worst case).
    assert!(cm.fits(16, 512), "conservative model must fit (16, 512)");
    // (1, 8192) must fit (single long text must always be processable).
    assert!(cm.fits(1, 8192), "conservative model must fit (1, 8192)");
}

// ── proptests ─────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn all_indices_present_exactly_once_proptest(
        seqs in prop::collection::vec(1usize..=8192, 0..=200),
        max_bytes in 1usize..=2_000_000_000,
    ) {
        let cm = CostModel::conservative(max_bytes);
        let chunks = super::super::bin_pack(&seqs, &cm);

        let mut found: Vec<usize> = chunks.into_iter().flatten().collect();
        found.sort_unstable();
        let expected: Vec<usize> = (0..seqs.len()).collect();
        prop_assert_eq!(found, expected);
    }

    #[test]
    fn chunks_never_empty(
        seqs in prop::collection::vec(1usize..=512, 1..=100),
        max_bytes in 1usize..=1_000_000,
    ) {
        let cm = CostModel::conservative(max_bytes);
        let chunks = super::super::bin_pack(&seqs, &cm);
        for chunk in &chunks {
            prop_assert!(!chunk.is_empty(), "no chunk should be empty");
        }
    }
}
