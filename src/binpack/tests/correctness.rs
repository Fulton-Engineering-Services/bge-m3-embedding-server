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

use super::super::{bin_pack, CostModel};
use super::helpers::{linear_model, model};

// ── basic correctness ──────────────────────────────────────────────────

#[test]
fn empty_input_returns_empty() {
    let cm = linear_model(1.0, 1000);
    assert!(bin_pack(&[], &cm).is_empty());
}

#[test]
fn single_text_one_chunk() {
    let cm = linear_model(1.0, 1000);
    let chunks = bin_pack(&[100], &cm);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], vec![0]);
}

#[test]
fn all_texts_fit_in_one_chunk() {
    // Budget: 10 000 bytes; 10 texts × 50 tokens × 1 byte/token = 500 bytes.
    let cm = linear_model(1.0, 10_000);
    let seqs = vec![50usize; 10];
    let chunks = bin_pack(&seqs, &cm);
    assert_eq!(chunks.len(), 1);

    // All 10 original indices must be present.
    let mut found: Vec<usize> = chunks.into_iter().flatten().collect();
    found.sort_unstable();
    assert_eq!(found, (0..10).collect::<Vec<_>>());
}

#[test]
fn texts_split_across_chunks_by_budget() {
    // Budget: 100 tokens × 1 byte; each text is 60 tokens.
    // Two texts = 120 > 100, so each chunk holds exactly 1 text.
    let cm = linear_model(1.0, 100);
    let seqs = vec![60usize; 3];
    let chunks = bin_pack(&seqs, &cm);
    assert_eq!(chunks.len(), 3);
    for chunk in &chunks {
        assert_eq!(chunk.len(), 1);
    }
}

#[test]
fn one_huge_plus_many_tiny() {
    // Budget: 1000 bytes / 1 byte per token.
    // One text at 900 tokens; 100 texts at 5 tokens each.
    let cm = linear_model(1.0, 1000);
    let mut seqs = vec![5usize; 100];
    seqs.push(900); // last index = 100, seq = 900

    let chunks = bin_pack(&seqs, &cm);

    // The huge text should be alone (900 + 5 > 1000 for any pairing with a 5-token text).
    let huge_chunk = chunks.iter().find(|c| c.contains(&100));
    let huge_chunk = huge_chunk.expect("huge text must be in some chunk");
    assert_eq!(huge_chunk.len(), 1, "huge text should be alone");

    // All tiny texts should be packed densely: 5 tokens × 200 = 1000 ≤ 1000.
    // The tiny chunks should each hold 200 texts (floor(1000/5) = 200).
    let total_tiny: usize = chunks
        .iter()
        .filter(|c| !c.contains(&100))
        .map(Vec::len)
        .sum();
    assert_eq!(total_tiny, 100);

    // All original indices appear exactly once.
    let mut all_idx: Vec<usize> = chunks.into_iter().flatten().collect();
    all_idx.sort_unstable();
    assert_eq!(all_idx, (0..101).collect::<Vec<_>>());
}

#[test]
fn all_indices_present_exactly_once() {
    let cm = linear_model(10.0, 5000);
    let seqs = vec![32, 64, 128, 256, 512, 256, 128, 64, 32, 512];
    let chunks = bin_pack(&seqs, &cm);

    let mut found: Vec<usize> = chunks.into_iter().flatten().collect();
    found.sort_unstable();
    assert_eq!(found, (0..seqs.len()).collect::<Vec<_>>());
}

#[test]
fn zero_max_workspace_each_text_solo() {
    let cm = model(1.0, 1.0, 0);
    let seqs = vec![10usize; 5];
    let chunks = bin_pack(&seqs, &cm);
    // Even a single text costs > 0, so every text must be alone.
    assert_eq!(chunks.len(), 5);
    for chunk in &chunks {
        assert_eq!(chunk.len(), 1);
    }
}

#[test]
fn single_text_exceeding_budget_gets_own_chunk() {
    // Budget: 50 bytes. Single text at 100 tokens costs 100 > 50.
    // Must get its own chunk regardless.
    let cm = linear_model(1.0, 50);
    let chunks = bin_pack(&[100], &cm);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], vec![0]);
}

// ── quadratic-dominance test ───────────────────────────────────────────

#[test]
fn quadratic_dominance_long_seqs_get_smaller_chunks() {
    // Conservative defaults: a=16384, b=8.
    // At seq=512: cost per text ≈ 16384*512 + 8*512*512 = 8_388_608 + 2_097_152 ≈ 10.5 MB
    // At seq=8192: cost per text ≈ 16384*8192 + 8*8192*8192 = 134_217_728 + 536_870_912 ≈ 671 MB
    // So with 2 GB budget: 512-token texts ~190 per chunk, 8192-token texts ~2-3 per chunk.
    let cm = CostModel::conservative(CostModel::DEFAULT_MAX_WORKSPACE);

    let short_seqs: Vec<usize> = vec![512; 300];
    let long_seqs: Vec<usize> = vec![8192; 10];

    let short_chunks = bin_pack(&short_seqs, &cm);
    let long_chunks = bin_pack(&long_seqs, &cm);

    // Short chunks should pack many texts; long chunks should be much smaller.
    // cast_precision_loss: chunk counts are small (≤ 300), far within f64
    //   precision; the assertion only checks an order-of-magnitude ratio (5×).
    #[allow(clippy::cast_precision_loss)]
    let avg_short: f64 =
        short_chunks.iter().map(Vec::len).sum::<usize>() as f64 / short_chunks.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let avg_long: f64 =
        long_chunks.iter().map(Vec::len).sum::<usize>() as f64 / long_chunks.len() as f64;

    assert!(
        avg_short > avg_long * 5.0,
        "short-seq chunks ({avg_short:.1} avg) should be much larger than \
             long-seq chunks ({avg_long:.1} avg)"
    );
}
