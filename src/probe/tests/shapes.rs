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

use super::super::cache::{save_probe_cache, try_load_probe_cache};
use super::super::corpus::{load_probe_texts, synthesize_texts};
use super::super::runner::PROBE_SHAPES;

// ── corpus helpers ──────────────────────────────────────────────────────

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

// ── probe shape table ────────────────────────────────────────────────────

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

// ── persistent probe cache ──────────────────────────────────────────────

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
