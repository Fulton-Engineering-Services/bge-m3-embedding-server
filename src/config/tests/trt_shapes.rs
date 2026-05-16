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

//! Tests for `BGE_M3_TRT_WARMUP_SHAPES` parsing and the default 16-shape
//! warmup grid ordering invariants.

use std::collections::HashMap;

use super::super::{parse_trt_warmup_shapes, Config};
use super::helpers::lookup_from;

// --- BGE_M3_TRT_WARMUP_SHAPES ---

/// The default warmup grid is `{1, 4, 16, 32} × {128, 512, 2048, 8192}` in
/// batch-major order so the smallest batches (which dominate real router
/// traffic) compile first.
fn default_warmup_grid() -> Vec<(usize, usize)> {
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

#[test]
fn trt_warmup_shapes_none_yields_defaults() {
    assert_eq!(parse_trt_warmup_shapes(None), default_warmup_grid());
}

#[test]
fn trt_warmup_shapes_empty_string_yields_defaults() {
    assert_eq!(
        parse_trt_warmup_shapes(Some(String::new())),
        default_warmup_grid(),
    );
}

#[test]
fn trt_warmup_shapes_valid_tokens_parsed() {
    assert_eq!(
        parse_trt_warmup_shapes(Some("1x128,1x512".to_string())),
        vec![(1, 128), (1, 512)],
    );
}

#[test]
fn trt_warmup_shapes_invalid_token_skipped() {
    // "bad" is not a valid BxL token and should be silently skipped.
    assert_eq!(
        parse_trt_warmup_shapes(Some("1x128,bad,1x512".to_string())),
        vec![(1, 128), (1, 512)],
    );
}

#[test]
fn trt_warmup_shapes_all_invalid_yields_defaults() {
    assert_eq!(
        parse_trt_warmup_shapes(Some("bad,also_bad,nope".to_string())),
        default_warmup_grid(),
    );
}

#[test]
fn trt_warmup_shapes_config_field_defaults_without_env() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_warmup_shapes, default_warmup_grid());
}

#[test]
fn trt_warmup_shapes_config_field_set_from_env() {
    let map = HashMap::from([("BGE_M3_TRT_WARMUP_SHAPES", "4x256,1x8192")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_warmup_shapes, vec![(4, 256), (1, 8192)]);
}

#[test]
fn trt_warmup_shapes_default_grid_is_batch_major() {
    // Default grid: outer loop is batch, inner loop is sequence length.
    // The first four entries should all have batch=1; the last four should
    // all have batch=32. This protects the "smallest batches compile first"
    // ordering against accidental reshuffles.
    let defaults = default_warmup_grid();
    assert!(
        defaults[..4].iter().all(|(b, _)| *b == 1),
        "first four entries should be batch=1"
    );
    assert!(
        defaults[12..].iter().all(|(b, _)| *b == 32),
        "last four entries should be batch=32"
    );
    // Within batch=1 the sequence dimension grows monotonically.
    assert_eq!(defaults[..4], [(1, 128), (1, 512), (1, 2048), (1, 8192)]);
}

#[test]
fn trt_warmup_shapes_small_grid_override_works_for_local_dev() {
    // Operators running locally can collapse the grid to a single cheap shape
    // (e.g. `BGE_M3_TRT_WARMUP_SHAPES=1x128`) — that override path must keep
    // working so cold-start stays tractable on workstations.
    let map = HashMap::from([("BGE_M3_TRT_WARMUP_SHAPES", "1x128")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_warmup_shapes, vec![(1, 128)]);
}
