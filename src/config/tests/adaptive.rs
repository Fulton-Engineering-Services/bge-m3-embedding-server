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

//! Tests for adaptive-warmup and engine-propagation env vars:
//! `BGE_M3_ADAPTIVE_WARMUP_ENABLED`, `BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS`,
//! `BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR`, and
//! `BGE_M3_ENGINE_PROPAGATION_ENABLED`.

use std::collections::HashMap;

use super::super::Config;
use super::helpers::lookup_from;

// --- BGE_M3_ADAPTIVE_WARMUP_ENABLED ---

#[test]
fn adaptive_warmup_enabled_default_is_false() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(!cfg.adaptive_warmup_enabled);
}

#[test]
fn adaptive_warmup_enabled_set_to_1() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_ENABLED", "1")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.adaptive_warmup_enabled);
}

#[test]
fn adaptive_warmup_enabled_set_to_true_string() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_ENABLED", "true")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.adaptive_warmup_enabled);
}

#[test]
fn adaptive_warmup_enabled_set_to_yes_string() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_ENABLED", "yes")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.adaptive_warmup_enabled);
}

// --- BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS ---

#[test]
fn adaptive_warmup_quiet_secs_default_is_3() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.adaptive_warmup_quiet_secs, 3);
}

#[test]
fn adaptive_warmup_quiet_secs_set() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS", "10")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.adaptive_warmup_quiet_secs, 10);
}

#[test]
fn adaptive_warmup_quiet_secs_invalid_yields_default_3() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.adaptive_warmup_quiet_secs, 3);
}

// --- BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR ---

#[test]
fn adaptive_warmup_max_shapes_per_hour_default_is_12() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.adaptive_warmup_max_shapes_per_hour, 12);
}

#[test]
fn adaptive_warmup_max_shapes_per_hour_set() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR", "24")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.adaptive_warmup_max_shapes_per_hour, 24);
}

#[test]
fn adaptive_warmup_max_shapes_per_hour_invalid_yields_default_12() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.adaptive_warmup_max_shapes_per_hour, 12);
}

// --- BGE_M3_ENGINE_PROPAGATION_ENABLED ---

#[test]
fn engine_propagation_enabled_defaults_to_adaptive_warmup_when_unset_true() {
    let map = HashMap::from([("BGE_M3_ADAPTIVE_WARMUP_ENABLED", "1")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(
        cfg.adaptive_warmup_enabled,
        "adaptive_warmup_enabled must be true for this test to be meaningful"
    );
    assert!(
        cfg.engine_propagation_enabled,
        "engine_propagation_enabled must default to true when adaptive_warmup_enabled is true"
    );
}

#[test]
fn engine_propagation_enabled_defaults_to_false_when_adaptive_disabled() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(
        !cfg.adaptive_warmup_enabled,
        "adaptive_warmup_enabled must be false for this test to be meaningful"
    );
    assert!(
        !cfg.engine_propagation_enabled,
        "engine_propagation_enabled must default to false when adaptive_warmup_enabled is false"
    );
}

#[test]
fn engine_propagation_enabled_can_be_explicitly_disabled() {
    let map = HashMap::from([
        ("BGE_M3_ADAPTIVE_WARMUP_ENABLED", "1"),
        ("BGE_M3_ENGINE_PROPAGATION_ENABLED", "0"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(
        cfg.adaptive_warmup_enabled,
        "adaptive_warmup_enabled must be true for this test to be meaningful"
    );
    assert!(
        !cfg.engine_propagation_enabled,
        "engine_propagation_enabled=0 must override the adaptive_warmup_enabled default"
    );
}
