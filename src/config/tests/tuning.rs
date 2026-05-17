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

//! Tests for runtime-tuning env vars: `BGE_M3_IDLE_TIMEOUT_SECS`,
//! `BGE_M3_MEMORY_SAFETY_FACTOR`, `BGE_M3_MODEL`, and
//! `BGE_M3_HEARTBEAT_SECS`.

use std::collections::HashMap;

use super::super::{Config, ModelVariant};
use super::helpers::lookup_from;

// --- BGE_M3_IDLE_TIMEOUT_SECS ---

#[test]
fn idle_timeout_disabled_when_zero() {
    let map = HashMap::from([("BGE_M3_IDLE_TIMEOUT_SECS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.idle_timeout, None);
}

// --- BGE_M3_MEMORY_SAFETY_FACTOR ---

#[test]
fn memory_safety_factor_clamped() {
    let map_low = HashMap::from([("BGE_M3_MEMORY_SAFETY_FACTOR", "0.0")]);
    let cfg_low = Config::from_lookup(lookup_from(&map_low));
    assert!((cfg_low.memory_safety_factor - 0.1).abs() < 1e-9);

    let map_high = HashMap::from([("BGE_M3_MEMORY_SAFETY_FACTOR", "2.0")]);
    let cfg_high = Config::from_lookup(lookup_from(&map_high));
    assert!((cfg_high.memory_safety_factor - 1.0).abs() < 1e-9);
}

// --- BGE_M3_MODEL ---

#[test]
fn model_variant_defaults_to_fp16() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.model_variant, ModelVariant::Fp16);
}

#[test]
fn model_variant_fp32_when_set() {
    let map = HashMap::from([("BGE_M3_MODEL", "fp32")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.model_variant, ModelVariant::Fp32);
}

#[test]
fn model_variant_int8_when_set() {
    let map = HashMap::from([("BGE_M3_MODEL", "int8")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.model_variant, ModelVariant::Int8);
}

#[test]
fn model_variant_unknown_value_falls_back_to_fp16() {
    let map = HashMap::from([("BGE_M3_MODEL", "invalid")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.model_variant, ModelVariant::Fp16);
}

#[test]
fn model_variant_display() {
    assert_eq!(ModelVariant::Fp32.to_string(), "fp32");
    assert_eq!(ModelVariant::Fp16.to_string(), "fp16");
    assert_eq!(ModelVariant::Int8.to_string(), "int8");
}

// --- BGE_M3_HEARTBEAT_SECS ---

#[test]
fn heartbeat_secs_defaults_to_60() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.heartbeat_secs, 60);
}

#[test]
fn heartbeat_secs_custom_value() {
    let map = HashMap::from([("BGE_M3_HEARTBEAT_SECS", "120")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.heartbeat_secs, 120);
}

#[test]
fn heartbeat_secs_disabled_when_zero() {
    let map = HashMap::from([("BGE_M3_HEARTBEAT_SECS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.heartbeat_secs, 0);
}

#[test]
fn heartbeat_secs_invalid_falls_back_to_default() {
    let map = HashMap::from([("BGE_M3_HEARTBEAT_SECS", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(
        cfg.heartbeat_secs, 60,
        "invalid value should fall back to default"
    );
}
