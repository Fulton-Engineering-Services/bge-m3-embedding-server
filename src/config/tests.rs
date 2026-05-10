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

use super::*;
use std::collections::HashMap;

fn lookup_from<'a>(map: &'a HashMap<&'a str, &'a str>) -> impl Fn(&str) -> Option<String> + 'a {
    move |key| map.get(key).map(|&v| v.to_string())
}

#[test]
fn defaults_without_env_vars() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));

    assert_eq!(cfg.cache_dir, "/cache");
    assert_eq!(cfg.bind_addr, "0.0.0.0:8081");
    assert_eq!(cfg.workers, 2);
    assert_eq!(cfg.intra_threads, 1);
    assert_eq!(cfg.max_batch, 256);
    assert_eq!(cfg.max_seq_length, MODEL_MAX_SEQ);
    assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(300)));
    assert_eq!(cfg.model_variant, ModelVariant::Fp16);
    assert!((cfg.memory_safety_factor - 0.7).abs() < 1e-9);
    assert!(
        cfg.cost_model_override.is_none(),
        "probe should run by default"
    );
    assert_eq!(cfg.heartbeat_secs, 60);
}

#[test]
fn workers_clamps_to_minimum_1() {
    let map = HashMap::from([("BGE_M3_WORKERS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.workers, 1);
}

#[test]
fn intra_threads_defaults_to_1() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 1);
}

#[test]
fn intra_threads_custom_value() {
    let map = HashMap::from([("BGE_M3_INTRA_THREADS", "4")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 4);
}

#[test]
fn intra_threads_clamps_to_minimum_1() {
    let map = HashMap::from([("BGE_M3_INTRA_THREADS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 1);
}

#[test]
fn intra_threads_invalid_falls_back_to_default() {
    let map = HashMap::from([("BGE_M3_INTRA_THREADS", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 1);
}

#[test]
fn max_batch_clamps_to_minimum_1() {
    let map = HashMap::from([("BGE_M3_MAX_BATCH", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_batch, 1);
}

#[test]
fn max_seq_length_default_is_8192() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, 8192);
}

#[test]
fn max_seq_length_custom() {
    let map = HashMap::from([("BGE_M3_MAX_SEQ_LENGTH", "2048")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, 2048);
}

#[test]
fn max_seq_length_clamps_out_of_range() {
    // Over max
    let map = HashMap::from([("BGE_M3_MAX_SEQ_LENGTH", "99999")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, MODEL_MAX_SEQ);

    // Zero
    let map = HashMap::from([("BGE_M3_MAX_SEQ_LENGTH", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, MODEL_MAX_SEQ);
}

#[test]
fn disable_auto_budget_yields_conservative_model() {
    let map = HashMap::from([("BGE_M3_DISABLE_AUTO_BUDGET", "1")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg.cost_model_override.expect("override must be set");
    assert!((cm.a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON);
    assert!((cm.b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON);
    assert_eq!(cm.max_workspace_bytes, CostModel::DEFAULT_MAX_WORKSPACE);
}

#[test]
fn token_budget_translates_to_cost_model() {
    // With token_budget=8192 and conservative coefficients the workspace
    // must be a positive number.
    let map = HashMap::from([("BGE_M3_TOKEN_BUDGET", "8192")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg.cost_model_override.expect("override must be set");
    assert!(cm.max_workspace_bytes > 0);
    assert!((cm.a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON);
    assert!((cm.b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON);
}

#[test]
fn explicit_cost_model_override() {
    let map = HashMap::from([
        ("BGE_M3_COST_MODEL_A", "20000.0"),
        ("BGE_M3_COST_MODEL_B", "5.0"),
        ("BGE_M3_AVAILABLE_MEMORY_BYTES", "1073741824"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg.cost_model_override.expect("override must be set");
    assert!((cm.a - 20_000.0).abs() < 1e-9);
    assert!((cm.b - 5.0).abs() < 1e-9);
    assert_eq!(cm.max_workspace_bytes, 1_073_741_824);
}

#[test]
fn custom_values_are_applied() {
    let map = HashMap::from([
        ("BGE_M3_CACHE_DIR", "/tmp/models"),
        ("BGE_M3_BIND", "127.0.0.1:9090"),
        ("BGE_M3_WORKERS", "4"),
        ("BGE_M3_MAX_BATCH", "128"),
        ("BGE_M3_IDLE_TIMEOUT_SECS", "600"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));

    assert_eq!(cfg.cache_dir, "/tmp/models");
    assert_eq!(cfg.bind_addr, "127.0.0.1:9090");
    assert_eq!(cfg.workers, 4);
    assert_eq!(cfg.max_batch, 128);
    assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(600)));
}

#[test]
fn idle_timeout_disabled_when_zero() {
    let map = HashMap::from([("BGE_M3_IDLE_TIMEOUT_SECS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.idle_timeout, None);
}

#[test]
fn memory_safety_factor_clamped() {
    let map_low = HashMap::from([("BGE_M3_MEMORY_SAFETY_FACTOR", "0.0")]);
    let cfg_low = Config::from_lookup(lookup_from(&map_low));
    assert!((cfg_low.memory_safety_factor - 0.1).abs() < 1e-9);

    let map_high = HashMap::from([("BGE_M3_MEMORY_SAFETY_FACTOR", "2.0")]);
    let cfg_high = Config::from_lookup(lookup_from(&map_high));
    assert!((cfg_high.memory_safety_factor - 1.0).abs() < 1e-9);
}

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
