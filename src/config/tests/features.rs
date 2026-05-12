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

use std::collections::HashMap;

use super::super::{parse_trt_warmup_shapes, Config, EpSelection, ModelVariant};
use super::helpers::lookup_from;
use crate::binpack::CostModel;

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

// --- BGE_M3_TRT_WARMUP_SHAPES ---

#[test]
fn trt_warmup_shapes_none_yields_defaults() {
    assert_eq!(
        parse_trt_warmup_shapes(None),
        vec![(1, 128), (1, 512), (1, 2048), (1, 8192)],
    );
}

#[test]
fn trt_warmup_shapes_empty_string_yields_defaults() {
    assert_eq!(
        parse_trt_warmup_shapes(Some(String::new())),
        vec![(1, 128), (1, 512), (1, 2048), (1, 8192)],
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
        vec![(1, 128), (1, 512), (1, 2048), (1, 8192)],
    );
}

#[test]
fn trt_warmup_shapes_config_field_defaults_without_env() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(
        cfg.trt_warmup_shapes,
        vec![(1, 128), (1, 512), (1, 2048), (1, 8192)],
    );
}

#[test]
fn trt_warmup_shapes_config_field_set_from_env() {
    let map = HashMap::from([("BGE_M3_TRT_WARMUP_SHAPES", "4x256,1x8192")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_warmup_shapes, vec![(4, 256), (1, 8192)]);
}

// --- BGE_M3_EP (EpSelection) ---

#[test]
fn ep_defaults_to_cpu_when_unset() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.ep, EpSelection::Cpu);
}

#[test]
fn ep_cuda_when_set() {
    let map = HashMap::from([("BGE_M3_EP", "cuda")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.ep, EpSelection::Cuda);
}

#[test]
fn ep_tensorrt_when_set() {
    let map = HashMap::from([("BGE_M3_EP", "tensorrt")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.ep, EpSelection::TensorRt);
}

#[test]
fn ep_unknown_value_falls_back_to_cpu() {
    let map = HashMap::from([("BGE_M3_EP", "unknown_value")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.ep, EpSelection::Cpu);
}

#[test]
fn ep_selection_display() {
    assert_eq!(EpSelection::Cpu.to_string(), "cpu");
    assert_eq!(EpSelection::Cuda.to_string(), "cuda");
    assert_eq!(EpSelection::TensorRt.to_string(), "tensorrt");
}

// --- BGE_M3_GPU_VRAM_BUDGET_BYTES ---

#[test]
fn gpu_vram_budget_set_to_valid_value() {
    let map = HashMap::from([("BGE_M3_GPU_VRAM_BUDGET_BYTES", "10737418240")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_vram_budget_bytes, Some(10_737_418_240));
}

#[test]
fn gpu_vram_budget_unset_yields_none() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_vram_budget_bytes, None);
}

#[test]
fn gpu_vram_budget_invalid_value_yields_none() {
    let map = HashMap::from([("BGE_M3_GPU_VRAM_BUDGET_BYTES", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_vram_budget_bytes, None);
}

// --- GPU EP cost-model override ---

#[test]
fn gpu_ep_forces_cost_model_override_with_default_vram_budget() {
    // When a GPU EP is active and BGE_M3_GPU_VRAM_BUDGET_BYTES is unset,
    // cost_model_override is set to conservative(10 GiB) — the host-RAM
    // probe is bypassed and the default VRAM ceiling drives bin-packing.
    let map = HashMap::from([("BGE_M3_EP", "cuda")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg
        .cost_model_override
        .expect("GPU EP must set cost_model_override");
    let expected_vram: usize = 10 * 1024 * 1024 * 1024;
    assert_eq!(
        cm.max_workspace_bytes, expected_vram,
        "default VRAM budget should be 10 GiB"
    );
    assert!((cm.a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON);
    assert!((cm.b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON);
}

#[test]
fn gpu_ep_forces_cost_model_override_with_explicit_vram_budget() {
    // When BGE_M3_GPU_VRAM_BUDGET_BYTES is set alongside a GPU EP, the
    // explicit budget replaces the 10 GiB default.
    const EIGHT_GIB: usize = 8 * 1024 * 1024 * 1024;
    let map = HashMap::from([
        ("BGE_M3_EP", "tensorrt"),
        ("BGE_M3_GPU_VRAM_BUDGET_BYTES", "8589934592"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg
        .cost_model_override
        .expect("GPU EP must set cost_model_override");
    assert_eq!(
        cm.max_workspace_bytes, EIGHT_GIB,
        "explicit VRAM budget should override the 10 GiB default"
    );
}

#[test]
fn cpu_ep_does_not_force_cost_model_override() {
    // CPU EP leaves cost_model_override as resolved by the non-GPU path
    // (None → probe runs). Confirming CPU does NOT trigger the GPU override.
    let map = HashMap::from([("BGE_M3_EP", "cpu")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(
        cfg.cost_model_override.is_none(),
        "CPU EP should not override cost model — probe must run"
    );
}
