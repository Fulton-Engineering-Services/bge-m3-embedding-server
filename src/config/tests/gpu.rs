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

//! Tests for GPU device-count and GPU-EP cost-model override behaviour:
//! `BGE_M3_GPU_COUNT`, GPU EP workers field, and the automatic
//! `cost_model_override` applied when a GPU EP is active.

use std::collections::HashMap;

use super::super::Config;
use super::helpers::lookup_from;
use crate::binpack::CostModel;

// --- BGE_M3_GPU_COUNT ---

#[test]
fn gpu_count_defaults_to_at_least_one() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.gpu_count >= 1, "gpu_count must always be ≥ 1");
}

#[test]
fn gpu_count_env_override_respected() {
    let map = HashMap::from([("BGE_M3_GPU_COUNT", "4")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_count, 4);
}

#[test]
fn gpu_count_env_override_eight() {
    let map = HashMap::from([("BGE_M3_GPU_COUNT", "8")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_count, 8);
}

#[test]
fn gpu_count_env_zero_clamps_to_one() {
    let map = HashMap::from([("BGE_M3_GPU_COUNT", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_count, 1);
}

#[test]
fn gpu_count_invalid_value_yields_default() {
    let map = HashMap::from([("BGE_M3_GPU_COUNT", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(
        cfg.gpu_count >= 1,
        "invalid BGE_M3_GPU_COUNT should fall back to at least 1"
    );
}

// --- GPU EP workers clamp via EmbedPool (config stores raw workers) ---

#[test]
fn workers_field_unaffected_by_ep_in_config() {
    // The workers clamp for GPU EPs happens in EmbedPool::spawn (not config).
    // Config stores the raw parsed value so the health endpoint can report it.
    let map = HashMap::from([
        ("BGE_M3_WORKERS", "4"),
        ("BGE_M3_EP", "cuda"),
        ("BGE_M3_GPU_COUNT", "4"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.workers, 4);
    assert_eq!(cfg.gpu_count, 4);
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
