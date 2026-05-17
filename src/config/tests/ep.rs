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

//! Tests for execution-provider and warmup env vars: `BGE_M3_WARMUP_ONLY`,
//! `BGE_M3_EP`, `BGE_M3_GPU_VRAM_BUDGET_BYTES`,
//! `BGE_M3_TRT_MAX_WORKSPACE_BYTES`, and `BGE_M3_GPU_MEM_LIMIT_BYTES`.

use std::collections::HashMap;

use super::super::{Config, EpSelection};
use super::helpers::lookup_from;

// --- BGE_M3_WARMUP_ONLY ---

#[test]
fn warmup_only_defaults_to_false() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(!cfg.warmup_only);
}

#[test]
fn warmup_only_true_when_set_to_1() {
    let map = HashMap::from([("BGE_M3_WARMUP_ONLY", "1"), ("BGE_M3_EP", "tensorrt")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.warmup_only);
}

#[test]
fn warmup_only_true_when_set_to_true() {
    let map = HashMap::from([("BGE_M3_WARMUP_ONLY", "true"), ("BGE_M3_EP", "tensorrt")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.warmup_only);
}

#[test]
fn warmup_only_false_when_set_to_0() {
    let map = HashMap::from([("BGE_M3_WARMUP_ONLY", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(!cfg.warmup_only);
}

#[test]
fn warmup_only_with_non_trt_ep_still_parses_true() {
    // The WARN path: warmup_only=true with a CPU EP is allowed (exits 0
    // without compiling anything).  Config must still reflect the value.
    let map = HashMap::from([("BGE_M3_WARMUP_ONLY", "1")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(cfg.warmup_only);
    assert_eq!(cfg.ep, EpSelection::Cpu);
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

// --- BGE_M3_TRT_MAX_WORKSPACE_BYTES ---

#[test]
fn trt_max_workspace_bytes_set_to_valid_value() {
    let map = HashMap::from([("BGE_M3_TRT_MAX_WORKSPACE_BYTES", "4294967296")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_max_workspace_bytes, Some(4_294_967_296));
}

#[test]
fn trt_max_workspace_bytes_unset_yields_none() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_max_workspace_bytes, None);
}

#[test]
fn trt_max_workspace_bytes_invalid_yields_none() {
    let map = HashMap::from([("BGE_M3_TRT_MAX_WORKSPACE_BYTES", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.trt_max_workspace_bytes, None);
}

// --- BGE_M3_GPU_MEM_LIMIT_BYTES ---

#[test]
fn gpu_mem_limit_bytes_set() {
    let map = HashMap::from([("BGE_M3_GPU_MEM_LIMIT_BYTES", "8589934592")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_mem_limit_bytes, Some(8_589_934_592));
}

#[test]
fn gpu_mem_limit_bytes_unset_yields_none() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_mem_limit_bytes, None);
}

#[test]
fn gpu_mem_limit_bytes_invalid_yields_none() {
    let map = HashMap::from([("BGE_M3_GPU_MEM_LIMIT_BYTES", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.gpu_mem_limit_bytes, None);
}
