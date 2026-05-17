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

//! config tests.
//!
//! - `helpers`: `lookup_from` test helper.
//! - `defaults`: default values and clamp behaviour for workers, batches, and seq length.
//! - `budget`: `BGE_M3_DISABLE_AUTO_BUDGET`, `BGE_M3_TOKEN_BUDGET`, explicit cost-model
//!   overrides.
//! - `tuning`: `BGE_M3_IDLE_TIMEOUT_SECS`, `BGE_M3_MEMORY_SAFETY_FACTOR`, `BGE_M3_MODEL`,
//!   `BGE_M3_HEARTBEAT_SECS`.
//! - `trt_shapes`: `BGE_M3_TRT_WARMUP_SHAPES` parsing and default grid invariants.
//! - `ep`: `BGE_M3_WARMUP_ONLY`, `BGE_M3_EP`, `BGE_M3_GPU_VRAM_BUDGET_BYTES`,
//!   `BGE_M3_TRT_MAX_WORKSPACE_BYTES`, `BGE_M3_GPU_MEM_LIMIT_BYTES`.
//! - `adaptive`: `BGE_M3_ADAPTIVE_WARMUP_*` and `BGE_M3_ENGINE_PROPAGATION_ENABLED`.
//! - `gpu`: `BGE_M3_GPU_COUNT`, workers field, and GPU-EP cost-model override.
//! - `tls`: `BGE_M3_TLS_CERT_PATH`, `BGE_M3_TLS_KEY_PATH`, and `Config::validate` half-config guard.

mod adaptive;
mod budget;
mod defaults;
mod ep;
mod gpu;
mod helpers;
mod tls;
mod trt_shapes;
mod tuning;
