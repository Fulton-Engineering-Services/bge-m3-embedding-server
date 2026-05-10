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

//! bootstrap tests.
//!
//! - `helpers`: `make_test_state` and `test_cache_dir` helpers.
//! - `router_health`: `GET /health` router tests (not-ready, idle, dead).
//! - `router_dense`: dense endpoint error-code router tests.
//! - `router_sparse_both`: sparse, both, models, and request-id router tests.
//! - `readiness`: `run_readiness_probe` contract tests.
//! - `budget`: `compute_workspace_budget` arithmetic tests.
//! - `permits`: permit-gating and worst-case memory budget invariant.

mod budget;
mod helpers;
mod permits;
mod readiness;
mod router_dense;
mod router_health;
mod router_sparse_both;
