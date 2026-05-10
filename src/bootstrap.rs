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

//! Server-startup orchestration: routing, workspace budget, readiness probe,
//! and the background probe task that fits the cost model on first start.
//!
//! Submodules:
//! - `router`: axum `Router` construction + tracing/request-id layers.
//! - `budget`: pure workspace-budget arithmetic (`compute_workspace_budget`).
//! - `readiness`: the foreground readiness probe (`run_readiness_probe`,
//!   `run_readiness_checks_and_open`).
//! - `probe_task`: the background probe task (`spawn_probe_task`) used when
//!   the cost model has not been overridden and no EFS cache hit was found.

mod budget;
mod probe_task;
mod readiness;
mod router;

pub use readiness::run_readiness_probe;
pub use router::build_router;

#[cfg(test)]
mod tests;
