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

//! Blocking worker thread, request dispatch, and probe wiring.
//!
//! Submodules:
//! - `config`: per-worker execution policy (`WorkerConfig`).
//! - `guard`: lifecycle guard, JIT guard wiring, inference outcomes.
//! - `trt_retry`: TRT JIT-OOM detection and workspace-halving retry.
//! - `propagation`: engine propagation broadcast drain.
//! - `probe`: probe inference and arena priming helpers.
//! - `prewarm_strict`: prewarm postcondition readiness gate.
//! - `startup`: model load, cache GC, TRT prewarm, readiness signal.
//! - `run`: request loop and idle lifecycle.
//! - `dispatch`: per-request dense/sparse/dual/probe/adaptive-warmup handling.
//! - `logging`: abandoned-request observability.

mod config;
mod dispatch;
mod guard;
mod logging;
mod prewarm_strict;
mod probe;
mod propagation;
mod run;
mod startup;
mod trt_retry;

#[cfg(test)]
mod tests;

pub use config::WorkerConfig;
pub(crate) use probe::probe_run_dense;
pub(in crate::embedder) use run::run_worker;
