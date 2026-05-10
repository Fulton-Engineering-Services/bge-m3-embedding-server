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

//! Startup memory probe and cost-model coefficient fitter.
//!
//! Submodules:
//! - `runner`: drives the `(batch, seq)` shape sweep on the leader worker
//!   (`run_probe`, `PROBE_SHAPES`, the absolute-RSS guard, the arena warm-up).
//! - `cache`: persistent EFS-backed cache for fitted probe coefficients
//!   (`try_load_probe_cache`, `save_probe_cache`, `ProbeCache`).
//! - `fit`: ordinary least-squares fitter for the quadratic cost model
//!   (`fit_cost_model`, `DataPoint`).
//! - `corpus`: helpers that synthesize probe texts from the curated corpus.
//! - `validate`: tokenizer + ndarray shape check at the configured `max_seq`
//!   (no `session.run()`).

mod cache;
mod corpus;
mod fit;
mod runner;
mod validate;

pub(crate) use cache::{save_probe_cache, try_load_probe_cache};
pub(crate) use runner::run_probe;

#[cfg(test)]
mod tests;
