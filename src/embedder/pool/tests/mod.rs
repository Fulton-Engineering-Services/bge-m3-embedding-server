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

//! `EmbedPool` tests.
//!
//! - `helpers`: shared test helpers (`bad_cache_dir`, `test_cost_model_handle`).
//! - `spawn`: pool spawning and initialization failure propagation.
//! - `channel`: channel-closed error paths and fixture-pool round-trips.
//! - `lifecycle`: worker counts, model RSS, `EmbedStats`, queue depth, and
//!   `median_usize`.
//! - `math_helpers`: pure `normalize_l2`, `sparse_project`, and
//!   `sparse_maxpool` math.
//! - `corpus`: `REPO_REVISION` drift detection and benchmark corpus shape
//!   validation.

mod channel;
mod corpus;
mod helpers;
mod lifecycle;
mod math_helpers;
mod propagation;
mod spawn;
