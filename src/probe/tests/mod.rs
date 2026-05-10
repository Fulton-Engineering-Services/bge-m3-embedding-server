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

//! probe tests.
//!
//! - `helpers`: `make_dp` test helper.
//! - `fit_basic`: basic OLS correctness tests for `fit_cost_model`.
//! - `fit_production`: production-scale and rc8 kernel-switch regression tests.
//! - `shapes`: probe corpus, static shape table, and persistent cache tests.
//! - `branches`: branch coverage for empty-data paths and `validate_max_seq_shape`.

mod branches;
mod fit_basic;
mod fit_production;
mod helpers;
mod shapes;
