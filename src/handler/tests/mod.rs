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

//! handler tests.
//!
//! - `helpers`: `make_state` shared helper.
//! - `validation`: `validate_input` and `check_ready` unit tests.
//! - `health`: health handler tests including tuning-info serialization.
//! - `dense`: `dense_embeddings` handler tests (rejections and happy path).
//! - `sparse`: `sparse_embeddings` and `models` handler tests.
//! - `both`: `both_embeddings` handler tests.
//! - `permits`: permit-gating and worst-case memory budget invariant.

mod both;
mod dense;
mod health;
mod helpers;
mod permits;
mod sparse;
mod validation;
