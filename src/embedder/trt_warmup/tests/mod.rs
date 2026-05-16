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

//! Unit tests for the `trt_warmup` module.
//!
//! - `coverage`: `coverage_check_shapes`, `shard_shapes`, compile-time
//!   threshold assertions, and coverage-check correctness proofs.
//! - `postcondition`: `prewarm_persistence_postcondition_failed` and
//!   `prewarm_persistence_suspicious_undercount` predicate contract tests.
//! - `engine_count`: filesystem-backed `count_engine_files` snapshot
//!   contract tests.

mod coverage;
mod engine_count;
mod postcondition;
