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

//! Sparse embedding equivalence cases.
//!
//! Reserved for future sparse-only equivalence tests. The current
//! `equivalence_all_seq_lengths` (in `dense.rs`) only checks dense cosine
//! similarity; a parallel sparse check would compare BGE-M3 SPLADE-style
//! outputs against the reference `reference_sparse_seq_*.json` fixtures.
//!
//! Until such tests are added, this module is intentionally empty so the
//! `tests/equivalence/main.rs` harness can declare `mod sparse;` and the
//! file layout matches the source-layout-refactor plan.
