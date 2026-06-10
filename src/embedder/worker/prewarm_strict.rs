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

//! Prewarm postcondition readiness gate (`BGE_M3_PREWARM_STRICT`).

use crate::embedder::trt_warmup::{
    prewarm_persistence_postcondition_failed, prewarm_persistence_suspicious_undercount,
};

/// Decides whether a worker should refuse to signal ready after its prewarm
/// sweep based on the on-disk persistence postcondition.
///
/// Returns `true` iff `strict` is `true` AND at least one of the
/// [`prewarm_persistence_postcondition_failed`] /
/// [`prewarm_persistence_suspicious_undercount`] predicates fires for the
/// given `(fresh_compiles, engine_count_after)` snapshot.
///
/// The signature deliberately accepts primitive `usize` values rather than
/// `&PrewarmStats` so the unit tests in `worker/tests/prewarm_strict.rs`
/// stay decoupled from the `trt_warmup::PrewarmStats` struct shape; this
/// also lets the predicate be reused at future call sites (e.g. an admin
/// endpoint that wants to surface the same decision) without dragging in
/// the rest of the prewarm statistics.
///
/// # Strict-mode semantics
///
/// Strict mode (`prewarm_strict=true`) only blocks readiness when
/// `engine_count_after == 0` — i.e. **complete zero-plan failure** where
/// fresh compiles occurred but not a single `.engine` file landed on disk.
/// This is the catastrophic failure mode where every worker hits TRT autotuner
/// OOM mid-build, leaving the cache empty and every subsequent real request
/// returning HTTP 500.
///
/// **Partial undercounts** (e.g. 1 engine persisted out of 16 compiled) do
/// NOT block readiness — workers will serve traffic using the one cached shape
/// and JIT-compile any missing shapes on first request. This is acceptable:
/// partial persistence is most commonly caused by TRT's subgraph fusing
/// (multiple `(batch, seq)` shapes sharing one engine file), not by a hard
/// persistence failure.
///
/// If threshold-based undercount blocking becomes necessary in the future,
/// the [`prewarm_persistence_suspicious_undercount`] branch already has the
/// scaffolding — promote it from WARN to a readiness gate here.
pub(super) fn should_fail_readiness(
    fresh_compiles: usize,
    engine_count_after: usize,
    strict: bool,
) -> bool {
    if !strict {
        return false;
    }
    prewarm_persistence_postcondition_failed(fresh_compiles, engine_count_after)
        || prewarm_persistence_suspicious_undercount(fresh_compiles, engine_count_after)
}
