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

//! `TensorRT` engine pre-warm persistence postconditions.
//!
//! Two diagnostic predicates the worker calls after a prewarm sweep:
//!
//! * [`prewarm_persistence_postcondition_failed`] — **fatal** ERROR signal.
//!   Catches the catastrophic fresh-compiles → 0 engines on disk pattern
//!   that produced the 2026-05 codekeeper outage.
//! * [`prewarm_persistence_suspicious_undercount`] — **non-fatal** WARN
//!   signal. Retained for future extension; currently silent whenever
//!   `engine_count_after > 0` (see inline note).
//!
//! Both are pure functions over `(fresh_compiles, engine_count_after)` so
//! they can be unit-tested without spinning up an ORT session or a
//! filesystem fixture.  The companion fixture-backed tests in `tests.rs`
//! exercise them through the `count_engine_files` snapshot mechanism that
//! the worker uses in production.
//!
//! ## Why `engine_count_after`, not `engine_count_delta`
//!
//! ORT's TRT EP stores one profile-based `.engine` file per fused subgraph
//! that covers all `(batch, seq)` shapes compiled so far via `[min, max]`
//! ranges per input dimension.  When a new shape falls inside the existing
//! range the file is reused (cache hit); when it falls outside the range the
//! EP rewrites the file in-place with an expanded profile.  Either way the
//! on-disk file count stays at 1 after the first compile — `delta == 0` is
//! the normal steady-state, NOT a persistence failure.
//!
//! The only actionable signal is `engine_count_after == 0`: the TRT EP
//! reported `Ok(_)` from `session.run()` yet wrote no engine file at all.
//! That is the exact failure mode from the 2026-05 outage.

/// Decides whether a single worker's prewarm postcondition is violated.
///
/// The postcondition: if at least one shape on this worker reported a
/// **fresh compile** (`succeeded && !cache_hit`) but the on-disk `.engine`
/// file count is still zero, the TRT EP almost certainly emitted `Ok(_)`
/// from `session.run()` without actually persisting the engine plan.  This
/// is the silent-persistence failure mode behind the 2026-05 codekeeper
/// outage (1215 compile-success events / 0 engines on disk).
///
/// Returning `true` should produce an `ERROR` log so operators see the
/// failure in `CloudWatch` immediately.  Non-fresh-compile shards (cache hits
/// only) and shards where at least one engine file exists are accepted.
///
/// The check is keyed on `engine_count_after == 0` rather than
/// `engine_count_delta <= 0`.  ORT's TRT EP rewrites its single
/// profile-based engine file in-place as more shapes are compiled
/// (`delta == 0` at steady state), so a delta-based rule would produce
/// false-positive ERRORs on every shape after the first compile.
#[must_use]
pub(crate) fn prewarm_persistence_postcondition_failed(
    fresh_compiles: usize,
    engine_count_after: usize,
) -> bool {
    fresh_compiles > 0 && engine_count_after == 0
}

/// Minimum `fresh_compiles` count below which the suspicious-undercount
/// check is suppressed.
///
/// Retained for documentation and the test pin; the current implementation
/// of [`prewarm_persistence_suspicious_undercount`] is always silent when
/// `engine_count_after > 0`, making this floor rarely reached.
pub(crate) const SUSPICIOUS_UNDERCOUNT_MIN_FRESH: usize = 2;

/// Decides whether the on-disk `.engine` count is **suspiciously low**
/// relative to the number of fresh compiles, in a way not already caught by
/// [`prewarm_persistence_postcondition_failed`].
///
/// This is a **non-fatal diagnostic signal** intended to back a `WARN`
/// log only — it must never cause the process to exit non-zero.
///
/// **Current behaviour:** always returns `false` when `engine_count_after > 0`.
/// ORT's TRT EP writes one profile-based engine file that covers all compiled
/// shapes; `engine_count_delta == 0` after the first compile is the normal
/// steady-state, not an anomaly.  The only meaningful anomaly is
/// `engine_count_after == 0`, which is already caught by the ERROR predicate
/// above.  Future operators who need a ratio-based WARN for multi-engine
/// workloads can re-introduce it here without changing call sites.
#[must_use]
pub(crate) fn prewarm_persistence_suspicious_undercount(
    fresh_compiles: usize,
    engine_count_after: usize,
) -> bool {
    // Silence entirely when engine files exist on disk.  TRT EP in-place
    // profile extension means delta == 0 is healthy; flagging it would
    // produce false-positive WARNs on every shape after the first compile.
    if engine_count_after > 0 {
        return false;
    }
    // ERROR is already covering this (fresh > 0 && after == 0); no need to
    // double-fire at WARN level on the same evidence.
    if prewarm_persistence_postcondition_failed(fresh_compiles, engine_count_after) {
        return false;
    }
    if fresh_compiles < SUSPICIOUS_UNDERCOUNT_MIN_FRESH {
        return false;
    }
    // Unreachable: (engine_count_after == 0 && fresh_compiles >= MIN_FRESH)
    // implies postcondition_failed == true, handled above.
    false
}
