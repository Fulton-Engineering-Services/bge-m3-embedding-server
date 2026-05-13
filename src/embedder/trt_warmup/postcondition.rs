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
//!   Catches the catastrophic 1215 fresh-compiles → 0 engines on disk
//!   pattern that produced the 2026-05 codekeeper outage.
//! * [`prewarm_persistence_suspicious_undercount`] — **non-fatal** WARN
//!   signal. Catches large engines-vs-compiles ratio anomalies (e.g.
//!   1215 compiles / 5 engines) that the loose ERROR rule does not flag.
//!
//! Both are pure functions over `(fresh_compiles, engine_count_delta)`
//! pairs so they can be unit-tested without spinning up an ORT session
//! or a filesystem fixture.  The companion fixture-backed tests in
//! `tests.rs` exercise them through the `count_engine_files` snapshot
//! mechanism that the worker uses in production.

/// Decides whether a single worker's prewarm postcondition is violated.
///
/// The postcondition: if at least one shape on this worker reported a
/// **fresh compile** (`succeeded && !cache_hit`) but the on-disk `.engine`
/// file count did not increase, the TRT EP almost certainly emitted
/// `Ok(_)` from `session.run()` without actually persisting the engine
/// plan. This is the silent-persistence failure mode behind the 2026-05
/// codekeeper outage (400 cache-empty events / 1215 compile-success
/// events / 0 cache-found events / cache directory empty on disk).
///
/// Returning `true` should produce an `ERROR` log so operators see the
/// failure in `CloudWatch` immediately rather than discovering it later
/// from cold-cache compile times. Non-fresh-compile shards (cache hits
/// only) and shards that produced positive deltas are accepted.
///
/// This rule is **deliberately loose**: ORT's TRT EP names `.engine` files
/// by fused-subgraph identity + precision + GPU SM
/// (`TensorrtExecutionProvider_TRTKernel_<hash>_<precision>_sm<XX>.engine`),
/// not by `(batch, seq)`.  Many shapes legitimately share one engine file,
/// and profile-range extension can rewrite an existing engine in place — so
/// `engine_count_delta < fresh_compiles` is not on its own a defect.  The
/// "any positive delta passes" rule is intentionally tolerant of those
/// legitimate undercounts while still catching the 1215 → 0 catastrophic
/// case.  See [`prewarm_persistence_suspicious_undercount`] for a separate,
/// non-fatal WARN that flags large undercounts where the ERROR rule passes
/// but the ratio is still anomalous.
#[must_use]
pub(crate) fn prewarm_persistence_postcondition_failed(
    fresh_compiles: usize,
    engine_count_delta: i64,
) -> bool {
    fresh_compiles > 0 && engine_count_delta <= 0
}

/// Denominator for the suspicious-undercount heuristic in
/// [`prewarm_persistence_suspicious_undercount`]: an `engine_count_delta`
/// less than `fresh_compiles / UNDERCOUNT_RATIO_DIVISOR` is treated as
/// suspiciously low and produces a non-fatal WARN.
///
/// `2` is a conservative choice — it permits a 1:2 ratio (engine reuse
/// across multiple compiles) before raising any signal, so a worker that
/// compiles 16 shapes but produces 8 engine files is silent, while a
/// worker that compiles 1215 shapes but produces 5 engine files (the
/// kind of anomaly worth investigating) trips the warning.
pub(crate) const UNDERCOUNT_RATIO_DIVISOR: i64 = 2;

/// Minimum `fresh_compiles` count below which the suspicious-undercount
/// check is suppressed.
///
/// Tiny shards (one or two fresh compiles) generate too much noise relative
/// to their signal — a single-shape worker that reuses an existing engine
/// plan can show `fresh=1, delta=0` and be perfectly healthy (the loose
/// postcondition handles `delta=0` separately; here we want to avoid even
/// the WARN-level chatter).  At `fresh ≥ 2` the ratio test becomes
/// meaningful.
pub(crate) const SUSPICIOUS_UNDERCOUNT_MIN_FRESH: usize = 2;

/// Decides whether the on-disk `.engine` count is **suspiciously low**
/// relative to the number of fresh compiles reported by `session.run()`,
/// in a way that is NOT already caught by
/// [`prewarm_persistence_postcondition_failed`].
///
/// This is a **non-fatal diagnostic signal** intended to back a `WARN`
/// log only — it must never cause the process to exit non-zero.  ORT's TRT
/// EP can legitimately produce one `.engine` file per fused subgraph and
/// reuse it across many input shapes (the engine plan is keyed by
/// fused-subgraph identity + precision + GPU SM, not by `(batch, seq)`).
/// On a healthy worker compiling 4 distinct shapes you may see `delta = 1`
/// because the engine was rebuilt three times with expanding profile
/// ranges before settling on the final one.
///
/// The heuristic is "fewer than half as many engines as compiles":
///
/// ```text
/// engine_count_delta * UNDERCOUNT_RATIO_DIVISOR  <  fresh_compiles
/// ```
///
/// Suppressed when:
/// - `fresh_compiles < SUSPICIOUS_UNDERCOUNT_MIN_FRESH` — too noisy for tiny shards
/// - [`prewarm_persistence_postcondition_failed`] already returns `true`
///   — that path emits a louder `ERROR`, no need to double-fire on the
///   same evidence
///
/// Returning `true` should produce a `WARN` log with structured fields so
/// operators can investigate in `CloudWatch`.  It does **not** affect process
/// exit code, the `/health` endpoint, or any production logic.
#[must_use]
pub(crate) fn prewarm_persistence_suspicious_undercount(
    fresh_compiles: usize,
    engine_count_delta: i64,
) -> bool {
    if prewarm_persistence_postcondition_failed(fresh_compiles, engine_count_delta) {
        return false;
    }
    if fresh_compiles < SUSPICIOUS_UNDERCOUNT_MIN_FRESH {
        return false;
    }
    let fresh_i64 = i64::try_from(fresh_compiles).unwrap_or(i64::MAX);
    engine_count_delta.saturating_mul(UNDERCOUNT_RATIO_DIVISOR) < fresh_i64
}
