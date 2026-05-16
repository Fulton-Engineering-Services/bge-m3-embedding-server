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

//! Unit tests for the `BGE_M3_PREWARM_STRICT` decision predicate.
//!
//! The predicate is the single point of "should this worker refuse to signal
//! ready?" logic in the prewarm escalation path. Production calls it after
//! every prewarm sweep with the freshly-computed `(fresh_compiles,
//! engine_count_after)` snapshot. These tests fix its behaviour against
//! synthetic inputs so a regression in the escalation block (or in the
//! delegated `prewarm_persistence_*` predicates) is caught here rather than
//! in `CloudWatch`.
//!
//! The function is intentionally typed as `(usize, usize, bool) -> bool` —
//! not `(&PrewarmStats, bool) -> bool` — so the tests do not couple to the
//! `PrewarmStats` struct shape (which is owned by the sibling `trt_warmup`
//! module). Routing around the shared type keeps this test compile-stable
//! even if a sibling change adds or renames `PrewarmStats` fields.

use super::super::should_fail_readiness;

#[test]
fn strict_disabled_never_fails_readiness() {
    // Worst-case persistence-failure shape: many fresh compiles, zero engines
    // on disk. With strict=false we still must NOT fail readiness — preserves
    // pre-fix behaviour for operators who opt out.
    assert!(!should_fail_readiness(10, 0, false));
    assert!(!should_fail_readiness(1, 0, false));
    assert!(!should_fail_readiness(0, 0, false));
    assert!(!should_fail_readiness(5, 5, false));
}

#[test]
fn strict_enabled_healthy_never_fails_readiness() {
    // Healthy case: at least one engine on disk. The catastrophic
    // "fresh > 0 && after == 0" pattern is the only fatal one.
    assert!(!should_fail_readiness(0, 1, true));
    assert!(!should_fail_readiness(5, 1, true));
    assert!(!should_fail_readiness(10, 10, true));
}

#[test]
fn strict_enabled_cache_hit_only_does_not_fail() {
    // Every shape was a cache hit (fresh_compiles == 0). Empty cache directory
    // means the cache hits all came from the coverage-check fast path running
    // against a directory that was already empty — operationally implausible
    // but the predicate should never flag this as fatal because no compile
    // success was claimed.
    assert!(!should_fail_readiness(0, 0, true));
}

#[test]
fn strict_enabled_one_fresh_compile_with_zero_engines_fails() {
    // The 2026-05 codekeeper outage signature: ORT TRT returned Ok(_) from
    // session.run() (counted as fresh_compiles=1) but the .engine file was
    // never persisted to disk (engine_count_after=0). Strict mode must
    // escalate.
    assert!(should_fail_readiness(1, 0, true));
}

#[test]
fn strict_enabled_many_fresh_compiles_with_zero_engines_fails() {
    // Same pattern as the production outage at higher compile volume — the
    // 1215 compile-success / 0 engines case. Must escalate.
    assert!(should_fail_readiness(1215, 0, true));
    assert!(should_fail_readiness(usize::MAX, 0, true));
}

#[test]
fn strict_flag_gates_escalation_at_decision_boundary() {
    // Paired (strict=false, strict=true) calls on the same fatal inputs:
    // the only difference between the two outputs must be the strict flag.
    let fatal_inputs = [(1usize, 0usize), (4, 0), (100, 0)];
    for (fresh, after) in fatal_inputs {
        assert!(
            !should_fail_readiness(fresh, after, false),
            "strict=false must never escalate, even on fatal shape ({fresh}, {after})"
        );
        assert!(
            should_fail_readiness(fresh, after, true),
            "strict=true must escalate on fatal shape ({fresh}, {after})"
        );
    }
}
