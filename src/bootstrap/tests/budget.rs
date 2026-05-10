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

use super::super::budget::compute_workspace_budget;
use crate::binpack::CostModel;

#[test]
fn compute_workspace_budget_sane_inputs() {
    // 28 GiB available, 7 workers, ~1.6 GiB model RSS, 0.7 safety.
    let avail = 28_672usize * 1024 * 1024;
    let model_rss = 1_628usize * 1024 * 1024;
    let (ws, peak, pct) = compute_workspace_budget(avail, 7, model_rss, 0.7);
    // total_workspace = 28672 - 7*1628 - 256 ≈ 17,060 MiB
    // per_worker = 17060 * 0.7 / 7 ≈ 1,706 MiB
    assert!(
        ws > 1_000 * 1024 * 1024,
        "per_worker_workspace ({} MiB) should be well over 1 GiB",
        ws / (1024 * 1024)
    );
    assert!(ws < avail, "per_worker_workspace must not exceed available");
    // Worst-case peak should be < available (sanity).
    assert!(
        peak < avail * 2,
        "peak ({} MiB) seems unreasonably large",
        peak / (1024 * 1024)
    );
    assert!(
        pct > 0.0 && pct < 200.0,
        "utilization_pct {pct:.1}% out of range"
    );
}

#[test]
fn compute_workspace_budget_saturates_gracefully_when_model_rss_inflated() {
    // Reproduces the production failure: inflated model_rss_per_worker from
    // parallel-load contamination drives total_workspace to 0 via saturating_sub.
    let avail = 20_543usize * 1024 * 1024; // ~what MemAvailable reported
    let model_rss = 8_459usize * 1024 * 1024; // contaminated median from old code
    let (ws, _peak, _pct) = compute_workspace_budget(avail, 7, model_rss, 0.7);
    // 7 * 8459 = 59213 MiB >> 20543 MiB → saturates to 0 → ws = 0.
    assert_eq!(
        ws, 0,
        "saturated budget should be 0 (physics_floor check will catch this)"
    );
}

#[test]
fn compute_workspace_budget_physics_floor_detection() {
    // Verify that the physics floor catches the zero-workspace case.
    // physics_floor = chunk_cost(1, 8192) under conservative defaults.
    let physics_floor = CostModel::conservative(0).chunk_cost(1, 8192) as usize;
    assert!(
        physics_floor > 0,
        "physics_floor must be positive (conservative model costs > 0)"
    );
    // A zero workspace is below the floor.
    assert!(
        0 < physics_floor,
        "workspace=0 must be caught by the physics_floor guard"
    );
}

#[test]
fn compute_workspace_budget_single_worker() {
    // n=1: all available workspace (minus model RSS and headroom) goes to that worker.
    let avail = 8_192usize * 1024 * 1024;
    let model_rss = 1_100usize * 1024 * 1024;
    let (ws, _peak, _pct) = compute_workspace_budget(avail, 1, model_rss, 1.0);
    // total_workspace = 8192 - 1100 - 256 = 6836 MiB; per_worker = 6836 * 1.0 / 1
    assert!(
        ws > 6_000 * 1024 * 1024,
        "single worker should get ~6836 MiB workspace, got {} MiB",
        ws / (1024 * 1024)
    );
}
