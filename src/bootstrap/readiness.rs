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

//! Foreground readiness probe.
//!
//! Waits for the worker pool's init handle, computes the per-worker workspace
//! budget, writes static [`crate::state::TuningInfo`], resolves the cost model
//! (override → EFS cache hit → background probe), and finally flips
//! `state.ready` once the dense + sparse readiness checks succeed.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tracing::info;

use super::budget::compute_workspace_budget;
use super::probe_task::spawn_probe_task;
use crate::binpack::CostModel;
use crate::probe;
use crate::state::{AppState, ProbeStatus, TuningInfo};
use crate::sysinfo;

/// Runs after all workers finish loading their model instances.
///
/// # Sequence
///
/// 1. Wait for worker pool initialisation to finish.
/// 2. Read `pool.model_rss_per_worker_bytes()` — the median RSS delta measured
///    inside each worker's `spawn_blocking` closure around `load_models()`.
///    Workers load sequentially (one at a time), so each delta reflects only
///    that worker's ORT session allocation with no parallel-load contamination.
/// 3. Detect available memory; compute `per_worker_workspace` via
///    `compute_workspace_budget`. Fail fast if the budget is below the
///    physics-based floor (cannot fit even one text at `max_seq_length`).
/// 4. Write static [`TuningInfo`] to `OnceLock`.
/// 5. Resolve the cost model — one of three paths:
///    - cost-model override set: apply immediately, `probe_status = Disabled`.
///    - EFS cache hit: apply cached `(a, b)` via `ArcSwap`, `probe_status = CacheHit`.
///    - cache miss: set `probe_status = Running`, launch background probe task.
/// 6. Run dense + sparse readiness calls to confirm the worker pool is healthy.
/// 7. Flip `state.ready = true` — `/health` returns `200 ok` from this point on.
///    If the probe is still running in the background, the bin-packer uses
///    conservative defaults until the `ArcSwap` is updated (typically ~120 s).
///
// cast_possible_truncation: physics_floor is a u128 workspace estimate; truncating
//   to usize is safe because per_worker_workspace is itself bounded by available_bytes
//   which fits comfortably in usize on any 64-bit target.
// cast_precision_loss / cast_sign_loss: delegated to compute_workspace_budget.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub async fn run_readiness_probe(
    init_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    state: Arc<AppState>,
    cfg_max_seq: usize,
    cfg_workers: usize,
    cfg_safety: f64,
    cost_model_override: Option<CostModel>,
    cache_dir: PathBuf,
    model_variant_str: String,
    disable_probe_cache: bool,
) -> anyhow::Result<()> {
    init_handle
        .await
        .map_err(|e| anyhow::anyhow!("Worker pool task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("Worker pool initialization failed: {e}"))?;

    // --- Memory detection ---
    let mem = sysinfo::detect_available_memory();
    info!(
        available_bytes = mem.available_bytes,
        source = %mem.source,
        "Memory detected"
    );

    // Per-worker model RSS is the median of per-worker deltas collected by
    // EmbedPool::spawn. Workers load sequentially (one at a time) so each
    // delta reflects only that worker's ORT session allocation. The median
    // is robust to one outlier from page-cache settling or ORT arena jitter.
    let model_rss_per_worker = state.pool.model_rss_per_worker_bytes();
    info!(
        model_rss_per_worker_mb = model_rss_per_worker / (1024 * 1024),
        "Measured model RSS per worker (median across all workers)"
    );

    // Compute per-worker workspace ceiling.
    let (per_worker_workspace, worst_case_peak, utilization_pct) = compute_workspace_budget(
        mem.available_bytes,
        cfg_workers,
        model_rss_per_worker,
        cfg_safety,
    );

    info!(
        worst_case_peak_mb = worst_case_peak / (1024 * 1024),
        available_mb = mem.available_bytes / (1024 * 1024),
        utilization_pct = format!("{utilization_pct:.1}"),
        per_worker_workspace_mb = per_worker_workspace / (1024 * 1024),
        "Workspace budget computed (worst-case all-workers-peak)"
    );
    if utilization_pct > 90.0 {
        tracing::warn!(
            utilization_pct = format!("{utilization_pct:.1}"),
            "Worst-case workspace peak exceeds 90% of available memory; \
             consider lowering BGE_M3_MEMORY_SAFETY_FACTOR or BGE_M3_WORKERS"
        );
    }

    // Physics-based safety floor: the minimum workspace required to run a
    // single text at the configured max sequence length under conservative
    // cost-model coefficients. If the computed per_worker_workspace falls
    // below this floor, the measurement upstream is broken (e.g. inflated
    // model_rss_per_worker driving total_workspace to zero via saturating_sub).
    // Continuing in this state degrades bin_pack to batch=1 and produces
    // silent throughput collapse — fail fast instead so ECS restarts the task
    // and the operator sees a clear error rather than a degraded service.
    let physics_floor = CostModel::conservative(0).chunk_cost(1, cfg_max_seq) as usize;
    if per_worker_workspace < physics_floor {
        return Err(anyhow::anyhow!(
            "Computed per_worker_workspace ({per_worker_workspace} B = {} MiB) is below the \
             physics-based minimum ({physics_floor} B = {} MiB) needed to run one text at \
             max_seq_length={cfg_max_seq}. Likely causes: model_rss_per_worker ({} MiB) is \
             over-estimated (parallel-load contamination), BGE_M3_MEMORY_SAFETY_FACTOR too low \
             ({cfg_safety}), BGE_M3_WORKERS too high ({cfg_workers}) for available memory \
             ({} MiB), or BGE_M3_AVAILABLE_MEMORY_BYTES override too small.",
            per_worker_workspace / (1024 * 1024),
            physics_floor / (1024 * 1024),
            model_rss_per_worker / (1024 * 1024),
            mem.available_bytes / (1024 * 1024),
        ));
    }

    // Write static memory + budget info now so /health always shows these fields
    // even while the background probe is still running.
    let _ = state.tuning.set(TuningInfo::new(
        &mem,
        model_rss_per_worker,
        worst_case_peak,
        utilization_pct,
    ));

    // The cgroup-limit byte count (the actual kernel ceiling, not the
    // safety-discounted budget) is threaded into run_probe so the per-shape
    // RSS guard can compare against the real ceiling rather than the
    // discounted per_worker_workspace value.
    let cgroup_limit_bytes = mem.available_bytes;

    // --- Cost model resolution ---
    if let Some(cm) = cost_model_override {
        info!(
            a = cm.a,
            b = cm.b,
            max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
            "Using pre-configured cost model (probe skipped)"
        );
        state.cost_model.store(Arc::new(cm));
        state
            .probe_status
            .store(ProbeStatus::Disabled as u8, Ordering::Release);
        // No probe — run readiness checks inline and open traffic.
        run_readiness_checks_and_open(&state).await?;
    } else if !disable_probe_cache {
        // Try to load cached coefficients from EFS.
        if let Some((a, b)) =
            probe::try_load_probe_cache(&cache_dir, &model_variant_str, cfg_max_seq)
        {
            let cm = CostModel {
                a,
                b,
                max_workspace_bytes: per_worker_workspace,
            };
            info!(
                a,
                b,
                max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
                "Cost model loaded from EFS cache"
            );
            state.cost_model.store(Arc::new(cm));
            state
                .probe_status
                .store(ProbeStatus::CacheHit as u8, Ordering::Release);
            // Cache hit — run readiness checks inline and open traffic.
            run_readiness_checks_and_open(&state).await?;
        } else {
            // Cache miss — probe must run. See `spawn_probe_task` for the
            // serialisation protocol that holds all cfg_workers permits across
            // the probe + readiness window.
            spawn_probe_task(
                Arc::clone(&state),
                cfg_workers,
                cfg_max_seq,
                per_worker_workspace,
                cgroup_limit_bytes,
                cache_dir,
                model_variant_str,
                /* save_cache = */ true,
            )
            .await;
            return Ok(());
        }
    } else {
        // BGE_M3_DISABLE_PROBE_CACHE=1 but no override — run probe without caching.
        spawn_probe_task(
            Arc::clone(&state),
            cfg_workers,
            cfg_max_seq,
            per_worker_workspace,
            cgroup_limit_bytes,
            cache_dir,
            model_variant_str,
            /* save_cache = */ false,
        )
        .await;
        return Ok(());
    }

    Ok(())
}

/// Runs the dense + sparse readiness calls and flips `state.ready`.
///
/// Called from the override/cache-hit paths (inline, before returning from
/// `run_readiness_probe`) and from the background probe task (after the probe
/// completes) so that readiness checks never run concurrently with the probe.
pub(super) async fn run_readiness_checks_and_open(state: &AppState) -> anyhow::Result<()> {
    state
        .pool
        .dense(vec!["ready".into()])
        .await
        .map_err(|e| anyhow::anyhow!("Dense readiness probe failed: {e}"))?;

    state
        .pool
        .sparse(vec!["ready".into()])
        .await
        .map_err(|e| anyhow::anyhow!("Sparse readiness probe failed: {e}"))?;

    state
        .ready
        .store(true, std::sync::atomic::Ordering::Release);
    tracing::info!("Models ready — accepting requests");
    Ok(())
}
