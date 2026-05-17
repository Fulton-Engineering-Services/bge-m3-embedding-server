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

//! Background probe task spawned when the cost model has not been overridden
//! and the EFS cache is empty (or disabled).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use tracing::info;

use super::readiness::run_readiness_checks_and_open;
use crate::binpack::CostModel;
use crate::probe;
use crate::state::{AppState, ProbeStatus};

/// Spawns the background probe task with proper permit ownership.
///
/// # Serialisation protocol
///
/// 1. Set `probe_status = Running`.
/// 2. Acquire `cfg_workers - 1` permits via `acquire_many_owned` — combined
///    with the 1 permit already reserved at startup, this drains the
///    semaphore to 0 so all incoming `/v1/embeddings*` requests queue
///    behind the gate while the probe is in flight.
/// 3. Move the [`tokio::sync::OwnedSemaphorePermit`] into the spawned
///    task. Its destructor is invoked just before `add_permits(cfg_workers)`
///    at the end of the task, restoring full traffic concurrency.
///
/// **Rationale for `acquire_many_owned`:** `tokio::spawn` returns
/// synchronously before the spawned task starts executing. A permit bound to
/// a local variable in the parent function would be dropped immediately at
/// the end of that function — before the probe begins — leaving the semaphore
/// un-drained and allowing real traffic to contaminate per-shape RSS
/// measurements. `acquire_many_owned` returns an `OwnedSemaphorePermit`
/// independent of the source `Semaphore` lifetime, so it survives the move
/// into the async closure and is held for the full duration of the probe.
#[allow(clippy::too_many_arguments)]
pub(super) async fn spawn_probe_task(
    state: Arc<AppState>,
    cfg_workers: usize,
    cfg_max_seq: usize,
    per_worker_workspace: usize,
    cgroup_limit_bytes: usize,
    cache_dir: PathBuf,
    model_variant_str: String,
    save_cache: bool,
) {
    state
        .probe_status
        .store(ProbeStatus::Running as u8, Ordering::Release);

    // Drain all remaining permits. The semaphore starts with
    // `max(cfg_workers - 1, 1)` permits at startup (one slot reserved for
    // the probe worker); we acquire the remaining `cfg_workers - 1` here
    // so the count drops to 0 for the duration of the probe.
    //
    // `acquire_many_owned` returns an `OwnedSemaphorePermit` that we move
    // into the spawned task closure. The permit's drop handler returns the
    // permits to the semaphore — we manually call `add_permits(cfg_workers)`
    // in the task to also release the originally-reserved probe slot.
    let probe_permit = Arc::clone(&state.request_permits)
        .acquire_many_owned(u32::try_from(cfg_workers.saturating_sub(1)).unwrap_or(u32::MAX))
        .await
        .ok();

    tokio::spawn(async move {
        // Forget the OwnedSemaphorePermit at the end; we manually
        // add_permits(cfg_workers) below so the count goes from 0
        // straight to cfg_workers (releasing both the drained permits
        // and the originally-reserved probe slot in one operation).
        if let Some(p) = probe_permit {
            p.forget();
        }

        let (a, b) = probe::run_probe(
            &state.pool,
            cfg_max_seq,
            per_worker_workspace,
            cgroup_limit_bytes,
        )
        .await;
        let cm = CostModel {
            a,
            b,
            max_workspace_bytes: per_worker_workspace,
        };
        info!(
            a = cm.a,
            b = cm.b,
            max_workspace_mb = cm.max_workspace_bytes / (1024 * 1024),
            "Probe complete — updating cost model"
        );
        state.cost_model.store(Arc::new(cm));
        // Distinguish real fit from conservative fallback.
        let status = if (a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON
            && (b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON
        {
            ProbeStatus::Failed
        } else {
            if save_cache {
                probe::save_probe_cache(&cache_dir, &model_variant_str, cfg_max_seq, a, b);
            }
            ProbeStatus::Complete
        };
        state.probe_status.store(status as u8, Ordering::Release);
        info!(probe_status = status.as_str(), "Probe status updated");

        // Readiness checks run inside the probe task so they do not
        // contaminate the probe's RSS measurements.
        if let Err(e) = run_readiness_checks_and_open(&state).await {
            tracing::error!(error = %e, "Post-probe readiness check failed");
        }
        // Release the drained permits AND the originally-reserved probe
        // slot in one operation. Net effect: semaphore count goes from 0
        // back to cfg_workers, opening traffic at full concurrency.
        state.request_permits.add_permits(cfg_workers);
    });
}
