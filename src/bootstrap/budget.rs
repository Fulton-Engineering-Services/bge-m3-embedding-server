//! Pure workspace-budget arithmetic shared between the readiness probe and
//! the unit tests.

use crate::embedder::OS_HEADROOM_BYTES;

/// Computes per-worker workspace budget and derived stats from memory inputs.
///
/// # Returns
///
/// `(per_worker_workspace, worst_case_peak, utilization_pct)` where:
/// - `per_worker_workspace`: bytes available to one worker for a single
///   `session.run()` call (passed as `rss_ceiling` to the probe).
/// - `worst_case_peak`: total bytes consumed when all workers run
///   simultaneously at budget ceiling (used for the 90% OOM warning).
/// - `utilization_pct`: `worst_case_peak / available_bytes × 100`.
///
/// Extracted as a pure function so the budget logic is unit-testable
/// independently of the async readiness probe machinery.
//
// cast_precision_loss: available_bytes ≤ ~28 GB (Fargate limit), total_workspace
//   similarly bounded; f64 has 2^52 mantissa (~4.5 PB) — no precision loss.
// cast_possible_truncation: per_worker_workspace is a byte budget; truncating
//   sub-byte fractions is intentional and harmless.
// cast_sign_loss: total_workspace is derived from saturating_sub — always ≥ 0.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(super) fn compute_workspace_budget(
    available_bytes: usize,
    n_workers: usize,
    model_rss_per_worker: usize,
    safety_factor: f64,
) -> (usize, usize, f64) {
    let total_workspace = available_bytes
        .saturating_sub(n_workers.saturating_mul(model_rss_per_worker))
        .saturating_sub(OS_HEADROOM_BYTES);
    let per_worker_workspace = (total_workspace as f64 * safety_factor / n_workers as f64) as usize;

    let worst_case_peak = n_workers
        .saturating_mul(per_worker_workspace)
        .saturating_add(n_workers.saturating_mul(model_rss_per_worker))
        .saturating_add(OS_HEADROOM_BYTES);

    let utilization_pct = if available_bytes > 0 {
        worst_case_peak as f64 / available_bytes as f64 * 100.0
    } else {
        0.0
    };

    (per_worker_workspace, worst_case_peak, utilization_pct)
}
