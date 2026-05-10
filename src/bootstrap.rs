//! Server-startup orchestration: routing, workspace budget, readiness probe,
//! and the background probe task that fits the cost model on first start.
//!
//! Submodules:
//! - `router`: axum `Router` construction + tracing/request-id layers.
//! - `budget`: pure workspace-budget arithmetic (`compute_workspace_budget`).
//! - `readiness`: the foreground readiness probe (`run_readiness_probe`,
//!   `run_readiness_checks_and_open`).
//! - `probe_task`: the background probe task (`spawn_probe_task`) used when
//!   the cost model has not been overridden and no EFS cache hit was found.

mod budget;
mod probe_task;
mod readiness;
mod router;

pub use readiness::run_readiness_probe;
pub use router::build_router;

#[cfg(test)]
mod tests;
