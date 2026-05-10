use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::state::{AppState, ProbeStatus};

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ready = state.ready.load(Ordering::Acquire);
    let live = state.pool.live_worker_count();
    let loaded = state.pool.loaded_worker_count();
    let total = state.total_workers;

    if !ready {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response();
    }

    if live == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "fail",
                "workers": { "live": live, "total": total }
            })),
        )
            .into_response();
    }

    if loaded == 0 {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "idle",
                "workers": { "live": live, "total": total }
            })),
        )
            .into_response();
    }

    let status = if live < total { "warn" } else { "ok" };

    // Read the live cost model and probe status atomically.
    let cm = state.cost_model.load();
    let probe_status = ProbeStatus::from_u8(state.probe_status.load(Ordering::Acquire)).as_str();

    let mut tuning = serde_json::json!({
        "a_bytes_per_token": cm.a,
        "b_bytes_per_token_sq": cm.b,
        "max_workspace_bytes": cm.max_workspace_bytes,
        "probe_status": probe_status,
    });

    // Add static memory fields when available (written before probe starts).
    if let Some(ti) = state.tuning.get() {
        tuning["memory_source"] = serde_json::Value::String(ti.memory_source.clone());
        tuning["available_bytes"] =
            serde_json::Value::Number(serde_json::Number::from(ti.available_bytes));
        tuning["model_rss_bytes_per_worker"] =
            serde_json::Value::Number(serde_json::Number::from(ti.model_rss_bytes_per_worker));
    }

    let body = serde_json::json!({
        "status": status,
        "workers": { "live": live, "total": total },
        "max_seq_length": state.max_seq_length,
        "tuning": tuning,
    });

    (StatusCode::OK, Json(body)).into_response()
}
