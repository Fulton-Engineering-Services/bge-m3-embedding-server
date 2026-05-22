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

//! `GET /health` and `GET /health/deep` handlers.
//!
//! `/health` returns lightweight readiness status from in-memory atomics.
//! `/health/deep` runs a tiny canary inference (batch=1, seq≈8 tokens) and
//! returns `503` if the actual embedding pipeline is broken.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

use crate::state::{AppState, ProbeStatus};

/// Fixed canary text for `/health/deep`. Tokenises to ~8 tokens with the
/// BGE-M3 `SentencePiece` vocabulary, meeting the "batch=1, seq=8" goal.
const DEEP_HEALTH_CANARY: &str = "embedding service canary health check ok";

/// Timeout for the canary embed call in `/health/deep`.
///
/// If the worker pool is so overloaded or broken that even the canary cannot
/// complete within this budget, the endpoint returns 503. 30 s is generous
/// for a single batch-1 request but avoids masking a genuinely hung session.
const DEEP_HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Handles `GET /health` — returns readiness status, worker counts, and tuning diagnostics.
///
/// Returns `503` while models are loading or if all workers have exited; returns
/// `200 ok` (or `200 warn` when fewer workers are live than configured) with the
/// current cost-model coefficients and probe status in the `tuning` block.
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

/// Handles `GET /health/deep` — runs a tiny canary inference and returns
/// `503` if the actual embedding pipeline is broken.
///
/// Unlike `GET /health` (which reads only in-memory atomics), this handler
/// submits a real `embed_dense` call through the worker pool and exercises
/// the full tokenize → ORT `session.run()` → projection path, including
/// `TensorRT` engine dispatch on GPU builds. It is the strongest available
/// liveness signal because it catches the silent-failure mode observed in
/// the 2026-05 incident: `/health` returned `200 ok` while every real
/// embedding request returned `500` due to a broken TRT CUDA context.
///
/// # Response codes
///
/// | Code | Condition |
/// |---|---|
/// | `200 ok` | Server is ready and the canary embed succeeded |
/// | `503 loading` | Server is still loading models |
/// | `503 fail` | Canary embed failed or timed out |
///
/// # ECS and ALB configuration
///
/// Point both `ECS healthCheck.command` and the ALB target-group health check
/// at `/health/deep`. The 30-second inference timeout ensures the health check
/// never hangs indefinitely; keep the ECS `healthCheckGracePeriodSeconds`
/// large enough to cover TRT cold-start (≥ 10 800 s for a full 16-shape grid).
pub async fn health_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if !state.ready.load(Ordering::Acquire) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response();
    }

    let canary = tokio::time::timeout(
        DEEP_HEALTH_TIMEOUT,
        state.pool.dense(vec![DEEP_HEALTH_CANARY.to_string()]),
    )
    .await;

    match canary {
        Ok(Ok(_)) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response(),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "health/deep: canary embed failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"status": "fail", "error": e.to_string()})),
            )
                .into_response()
        }
        Err(_elapsed) => {
            tracing::warn!(
                timeout_secs = DEEP_HEALTH_TIMEOUT.as_secs(),
                "health/deep: canary embed timed out"
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"status": "fail", "error": "canary embed timed out"})),
            )
                .into_response()
        }
    }
}
