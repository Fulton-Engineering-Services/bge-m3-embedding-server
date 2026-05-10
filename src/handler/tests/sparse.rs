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

use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::body::to_bytes;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use tokio::sync::Semaphore;

use super::super::{models, sparse_embeddings};
use super::helpers::make_state;
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::error::AppError;
use crate::models::{SparseRequest, TextInput};
use crate::state::{AppState, ProbeStatus};

// ── sparse_embeddings handler ─────────────────────────────────────────────

#[tokio::test]
async fn sparse_embeddings_rejects_when_not_ready() {
    let state = make_state(false, 256);
    let req = SparseRequest {
        input: TextInput(vec!["hello".to_string()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn sparse_embeddings_rejects_when_pool_dead() {
    let state = make_state(true, 256);
    let req = SparseRequest {
        input: TextInput(vec!["hello".to_string()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}

#[tokio::test]
async fn sparse_embeddings_rejects_empty_input() {
    let state = make_state(false, 256);
    let req = SparseRequest {
        input: TextInput(vec![]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn sparse_embeddings_rejects_over_batch() {
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], vec![]),
        ready: AtomicBool::new(true),
        max_batch: 2,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = SparseRequest {
        input: TextInput(vec!["a".into(), "b".into(), "c".into()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::InvalidRequest(ref msg)) if msg.contains("exceeds")),
        "expected InvalidRequest for over-batch, got: {result:?}"
    );
}

#[tokio::test]
async fn sparse_embeddings_returns_correct_shape() {
    let sparse_fixture = vec![crate::embedder::SparseEmbedding {
        indices: vec![42usize],
        values: vec![0.5f32],
    }];
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(vec![], sparse_fixture),
        ready: AtomicBool::new(true),
        max_batch: 256,
        total_workers: 1,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    });
    let req = SparseRequest {
        input: TextInput(vec!["hello".into()]),
    };
    let result = sparse_embeddings(State(state), Json(req)).await;
    assert!(
        result.is_ok(),
        "expected Ok but got error from sparse handler"
    );
    let Json(resp) = result.expect("sparse_embeddings should succeed");
    assert_eq!(resp.data.len(), 1);
    assert_eq!(resp.data[0].sparse_values.indices, vec![42u32]);
    assert_eq!(resp.data[0].sparse_values.values, vec![0.5f32]);
}

// ── models handler ────────────────────────────────────────────────────────

#[tokio::test]
async fn models_handler_returns_bge_m3_entry() {
    let state = make_state(true, 256);
    let response = models(State(state)).await.into_response();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["data"][0]["id"], "bge-m3");
    assert_eq!(json["data"][0]["object"], "model");
}
