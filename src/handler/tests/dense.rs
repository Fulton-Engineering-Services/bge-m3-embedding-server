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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8};

use arc_swap::ArcSwap;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use tokio::sync::Semaphore;

use super::super::dense_embeddings;
use super::helpers::make_state;
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::error::AppError;
use crate::models::{DenseRequest, TextInput};
use crate::state::{AppState, ProbeStatus};

// ── rejection paths ───────────────────────────────────────────────────────

#[tokio::test]
async fn dense_embeddings_rejects_when_not_ready() {
    let state = make_state(false, 256);
    let req = DenseRequest {
        input: TextInput(vec!["hello".to_string()]),
        model: None,
    };
    let result = dense_embeddings(State(state), HeaderMap::new(), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn dense_embeddings_rejects_when_pool_dead() {
    let state = make_state(true, 256);
    let req = DenseRequest {
        input: TextInput(vec!["hello".to_string()]),
        model: None,
    };
    let result = dense_embeddings(State(state), HeaderMap::new(), Json(req)).await;
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}

#[tokio::test]
async fn dense_embeddings_rejects_empty_input() {
    let state = make_state(false, 256);
    let req = DenseRequest {
        input: TextInput(vec![]),
        model: None,
    };
    let result = dense_embeddings(State(state), HeaderMap::new(), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

#[tokio::test]
async fn dense_embeddings_rejects_over_batch() {
    let state = make_state(false, 2);
    let req = DenseRequest {
        input: TextInput(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        model: None,
    };
    let result = dense_embeddings(State(state), HeaderMap::new(), Json(req)).await;
    assert!(matches!(result, Err(AppError::ServiceUnavailable(_))));
}

// ── happy path ────────────────────────────────────────────────────────────

#[tokio::test]
async fn dense_embeddings_returns_correct_shape() {
    let fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let state = Arc::new(AppState {
        pool: EmbedPool::with_fixed_responses(fixture, vec![]),
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
    let req = DenseRequest {
        input: TextInput(vec!["hello".into(), "world".into()]),
        model: None,
    };
    let result = dense_embeddings(State(state), HeaderMap::new(), Json(req)).await;
    assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
    let Json(resp) = result.expect("dense_embeddings should succeed");
    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2, 0.3]);
    assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
    assert_eq!(resp.data[0].index, 0);
    assert_eq!(resp.data[1].index, 1);
    assert_eq!(resp.object, "list");
    assert_eq!(resp.model, "bge-m3");
}
