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

//! Tests for dispatch helpers that do not require a loaded ORT session.

use super::super::dispatch::reply_request_load_error;
use super::super::guard::log_client_abandoned_before_dispatch;
use crate::embedder::types::EmbedRequest;

#[test]
fn log_client_abandoned_before_dispatch_does_not_panic() {
    log_client_abandoned_before_dispatch(3, "dense", 16);
}

#[test]
fn reply_load_error_dense_delivers_err() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let req = EmbedRequest::Dense {
        texts: vec!["hello".to_string()],
        reply: tx,
    };
    reply_request_load_error(req, anyhow::anyhow!("reload failed"));
    assert!(rx.try_recv().unwrap().is_err());
}

#[test]
fn reply_load_error_sparse_delivers_err() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let req = EmbedRequest::Sparse {
        texts: vec![],
        reply: tx,
    };
    reply_request_load_error(req, anyhow::anyhow!("reload failed"));
    assert!(rx.try_recv().unwrap().is_err());
}

#[test]
fn reply_load_error_both_delivers_err() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let req = EmbedRequest::Both {
        texts: vec![],
        reply: tx,
    };
    reply_request_load_error(req, anyhow::anyhow!("reload failed"));
    assert!(rx.try_recv().unwrap().is_err());
}

#[test]
fn reply_load_error_probe_delivers_err() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let req = EmbedRequest::Probe {
        texts: vec![],
        reply: tx,
    };
    reply_request_load_error(req, anyhow::anyhow!("reload failed"));
    let err = rx.try_recv().unwrap().expect_err("probe must fail");
    assert!(err.to_string().contains("reload failed"));
}

#[test]
fn reply_load_error_adaptive_warmup_delivers_err() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let req = EmbedRequest::AdaptiveWarmup {
        batch: 1,
        seq: 128,
        ack: tx,
    };
    reply_request_load_error(req, anyhow::anyhow!("reload failed"));
    assert!(rx.try_recv().unwrap().is_err());
}
