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

//! Tests for shape-guard wiring, inference outcome classification, and the
//! shared post-inference finalize path.

use std::collections::HashSet;

use super::super::guard::{
    EmbedRouteContext, InferenceOutcome, adaptive_warmup_non_trt_compile_ms, build_shape_guard,
    classify_inference_outcome, finalize_embed_route, next_consecutive_failures,
    should_unload_on_outcome,
};
use super::helpers::{sample_embed_stats, test_worker_config};
use crate::config::EpSelection;
use crate::embedder::jit_guard::TrtJitRejection;

#[test]
fn build_shape_guard_none_on_cpu_ep() {
    let cfg = test_worker_config(EpSelection::Cpu, 8192);
    assert!(build_shape_guard(&cfg).is_none());
}

#[test]
fn build_shape_guard_none_when_disabled() {
    let mut cfg = test_worker_config(EpSelection::TensorRt, 2048);
    cfg.trt_inband_jit_guard_enabled = false;
    assert!(build_shape_guard(&cfg).is_none());
}

#[test]
fn build_shape_guard_active_on_trt_with_ceiling() {
    let cfg = test_worker_config(EpSelection::TensorRt, 2048);
    let guard = build_shape_guard(&cfg).expect("TRT + guard enabled must build guard");
    assert!(guard.admit(1, 8192).is_err());
    assert!(guard.admit(1, 2048).is_ok());
}

#[test]
fn classify_outcome_ok_on_success() {
    assert_eq!(
        classify_inference_outcome(false, false, false, 3, 5),
        InferenceOutcome::Ok
    );
}

#[test]
fn classify_outcome_rejected_before_generic_failure() {
    assert_eq!(
        classify_inference_outcome(false, true, true, 0, 5),
        InferenceOutcome::Rejected
    );
}

#[test]
fn classify_outcome_trt_fatal_highest_priority() {
    assert_eq!(
        classify_inference_outcome(true, true, true, 0, 5),
        InferenceOutcome::TrtFatal
    );
}

#[test]
fn classify_outcome_failure_below_threshold() {
    assert_eq!(
        classify_inference_outcome(false, false, true, 2, 5),
        InferenceOutcome::Failure
    );
}

#[test]
fn classify_outcome_circuit_break_at_threshold() {
    assert_eq!(
        classify_inference_outcome(false, false, true, 4, 5),
        InferenceOutcome::CircuitBreak
    );
}

#[test]
fn next_consecutive_failures_resets_on_ok_and_circuit_break() {
    assert_eq!(next_consecutive_failures(InferenceOutcome::Ok, 4), 0);
    assert_eq!(
        next_consecutive_failures(InferenceOutcome::CircuitBreak, 4),
        0
    );
}

#[test]
fn next_consecutive_failures_increments_on_failure() {
    assert_eq!(next_consecutive_failures(InferenceOutcome::Failure, 2), 3);
}

#[test]
fn next_consecutive_failures_unchanged_on_rejection() {
    assert_eq!(next_consecutive_failures(InferenceOutcome::Rejected, 2), 2);
}

#[test]
fn should_unload_only_on_circuit_break() {
    assert!(should_unload_on_outcome(InferenceOutcome::CircuitBreak));
    assert!(!should_unload_on_outcome(InferenceOutcome::Failure));
    assert!(!should_unload_on_outcome(InferenceOutcome::Rejected));
}

#[test]
fn finalize_embed_route_success_sends_reply() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let mut warmed = HashSet::new();
    let stats = sample_embed_stats(12, 128);
    let ctx = EmbedRouteContext {
        worker_id: 0,
        route: "dense",
        consecutive_failures: 0,
        circuit_breaker_threshold: 5,
        jit_suspect_tx: None,
        engine_propagation_tx: None,
        batch_len: 1,
    };
    let outcome = finalize_embed_route(&ctx, Ok((vec![vec![0.1f32]], stats)), tx, 12, &mut warmed);
    assert_eq!(outcome, InferenceOutcome::Ok);
    let reply = rx
        .try_recv()
        .expect("reply must be sent")
        .expect("ok embed");
    assert_eq!(reply.1.max_chunk_seq, 128);
}

#[test]
fn finalize_embed_route_jit_rejection_maps_to_rejected() {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let mut warmed = HashSet::new();
    let err: anyhow::Error = TrtJitRejection {
        batch: 4,
        seq: 8192,
        guard_seq: 4096,
        warmed_seq_ceiling: 2048,
    }
    .into();
    let ctx = EmbedRouteContext {
        worker_id: 1,
        route: "both",
        consecutive_failures: 2,
        circuit_breaker_threshold: 5,
        jit_suspect_tx: None,
        engine_propagation_tx: None,
        batch_len: 4,
    };
    let outcome = finalize_embed_route::<Vec<Vec<f32>>>(&ctx, Err(err), tx, 5, &mut warmed);
    assert_eq!(outcome, InferenceOutcome::Rejected);
    assert_eq!(next_consecutive_failures(outcome, 2), 2);
    assert!(rx.try_recv().expect("reply sent").is_err());
}

#[test]
fn finalize_embed_route_slow_inference_broadcasts_shape() {
    let (jit_tx, mut jit_rx) = tokio::sync::mpsc::channel(4);
    let (prop_tx, _prop_rx) = tokio::sync::broadcast::channel(4);
    let (reply_tx, _reply_rx) = tokio::sync::oneshot::channel();
    let mut warmed = HashSet::new();
    let stats = sample_embed_stats(6_000, 512);
    let ctx = EmbedRouteContext {
        worker_id: 0,
        route: "sparse",
        consecutive_failures: 0,
        circuit_breaker_threshold: 5,
        jit_suspect_tx: Some(&jit_tx),
        engine_propagation_tx: Some(&prop_tx),
        batch_len: 8,
    };
    let outcome = finalize_embed_route::<Vec<Vec<f32>>>(
        &ctx,
        Ok((vec![], stats)),
        reply_tx,
        6_000,
        &mut warmed,
    );
    assert_eq!(outcome, InferenceOutcome::Ok);
    assert!(warmed.contains(&(8, 512)));
    assert_eq!(jit_rx.try_recv().unwrap(), (8, 512));
}

#[test]
fn adaptive_warmup_non_trt_returns_zero() {
    assert_eq!(adaptive_warmup_non_trt_compile_ms(), 0);
}
