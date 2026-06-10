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

use super::*;

const GUARD_SEQ: usize = 4096;

#[test]
fn admits_seq_below_guard_threshold_even_when_uncovered() {
    // seq < guard_seq: a cold JIT here is bounded; always admit so the
    // profile can grow naturally.
    let guard = TrtJitGuard::new(GUARD_SEQ, 0);
    assert!(guard.admit(32, 128).is_ok());
    assert!(guard.admit(256, 2048).is_ok());
    assert!(guard.admit(8, GUARD_SEQ - 1).is_ok());
}

#[test]
fn admits_dangerous_seq_when_within_warmed_coverage() {
    // Healthy steady state: ceiling at the top tier covers every reachable
    // shape, so even the maximum sequence length is admitted (cache hit).
    let guard = TrtJitGuard::new(GUARD_SEQ, 8192);
    assert!(guard.admit(1, 8192).is_ok());
    assert!(guard.admit(16, 8192).is_ok());
    assert!(guard.admit(8, 5000).is_ok());
}

#[test]
fn refuses_dangerous_uncovered_seq() {
    // The worker-3 failure mode: seq=8192 shard failed to compile, so the
    // ceiling stayed at the highest tier that did compile (2048). A real
    // seq=8192 request must be refused, not crash the process.
    let guard = TrtJitGuard::new(GUARD_SEQ, 2048);
    let err = guard
        .admit(8, 8192)
        .expect_err("must refuse uncovered seq=8192");
    assert_eq!(err.batch, 8);
    assert_eq!(err.seq, 8192);
    assert_eq!(err.guard_seq, GUARD_SEQ);
    assert_eq!(err.warmed_seq_ceiling, 2048);
}

#[test]
fn refuses_when_nothing_warmed() {
    // ceiling == 0 (no warmup succeeded / empty grid): any dangerous-tier
    // sequence is refused.
    let guard = TrtJitGuard::new(GUARD_SEQ, 0);
    assert!(guard.admit(1, GUARD_SEQ).is_err());
    assert!(guard.admit(1, 8192).is_err());
}

#[test]
fn boundary_seq_equal_to_guard_seq_is_dangerous() {
    // seq == guard_seq is in the dangerous range (>=). Uncovered -> refuse.
    let guard = TrtJitGuard::new(GUARD_SEQ, 2048);
    assert!(guard.admit(1, GUARD_SEQ).is_err());
}

#[test]
fn boundary_seq_equal_to_ceiling_is_covered() {
    // seq == ceiling is covered (the profile includes the ceiling) -> admit.
    let guard = TrtJitGuard::new(GUARD_SEQ, 8192);
    assert!(guard.admit(1, 8192).is_ok());
}

#[test]
fn guard_chunks_none_admits_everything() {
    let seq_lens = vec![8192, 8192, 8192];
    let chunks = vec![vec![0, 1, 2]];
    assert!(guard_chunks(None, &chunks, &seq_lens).is_ok());
}

#[test]
fn guard_chunks_uses_chunk_max_seq_and_batch() {
    // One chunk with three texts; max seq 8192 is the dangerous, uncovered tier.
    let guard = TrtJitGuard::new(GUARD_SEQ, 2048);
    let seq_lens = vec![128, 8192, 512];
    let chunks = vec![vec![0, 1, 2]];
    let err = guard_chunks(Some(&guard), &chunks, &seq_lens)
        .expect_err("chunk max seq 8192 must be refused");
    assert_eq!(err.seq, 8192);
    assert_eq!(err.batch, 3, "batch is the chunk length");
}

#[test]
fn guard_chunks_admits_small_seq_chunks() {
    let guard = TrtJitGuard::new(GUARD_SEQ, 2048);
    let seq_lens = vec![128, 512, 2048];
    let chunks = vec![vec![0, 1], vec![2]];
    assert!(guard_chunks(Some(&guard), &chunks, &seq_lens).is_ok());
}

#[test]
fn guard_chunks_refuses_first_offending_chunk() {
    // First chunk safe (seq 512), second chunk dangerous (seq 8192).
    let guard = TrtJitGuard::new(GUARD_SEQ, 2048);
    let seq_lens = vec![512, 8192];
    let chunks = vec![vec![0], vec![1]];
    let err = guard_chunks(Some(&guard), &chunks, &seq_lens).expect_err("second chunk unsafe");
    assert_eq!(err.seq, 8192);
}

#[test]
fn is_trt_shape_rejected_detects_direct_error() {
    let err: anyhow::Error = TrtJitRejection {
        batch: 4,
        seq: 8192,
        guard_seq: GUARD_SEQ,
        warmed_seq_ceiling: 2048,
    }
    .into();
    assert!(is_trt_shape_rejected(&err));
}

#[test]
fn is_trt_shape_rejected_detects_through_context_chain() {
    // The worker wraps embed errors with .context(); detection must walk the
    // source chain rather than only inspecting the top-level error.
    let base: anyhow::Error = TrtJitRejection {
        batch: 4,
        seq: 8192,
        guard_seq: GUARD_SEQ,
        warmed_seq_ceiling: 2048,
    }
    .into();
    let wrapped = base.context("dual embed error");
    assert!(is_trt_shape_rejected(&wrapped));
}

#[test]
fn is_trt_shape_rejected_false_for_other_errors() {
    let err = anyhow::anyhow!("Could not find any implementation for Unsqueeze node");
    assert!(!is_trt_shape_rejected(&err));
}
