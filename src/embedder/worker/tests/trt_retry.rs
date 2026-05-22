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

//! Tests for `is_trt_jit_oom` error-pattern matching and
//! `embed_with_trt_retry` retry semantics and workspace-halving logic.

use super::super::{embed_with_trt_retry, is_trt_jit_oom};
use crate::binpack::CostModel;

// --- is_trt_jit_oom ---

#[test]
fn is_trt_jit_oom_matches_user_allocator_error() {
    let e = anyhow::anyhow!("TRT: User allocator error during cuDNN workspace setup");
    assert!(is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_matches_no_impl_with_workspace() {
    let e = anyhow::anyhow!("Could not find any implementation for node; workspace too small");
    assert!(is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_matches_no_impl_with_alloc_keyword() {
    // "alloc" branch: allocation failure during TRT autotuning reported via
    // the "could not find any implementation" + "alloc" combination.
    let e = anyhow::anyhow!("Could not find any implementation for node /MatMul_42; alloc failed");
    assert!(is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_does_not_match_no_impl_without_alloc() {
    // "Could not find any implementation" without workspace/alloc = unsupported op, not OOM
    let e = anyhow::anyhow!("Could not find any implementation for node /Reshape_3");
    assert!(!is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_does_not_match_unrelated_error() {
    let e = anyhow::anyhow!("OrtStatus: NOT_IMPLEMENTED: opset 18 not supported");
    assert!(!is_trt_jit_oom(&e));
}

#[test]
fn is_trt_jit_oom_does_not_match_empty() {
    let e = anyhow::anyhow!("");
    assert!(!is_trt_jit_oom(&e));
}

// --- is_trt_jit_oom: "failed to create engine" branch ---
//
// These cover the TRT EP build-time failure family observed in production
// on large fused-route requests (high batch and token counts). The
// application-level error string is:
//
//     Dual embed error: Non-zero status code returned while running
//     TRTKernel_graph_main_graph_<hash>_0 node. Name:'...' Status Message:
//     TensorRT EP failed to create engine from network.
//
// We classify this family as retryable JIT-OOM **only** when paired with a
// workspace/allocation/memory/tactic qualifier so retries don't waste a full
// inference budget on genuinely unsupported graphs.

/// Representative production error with a `workspace` qualifier
/// appended (simulating the case where ORT propagates the TRT logger output
/// into the outer error string). MUST classify as retryable JIT-OOM.
#[test]
fn is_trt_jit_oom_matches_failed_to_create_engine_with_workspace() {
    let e = anyhow::anyhow!(
        "Dual embed error: Non-zero status code returned while running \
         TRTKernel_graph_main_graph_15093723750161443578_0 node. \
         Name:'TensorrtExecutionProvider_TRTKernel_graph_main_graph_\
         15093723750161443578_0_0' Status Message: TensorRT EP failed to \
         create engine from network. Workspace size insufficient for tactic."
    );
    assert!(is_trt_jit_oom(&e));
}

/// Verbatim production error WITHOUT any workspace/alloc/memory/tactic
/// qualifier. Under Option A, this CURRENTLY classifies as non-retryable —
/// see the doc-comment on `is_trt_jit_oom` for the rationale and the open
/// question about whether the TRT root-cause is propagated into the outer
/// error string or only into a separate tracing target.
#[test]
fn is_trt_jit_oom_does_not_match_failed_to_create_engine_without_qualifier() {
    let e = anyhow::anyhow!(
        "Dual embed error: Non-zero status code returned while running \
         TRTKernel_graph_main_graph_15093723750161443578_0 node. \
         Name:'TensorrtExecutionProvider_TRTKernel_graph_main_graph_\
         15093723750161443578_0_0' Status Message: TensorRT EP failed to \
         create engine from network."
    );
    assert!(
        !is_trt_jit_oom(&e),
        "Option A: 'failed to create engine' without a qualifier MUST NOT \
         retry; we accept this slips through until CloudWatch confirms the \
         TRT logger output is propagated into the outer error string"
    );
}

/// `tactic` is a TRT plan-builder term — when present alongside
/// `failed to create engine`, retry-with-halved-workspace is the right move.
#[test]
fn is_trt_jit_oom_matches_failed_to_create_engine_with_tactic() {
    let e = anyhow::anyhow!(
        "TensorRT EP failed to create engine from network. \
         No suitable tactic found for layer /MatMul_42."
    );
    assert!(is_trt_jit_oom(&e));
}

/// `unsupported` (graph-level problem) MUST NOT trigger retry — retry is
/// pointless and doubles caller-visible latency.
#[test]
fn is_trt_jit_oom_does_not_match_failed_to_create_engine_with_unsupported() {
    let e = anyhow::anyhow!(
        "TensorRT EP failed to create engine from network. \
         Unsupported op: ConstantOfShape with bool output."
    );
    assert!(
        !is_trt_jit_oom(&e),
        "'unsupported' is a graph-shape problem, not OOM — must not retry"
    );
}

/// `oom` alone (without `out of memory`) is a recognised qualifier.
/// MUST classify as retryable since `oom` is in the qualifier set.
#[test]
fn is_trt_jit_oom_matches_failed_to_create_engine_with_oom_keyword() {
    let e = anyhow::anyhow!(
        "TensorRT EP failed to create engine from network. \
         CUDA error: oom (device 0)."
    );
    assert!(is_trt_jit_oom(&e));
}

/// `out of memory` is the canonical CUDA OOM string. MUST classify as
/// retryable since `memory` is in the qualifier set.
#[test]
fn is_trt_jit_oom_matches_failed_to_create_engine_with_out_of_memory() {
    let e = anyhow::anyhow!(
        "TensorRT EP failed to create engine from network. \
         CUDA error: out of memory (cudaMalloc returned 2)."
    );
    assert!(is_trt_jit_oom(&e));
}

// --- embed_with_trt_retry ---

#[test]
fn embed_with_trt_retry_succeeds_without_retry() {
    let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
    let calls = std::cell::Cell::new(0u32);
    let result = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            Ok(42u64)
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(result.unwrap(), 42);
    assert_eq!(calls.get(), 1);
}

#[test]
fn embed_with_trt_retry_fires_exactly_once_on_oom() {
    let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
    let calls = std::cell::Cell::new(0u32);
    let result: anyhow::Result<u64> = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(anyhow::anyhow!("User allocator error in TRT"))
            } else {
                Ok(99)
            }
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(result.unwrap(), 99);
    assert_eq!(calls.get(), 2);
}

#[test]
fn embed_with_trt_retry_halves_workspace_on_retry() {
    let original = 2 * 1024 * 1024 * 1024usize;
    let cm = CostModel::conservative(original);
    let observed_workspace = std::cell::Cell::new(0usize);
    let calls = std::cell::Cell::new(0u32);
    let _: anyhow::Result<u64> = embed_with_trt_retry(
        |cm| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(anyhow::anyhow!("User allocator error in TRT"))
            } else {
                observed_workspace.set(cm.max_workspace_bytes);
                Ok(0)
            }
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(observed_workspace.get(), original / 2);
}

#[test]
fn embed_with_trt_retry_propagates_non_oom_error_immediately() {
    let cm = CostModel::conservative(1024);
    let calls = std::cell::Cell::new(0u32);
    let result: anyhow::Result<u64> = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("Some unrelated error"))
        },
        &cm,
        0,
        "test",
    );
    assert!(result.is_err());
    assert_eq!(calls.get(), 1, "non-OOM error must not retry");
}

/// Edge case: `max_workspace_bytes` = 0 halves to 0, then floors to 1 MiB (COR-4).
#[test]
fn embed_with_trt_retry_halved_workspace_floors_at_1_mib_when_original_is_zero() {
    let cm = CostModel::conservative(0);
    let observed_workspace = std::cell::Cell::new(0usize);
    let calls = std::cell::Cell::new(0u32);
    let _: anyhow::Result<u64> = embed_with_trt_retry(
        |cm| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(anyhow::anyhow!("User allocator error in TRT"))
            } else {
                observed_workspace.set(cm.max_workspace_bytes);
                Ok(0)
            }
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(
        observed_workspace.get(),
        1024 * 1024,
        "halved workspace must floor at 1 MiB when original is 0"
    );
}

/// Edge case: `max_workspace_bytes` = 1 halves to 0, then floors to 1 MiB (COR-4).
#[test]
fn embed_with_trt_retry_halved_workspace_floors_at_1_mib_when_original_is_1() {
    let cm = CostModel::conservative(1);
    let observed_workspace = std::cell::Cell::new(0usize);
    let calls = std::cell::Cell::new(0u32);
    let _: anyhow::Result<u64> = embed_with_trt_retry(
        |cm| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Err(anyhow::anyhow!("User allocator error in TRT"))
            } else {
                observed_workspace.set(cm.max_workspace_bytes);
                Ok(0)
            }
        },
        &cm,
        0,
        "test",
    );
    assert_eq!(
        observed_workspace.get(),
        1024 * 1024,
        "halved workspace must floor at 1 MiB when original is 1"
    );
}

#[test]
fn embed_with_trt_retry_propagates_second_failure() {
    let cm = CostModel::conservative(1024);
    let calls = std::cell::Cell::new(0u32);
    let result: anyhow::Result<u64> = embed_with_trt_retry(
        |_cm| {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("User allocator error in TRT"))
        },
        &cm,
        0,
        "test",
    );
    assert!(result.is_err());
    assert_eq!(calls.get(), 2);
}
