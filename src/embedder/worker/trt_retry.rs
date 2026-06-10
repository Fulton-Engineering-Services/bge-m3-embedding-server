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

//! TRT JIT-OOM detection and single retry with halved workspace budget.

use crate::binpack::CostModel;

/// Mirrors `trt_warmup::CACHE_HIT_THRESHOLD_MS`.  Used to classify a
/// per-request inference as a probable TRT engine cache miss so the
/// adaptive warmup task can proactively compile the engine.
pub(super) const CHUNK_CACHE_HIT_THRESHOLD_MS: u64 = 5_000;

/// Returns `true` when an ORT error string indicates a TRT JIT workspace
/// overflow — a condition that may resolve with a smaller batch or halved
/// workspace budget.
///
/// Patterns verified against ORT 2.0.0-rc.12 `TensorRT` EP
/// (`ort/src/ep/tensorrt.rs`). Re-verify on every ORT version bump.
///
/// # Patterns matched
///
/// 1. **`user allocator error`** — direct CUDA allocation failure surfaced
///    by ORT's user-allocator shim during TRT kernel autotuning.
/// 2. **`could not find any implementation` + (`workspace` | `alloc`)** —
///    TRT kernel-autotuner declared no tactic fits, *and* the qualifier
///    confirms the cause is allocation-driven (otherwise this string also
///    matches genuine unsupported-op cases where retry is pointless).
/// 3. **`failed to create engine` + (`workspace` | `alloc` | `memory` |
///    `oom` | `tactic`)** — TRT EP build-time failure observed in
///    production on large fused-route requests (e.g. `/v1/embeddings:both`
///    with high batch and token counts). The qualifier is mandatory: without
///    it, this same family also
///    covers unsupported-op and corrupted-cache cases where retrying
///    with a halved workspace is pointless and doubles caller-visible
///    latency. `alloc` subsumes `cuMemAlloc`; `memory` subsumes
///    `out of memory`.
///
/// # Known gap
///
/// The verbatim production error string often does NOT include any
/// qualifier — the TRT logger appears to emit workspace/alloc detail to a
/// separate tracing target rather than propagating it into the outer
/// `Status Message`. We chose **Option A** here (require a qualifier) over
/// **Option B** (retry every `failed to create engine` unconditionally) so
/// we don't regress `does_not_match_unsupported`-style cases. If a follow-up
/// `CloudWatch` investigation confirms the TRT root-cause is reliably
/// surfaced only in a sibling `target=ort` event and never in the embed
/// error string, we may need to relax this to Option B (or pipe the TRT
/// logger output into the embed error chain). Until then, this function
/// will continue to return `false` for the verbatim production message and
/// callers will see HTTP 500 on first build failure.
///
/// Tracking: <https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/issues/78>
pub(super) fn is_trt_jit_oom(e: &anyhow::Error) -> bool {
    let s = format!("{e}");
    let lowercase = s.to_lowercase();
    // "User allocator error" = direct CUDA allocation failure during TRT kernel autotuning.
    // "Could not find any implementation" qualifies only when the underlying cause is an
    // allocation failure (workspace or alloc in the message); without this qualifier it also
    // matches genuine unsupported-layer errors where retry is pointless and doubles latency.
    // "Failed to create engine" qualifies only when paired with a workspace/alloc/memory/oom/
    // tactic keyword for the same reason — see the doc-comment above.
    lowercase.contains("user allocator error")
        || (lowercase.contains("could not find any implementation")
            && (lowercase.contains("workspace") || lowercase.contains("alloc")))
        || (lowercase.contains("failed to create engine")
            && (lowercase.contains("workspace")
                || lowercase.contains("alloc")
                || lowercase.contains("memory")
                || lowercase.contains("oom")
                || lowercase.contains("tactic")))
}

/// Returns `true` when an ORT error string indicates a TRT engine build
/// failure severe enough that the worker should exit rather than retry.
///
/// Patterns matched:
///
/// 1. **`failed to build engine`** — top-level TRT engine build failure,
///    typically produced by `IBuilder::buildSerializedNetwork` on a corrupted
///    CUDA context or builder network state.
/// 2. **`failed to create engine from network`** — TRT network-level
///    builder failure (distinct from the per-kernel `failed to create engine`
///    OOM messages that `is_trt_jit_oom` already catches). This pattern
///    indicates the TRT engine builder itself is in an unrecoverable state;
///    halving the workspace and retrying will not help.
///
/// Unlike [`is_trt_jit_oom`], these patterns are matched **without** an
/// additional qualifier because they refer to different failure modes.
/// `failed to create engine from network` is always fatal regardless of
/// the surrounding context. `failed to build engine` may overlap with the
/// OOM retry patterns; we detect it here only when `is_trt_jit_oom` has
/// already returned `false` (and the retry was therefore skipped).
///
/// When this function returns `true` from within `run_worker`, the worker
/// exits immediately (returns `Err`), causing `WorkerGuard` to decrement
/// `live_workers`. ECS replaces the task once all workers have exited,
/// resetting the CUDA driver state.
///
/// The unqualified `failed to create engine from network` pattern is included
/// without an OOM qualifier because it has been observed in practice as the
/// terminal error after the TRT autotuner exhausts its tactic candidates —
/// typically because of a pathological scratch-buffer allocation request that
/// the CUDA allocator cannot satisfy. The TRT EP's `trt_max_workspace_bytes`
/// option does not bound autotuner tactic scratch, so a per-tactic allocation
/// of many gigabytes (or even terabytes) on a fused multi-precision foreign
/// node is reachable for shapes outside the pre-warmed engine cache. Once
/// this pattern fires, the CUDA context is considered unrecoverable for the
/// lifetime of the process; keep this function's pattern set minimal and
/// explicit and do not fold it into `is_trt_jit_oom`.
pub(super) fn is_trt_engine_build_fatal(e: &anyhow::Error) -> bool {
    // `{e:#}` renders the full anyhow source chain (context + cause), so this
    // detection still fires when the caller has wrapped the underlying ORT
    // error with `anyhow::Error::context(...)` (as `run_worker` does, e.g.
    // "Dual embed error: <original>"). A plain `{e}` would show only the
    // outermost context string and miss the build-failure substring.
    let lowercase = format!("{e:#}").to_lowercase();
    lowercase.contains("failed to build engine")
        || lowercase.contains("failed to create engine from network")
}
/// Wraps an embed call with the standard TRT JIT-OOM retry-once-with-halved-budget
/// pattern.
///
/// If `embed_fn` fails and [`is_trt_jit_oom`] matches the error, retries once
/// with `max_workspace_bytes / 2`. Logs `trt_jit_retry` on the first attempt and
/// `trt_jit_retry_exhausted` when the retry also fails. Returns the final result.
pub(super) fn embed_with_trt_retry<T, F>(
    mut embed_fn: F,
    base_cm: &CostModel,
    worker_id: usize,
    route: &'static str,
) -> anyhow::Result<T>
where
    F: FnMut(&CostModel) -> anyhow::Result<T>,
{
    match embed_fn(base_cm) {
        Ok(v) => Ok(v),
        Err(e) if is_trt_jit_oom(&e) => {
            let halved = CostModel {
                // Floor at 1 MiB to prevent integer-division from reaching 0
                // when max_workspace_bytes is very small (e.g. in tests).
                max_workspace_bytes: (base_cm.max_workspace_bytes / 2).max(1024 * 1024),
                ..*base_cm
            };
            tracing::warn!(
                worker_id,
                route,
                original_workspace_mb = base_cm.max_workspace_bytes / (1024 * 1024),
                halved_workspace_mb = halved.max_workspace_bytes / (1024 * 1024),
                error = %e,
                "trt_jit_retry"
            );
            embed_fn(&halved).map_err(|e2| {
                tracing::error!(
                    worker_id,
                    route,
                    error = %e2,
                    "trt_jit_retry_exhausted"
                );
                e2
            })
        }
        Err(e) => Err(e),
    }
}
