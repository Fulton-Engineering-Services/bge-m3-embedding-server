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

//! In-band `TensorRT` JIT admission guard.
//!
//! ## The failure this prevents
//!
//! When a chunk shape `(batch, seq)` reaches `session.run()` that the worker's
//! `TensorRT` engine profile does **not** already cover, the TRT EP compiles
//! an engine for it in-band (in the middle of a real request). On the fused
//! dual-output `/v1/embeddings:both` graph at the maximum sequence length
//! (`seq = 8192`) the kernel autotuner can request *pathological* scratch
//! allocations - tens of gigabytes up to multiple terabytes on a single
//! `LayerNorm + MatMul` foreign node. `BGE_M3_TRT_MAX_WORKSPACE_BYTES` does
//! **not** bound autotuner tactic scratch (a TRT EP limitation), so on a
//! VRAM-saturated device (e.g. the warmup-shard worker already holding the
//! `seq=8192` engines at 90%+ VRAM) the CUDA allocator faults and the process
//! dies via SIGSEGV / OOM-kill **before** any `Result` is returned. None of
//! the existing reactive safety nets (`is_trt_jit_oom` retry, the
//! `is_trt_engine_build_fatal` worker-exit, the circuit breaker) can catch a
//! hard process death - the only defense is to never issue the dangerous run.
//!
//! Startup warmup *catches* a failed compile (`run_warmup_shape` logs a WARN
//! and continues) so a worker whose `seq=8192` shard failed to compile still
//! signals ready. The first real `seq≈8192` request then triggers the same
//! pathological allocation in-band, without warmup's caught-error safety net.
//!
//! ## The guard
//!
//! [`TrtJitGuard`] refuses - with a clean, retriable error that maps to HTTP
//! `503` - any chunk whose sequence length is in the dangerous range
//! (`seq >= guard_seq`) and is **not** already covered by the pool's warmed
//! engine profile (`seq > warmed_seq_ceiling`). Refusing one request is
//! strictly better than a SIGSEGV that kills every in-flight request on the
//! worker and forces an ECS task replacement.
//!
//! `warmed_seq_ceiling` is the maximum sequence length **any** worker in the
//! pool successfully warmed (fresh compile or warm-cache hit), shared via an
//! `AtomicUsize`. Because TRT engine plans live on the shared EFS cache and a
//! single profile-based engine file spans `[min_seq, max_seq]` across every
//! shape compiled to it, a successful warmup of `seq=8192` by *any* worker
//! means *every* worker can fast-load (not JIT) that shape - so the ceiling is
//! a sound pool-wide coverage signal. Conversely, if the `seq=8192` shard
//! failed on every worker, no plan exists on disk, the ceiling stays at the
//! highest tier that *did* compile (e.g. 2048), and `seq=8192` requests are
//! refused instead of crashing the process.
//!
//! ## Why sequence length (not batch) is the discriminator
//!
//! The pathological allocation scales with the attention score matrix
//! (`O(batch · seq^2)`), which is dominated by `seq` at the top tier. Within a
//! compiled profile, intermediate batches are covered by the engine's
//! `[min_batch, max_batch]` range, and `bin_pack` already bounds the per-chunk
//! batch under the workspace budget (so `seq=8192` chunks never exceed
//! ~15-18 texts). The only reachable uncovered-and-dangerous region is "a
//! sequence length tier that warmup failed to compile", which is exactly what
//! the ceiling tracks.
//!
//! ## Self-healing
//!
//! The adaptive-warmup loop and cross-worker engine propagation both raise the
//! ceiling (via [`fetch_max`](std::sync::atomic::AtomicUsize::fetch_max)) when
//! they successfully compile a higher tier during an idle window, so coverage
//! that was refused at startup is admitted again once a plan lands on disk.

use std::fmt;

/// Error returned when [`TrtJitGuard`] refuses a chunk to avoid a pathological
/// in-band `TensorRT` JIT compile.
///
/// Carries the offending `(batch, seq)` and the coverage parameters so the
/// refusal is fully diagnosable in logs. Maps to HTTP `503 Service
/// Unavailable` (see `crate::error::AppError`'s `From<anyhow::Error>` impl):
/// the request is *retriable* - coverage may extend via adaptive warmup, or a
/// peer task may already cover the shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrtJitRejection {
    /// Chunk batch size (number of texts in the refused `session.run()` call).
    pub batch: usize,
    /// Chunk (padded) sequence length that triggered the refusal.
    pub seq: usize,
    /// `guard_seq` threshold in effect at refusal time.
    pub guard_seq: usize,
    /// Pool-wide max successfully-warmed sequence length at refusal time.
    pub warmed_seq_ceiling: usize,
}

impl fmt::Display for TrtJitRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "refusing in-band TensorRT JIT for chunk (batch={}, seq={}): \
             seq is at/above the guard threshold ({}) and exceeds the pool's \
             warmed engine coverage (max warmed seq={}). Issuing this run risks \
             a pathological autotuner allocation that can crash the worker. \
             The request is retriable once warmup coverage extends.",
            self.batch, self.seq, self.guard_seq, self.warmed_seq_ceiling
        )
    }
}

impl std::error::Error for TrtJitRejection {}

/// Per-request snapshot of the in-band JIT admission policy.
///
/// Cheap to construct (two `usize`s); the worker builds one per request from
/// the live `warmed_seq_ceiling` atomic so the decision always reflects the
/// latest pool-wide coverage. A `None` guard at the call sites disables
/// checking entirely (non-TRT EPs, or `BGE_M3_TRT_INBAND_JIT_GUARD=0`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrtJitGuard {
    guard_seq: usize,
    warmed_seq_ceiling: usize,
}

impl TrtJitGuard {
    /// Builds a guard from the danger threshold and the current pool-wide
    /// warmed-sequence ceiling.
    #[must_use]
    pub(crate) fn new(guard_seq: usize, warmed_seq_ceiling: usize) -> Self {
        Self {
            guard_seq,
            warmed_seq_ceiling,
        }
    }

    /// Decides whether a single chunk `(batch, seq)` may be dispatched to
    /// `session.run()`.
    ///
    /// Refuses iff the sequence length is in the dangerous range
    /// (`seq >= guard_seq`) **and** is not covered by the warmed profile
    /// (`seq > warmed_seq_ceiling`). Everything else is admitted:
    ///
    /// * `seq < guard_seq` - below the dangerous tier; a cold JIT here is
    ///   bounded and lets the profile grow naturally.
    /// * `seq <= warmed_seq_ceiling` - covered by an existing engine plan;
    ///   the run is a cache hit / fast disk-load, never a pathological JIT.
    pub(crate) fn admit(&self, batch: usize, seq: usize) -> Result<(), TrtJitRejection> {
        if seq >= self.guard_seq && seq > self.warmed_seq_ceiling {
            return Err(TrtJitRejection {
                batch,
                seq,
                guard_seq: self.guard_seq,
                warmed_seq_ceiling: self.warmed_seq_ceiling,
            });
        }
        Ok(())
    }
}

/// Validates every chunk produced by `bin_pack` against an optional guard.
///
/// Returns `Err(TrtJitRejection)` for the **first** chunk that would trigger a
/// dangerous in-band JIT, so the whole request is refused atomically before any
/// `session.run()` executes (no partially-computed output). A `None` guard
/// admits everything.
///
/// `chunks` are the original-index groups returned by
/// [`crate::binpack::bin_pack`]; `seq_lens` is the per-text tokenized length
/// (indexed by original position). The per-chunk shape is
/// `(chunk.len(), max(seq_lens[i] for i in chunk))`.
pub(crate) fn guard_chunks(
    guard: Option<&TrtJitGuard>,
    chunks: &[Vec<usize>],
    seq_lens: &[usize],
) -> Result<(), TrtJitRejection> {
    let Some(guard) = guard else {
        return Ok(());
    };
    for chunk in chunks {
        let chunk_seq = chunk.iter().map(|&i| seq_lens[i]).max().unwrap_or(0);
        guard.admit(chunk.len(), chunk_seq)?;
    }
    Ok(())
}

/// Returns `true` when `err` (or anything in its source chain) is a
/// [`TrtJitRejection`].
///
/// The worker uses this to (a) skip the inference circuit breaker for guard
/// refusals - a refusal means the worker is *healthy* and deliberately
/// protecting itself, not a failing GPU - and (b) the HTTP layer uses the same
/// chain walk to map the refusal to `503` rather than `500`. Walking the chain
/// (not just the top error) lets callers wrap the rejection with
/// `anyhow::Error::context` for additional log context without breaking
/// detection.
pub(crate) fn is_trt_shape_rejected(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<TrtJitRejection>().is_some())
}

#[cfg(test)]
mod tests;
