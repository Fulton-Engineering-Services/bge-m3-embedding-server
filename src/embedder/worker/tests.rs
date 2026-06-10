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

//! Unit tests for the `worker` module.
//!
//! - `trt_retry`: `is_trt_jit_oom` pattern matching and
//!   `embed_with_trt_retry` retry/workspace-halving logic.
//! - `propagation`: `drain_engine_propagation` deduplication, lagged-channel
//!   handling, and idempotency for already-warmed shapes.
//! - `prewarm_strict`: pure-function tests for the `BGE_M3_PREWARM_STRICT`
//!   decision predicate (`should_fail_readiness`).
//! - `guard`: shape-guard wiring, outcome classification, finalize path.
//! - `logging`: abandoned-request observability.
//! - `inference_complete`: post-inference cache-miss signaling.
//! - `dispatch`: reload-error reply routing and abandonment logging.
//! - `startup`: SM detection helper for TRT workers.

mod dispatch;
mod guard;
mod helpers;
mod inference_complete;
mod logging;
mod prewarm_strict;
mod propagation;
mod startup;
mod trt_retry;
