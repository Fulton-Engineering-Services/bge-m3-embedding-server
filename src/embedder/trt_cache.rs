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

//! `TensorRT` engine cache path construction, inspection, and durability.
//!
//! Why a dedicated module? Investigation of a production incident
//! showed two consecutive cold starts producing identical 172 s recompile
//! times for `1×8192` — the `{cache_dir}/trt-engines/` directory on EFS was
//! NOT being reused between container restarts. The most plausible root cause
//! is that ECS SIGKILL on OOM (`exitCode 137`) interrupts the container before
//! the kernel's writeback timer flushes buffered writes to EFS. The EFS inode
//! lists the engine files, but their data blocks are zero-length or partial,
//! so ORT/TRT silently treats them as cache misses and rebuilds from scratch.
//!
//! This module:
//! 1. Constructs the cache directory path (stable, no per-container ephemera).
//! 2. Inspects the directory at startup so the operator-visible INFO log shows
//!    "found N cached engines" or "empty (will compile)" — without this we
//!    cannot tell from `CloudWatch` whether the EFS mount is actually persisting.
//! 3. Exposes an explicit `fsync_cache_dir` that flushes both file data and
//!    directory metadata to disk, called after each successful TRT engine
//!    compile so an OOM-kill never strands a partially-written engine plan.
//!
//! TRT plan files embed `(GPU compute capability, CUDA version, TRT version,
//! ONNX model SHA, builder config)`. Within a homogeneous ASG (same instance
//! family, same AMI) these are stable, so the cache is reusable per-EC2-host.
//! ASGs that mix instance families (T4 → A10G) will see expected cache misses
//! when a task lands on a different GPU architecture.
//!
//! Several items below are referenced only from `session.rs` under
//! `#[cfg(all(not(target_os = "macos"), feature = "tensorrt"))]` — the CPU /
//! macOS build legitimately never calls them. The `#[allow(dead_code)]`
//! attribute below silences the resulting unused-warning under those builds;
//! the unit tests in this module keep the items exercised on every CI target.
//!
//! Submodules:
//! - `paths`: cache directory paths and [`TrtCacheInfo`].
//! - `inspect`: startup inspection and EFS write-probe.
//! - `enumerate`: SM-aware engine plan enumeration and counting.
//! - `prewarm_log`: operator-visible prewarm basename logging.
//! - `fsync`: post-compile cache durability.

#![allow(dead_code, unused_imports)]

mod enumerate;
mod fsync;
mod inspect;
mod paths;
mod prewarm_log;

#[cfg(test)]
mod tests;

pub(crate) use enumerate::{
    count_engine_files, count_engine_files_for_sm, engine_basenames_for_sm,
    engine_basenames_in_dir_sorted, engine_files_for_sm, matches_sm_suffix,
};
pub(crate) use fsync::fsync_cache_dir;
pub(crate) use inspect::{ensure_and_inspect, log_cache_state};
pub(crate) use paths::{
    TRT_ENGINE_SUBDIR, TRT_TIMING_SUBDIR, TrtCacheInfo, engine_cache_path, timing_cache_path,
};
pub(crate) use prewarm_log::{
    log_engine_basenames_before_prewarm, log_engine_basenames_before_prewarm_for_sm,
};
