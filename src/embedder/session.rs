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

//! ORT execution-provider configuration and session loading.

use std::path::Path;

use anyhow::Result;

use super::error::ort_err;
use super::model_files::download_model_files;
use super::tokenize::load_tokenizer;
#[cfg(all(not(target_os = "macos"), feature = "tensorrt"))]
use super::trt_cache;
use crate::config::{EpSelection, ModelVariant};

/// Returns the execution providers to use for this platform and EP selection.
///
/// On macOS: always uses the `CoreML` EP with `MLProgram` format and
/// `FastPrediction` specialisation strategy (overridable via
/// `BGE_M3_COREML_STRATEGY=default`), regardless of `ep`.
///
/// On Linux with the `tensorrt` feature: selects `TensorRT` when
/// `ep == EpSelection::TensorRt`, with engine caching, FP16, and the
/// specified `device_id` enabled. When `trt_max_workspace_bytes` is `Some`,
/// the workspace cap is forwarded to the TRT EP via `with_max_workspace_size`;
/// otherwise ORT's built-in default is used.
///
/// On Linux with the `cuda` feature: selects CUDA when
/// `ep == EpSelection::Cuda`, pinned to `device_id`. When
/// `gpu_mem_limit_bytes` is `Some`, the device memory limit is forwarded via
/// `with_memory_limit`; otherwise the EP uses all available device memory.
///
/// CPU fallback: returns an empty list so ORT falls back to MLAS.
///
/// `device_id` is computed by `EmbedPool::spawn` as
/// `worker_index % gpu_count` and is ignored on CPU EP and macOS.
///
/// Emits a single `INFO` log line tagged `"ORT execution providers configured"`
/// describing the configured EP, the EP that was actually built (the "active"
/// EP), and any cache paths handed to it. This is the source of truth in
/// `CloudWatch` for "is `TensorRT` really active or did we silently fall back?".
pub(super) fn execution_providers(
    cache_dir: &Path,
    ep: EpSelection,
    device_id: u32,
    trt_max_workspace_bytes: Option<usize>,
    gpu_mem_limit_bytes: Option<usize>,
) -> Vec<ort::ep::ExecutionProviderDispatch> {
    // macOS: always CoreML regardless of BGE_M3_EP.
    // The cfg blocks are mutually exclusive so only one branch is compiled per target.
    #[cfg(target_os = "macos")]
    {
        let _ = device_id;
        let _ = (trt_max_workspace_bytes, gpu_mem_limit_bytes);
        let coreml_cache = cache_dir.join("coreml");
        let strategy = match std::env::var("BGE_M3_COREML_STRATEGY").ok().as_deref() {
            Some("default") => ort::ep::coreml::SpecializationStrategy::Default,
            _ => ort::ep::coreml::SpecializationStrategy::FastPrediction,
        };
        let builder = ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_specialization_strategy(strategy)
            .with_model_cache_dir(coreml_cache.display().to_string());
        #[cfg(feature = "coreml-profile")]
        let builder = builder.with_profile_compute_plan(true);
        tracing::info!(
            ep_selection = %ep,
            ep_active = "CoreML",
            coreml_cache_path = %coreml_cache.display(),
            "ORT execution providers configured"
        );
        vec![builder.build()]
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Linux TensorRT (feature-gated).
        // ort 2.0.0-rc.12 uses `with_engine_cache` / `with_fp16` — not
        // `with_engine_cache_enable` / `with_fp16_enable` which don't exist.
        #[cfg(feature = "tensorrt")]
        if ep == EpSelection::TensorRt {
            // Inspect the cache directory BEFORE handing the path to ORT so
            // operators see in CloudWatch whether engine reuse is working.
            // Two consecutive cold starts producing the same compile time is
            // the symptom of an EFS mount that isn't actually persisting —
            // surfacing the count here is the fastest way to diagnose it.
            let cache_info = trt_cache::ensure_and_inspect(cache_dir);
            trt_cache::log_cache_state(&cache_info);

            let timing_cache = trt_cache::timing_cache_path(cache_dir);
            // The timing cache stores per-tactic kernel timings so the TRT
            // builder can skip the tactic-selection step on each subsequent
            // engine build. It is complementary to the engine cache — even
            // a cold engine cache benefits from a warm timing cache when
            // multiple shapes are compiled in the same warmup sweep.
            tracing::info!(
                ep_selection = %ep,
                ep_active = "TensorRT",
                device_id,
                engine_cache_path = %cache_info.path.display(),
                timing_cache_path = %timing_cache.display(),
                fp16 = true,
                error_on_failure = true,
                "ORT execution providers configured"
            );
            // `.error_on_failure()` upgrades the default silent-CPU-fallback
            // path to a hard error. ORT's `apply_execution_providers` defaults
            // `error_on_failure = false`, which means a failed registration
            // (e.g. `libonnxruntime_providers_tensorrt.so` missing from the
            // image — the 2026-05 codekeeper outage root cause) is logged as
            // a `WARN`/`ERROR` via the `ort` crate's internal tracing macros
            // and the loop falls back to CPU/MLAS without surfacing the
            // failure. With this set, `Session::builder().with_execution_providers(...)`
            // returns the error verbatim, which `load_session` already
            // converts into a worker-load failure — the worker exits non-zero
            // instead of silently serving CPU inference. Greppable in
            // CloudWatch via the new field `error_on_failure: true` on the
            // "ORT execution providers configured" event.
            let mut trt_ep = ort::ep::TensorRT::default()
                .with_device_id(device_id.cast_signed())
                .with_engine_cache(true)
                .with_engine_cache_path(cache_info.path.display().to_string())
                .with_timing_cache(true)
                .with_timing_cache_path(timing_cache.display().to_string())
                .with_fp16(true);
            if let Some(cap) = trt_max_workspace_bytes {
                trt_ep = trt_ep.with_max_workspace_size(cap);
            }
            return vec![trt_ep.build().error_on_failure()];
        }

        // Linux CUDA (feature-gated).
        #[cfg(feature = "cuda")]
        if ep == EpSelection::Cuda {
            let _ = cache_dir;
            tracing::info!(
                ep_selection = %ep,
                ep_active = "CUDA",
                device_id,
                error_on_failure = true,
                "ORT execution providers configured"
            );
            let mut cuda_ep = ort::ep::CUDA::default().with_device_id(device_id.cast_signed());
            if let Some(limit) = gpu_mem_limit_bytes {
                cuda_ep = cuda_ep.with_memory_limit(limit);
            }
            return vec![cuda_ep.build().error_on_failure()];
        }

        // CPU fallback (always available).
        let _ = (
            cache_dir,
            device_id,
            trt_max_workspace_bytes,
            gpu_mem_limit_bytes,
        );
        tracing::info!(
            ep_selection = %ep,
            ep_active = "CPU/MLAS",
            "ORT execution providers configured"
        );
        vec![]
    }
}

/// Builds an ORT session from the ONNX model file with the given execution providers.
///
/// `intra_threads` controls intra-op parallelism for matmul / attention kernels
/// inside a single `session.run()` call. The default (`1`) keeps per-worker RSS
/// predictable for the workspace probe; raise it to `floor(num_cpus / workers)`
/// on under-utilized hosts to recover CPU headroom. See
/// [`crate::config::Config::intra_threads`] for the operator-facing knob.
pub(super) fn load_session(
    model_path: &Path,
    eps: Vec<ort::ep::ExecutionProviderDispatch>,
    intra_threads: usize,
) -> Result<ort::session::Session> {
    let mut builder = ort::session::Session::builder().map_err(ort_err)?;
    if !eps.is_empty() {
        builder = builder.with_execution_providers(eps).map_err(ort_err)?;
    }
    let session = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(intra_threads.max(1))
        .map_err(ort_err)?
        .commit_from_file(model_path)
        .map_err(ort_err)?;
    Ok(session)
}

/// Downloads (if not already cached) and loads both the ORT session and the
/// tokenizer for the given model variant, returning them as a pair.
///
/// `device_id` selects the CUDA/TRT GPU device for this session. Computed by
/// `EmbedPool::spawn` as `worker_index % gpu_count`. Ignored on CPU EP and
/// macOS.
///
/// `trt_max_workspace_bytes` and `gpu_mem_limit_bytes` are forwarded verbatim
/// to [`execution_providers`]; see that function's documentation for semantics.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_models(
    cache_dir: &Path,
    show_download_progress: bool,
    model_variant: ModelVariant,
    max_seq_length: usize,
    intra_threads: usize,
    ep: EpSelection,
    device_id: u32,
    trt_max_workspace_bytes: Option<usize>,
    gpu_mem_limit_bytes: Option<usize>,
) -> Result<(ort::session::Session, tokenizers::Tokenizer)> {
    let files = download_model_files(cache_dir, show_download_progress, model_variant)?;
    let tokenizer = load_tokenizer(&files.tokenizer_path, max_seq_length)?;
    let eps = execution_providers(
        cache_dir,
        ep,
        device_id,
        trt_max_workspace_bytes,
        gpu_mem_limit_bytes,
    );
    let session = load_session(&files.onnx_path, eps, intra_threads)?;
    Ok((session, tokenizer))
}
