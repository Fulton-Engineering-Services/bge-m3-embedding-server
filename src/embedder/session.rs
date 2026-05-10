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
use crate::config::ModelVariant;

/// Returns the execution providers to use for this platform.
///
/// On macOS: uses the `CoreML` EP with `MLProgram` format and `FastPrediction`
/// specialisation strategy (overridable via `BGE_M3_COREML_STRATEGY=default`).
/// On all other platforms: returns an empty list, so ORT falls back to MLAS (CPU).
pub(super) fn execution_providers(cache_dir: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(target_os = "macos")]
    {
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
        vec![builder.build()]
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cache_dir;
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
pub(super) fn load_models(
    cache_dir: &Path,
    show_download_progress: bool,
    model_variant: ModelVariant,
    max_seq_length: usize,
    intra_threads: usize,
) -> Result<(ort::session::Session, tokenizers::Tokenizer)> {
    let files = download_model_files(cache_dir, show_download_progress, model_variant)?;
    let tokenizer = load_tokenizer(&files.tokenizer_path, max_seq_length)?;
    let eps = execution_providers(cache_dir);
    let session = load_session(&files.onnx_path, eps, intra_threads)?;
    Ok((session, tokenizer))
}
