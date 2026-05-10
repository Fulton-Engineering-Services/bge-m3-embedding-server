//! `HuggingFace` Hub download + cache-layout helpers for the BGE-M3 model files.

use std::path::{Path, PathBuf};

use anyhow::Result;
use tracing::info;

use crate::config::ModelVariant;

const REPO_ID: &str = "BAAI/bge-m3";
/// Pinned HF commit — prevents silent model updates and provides supply-chain
/// integrity for the ONNX weights and tokenizer. Update this hash intentionally
/// after verifying a new revision produces equivalent embeddings.
const REPO_REVISION: &str = "5617a9f61b028005a4858fdac845db406aefb181";

const XENOVA_REPO_ID: &str = "Xenova/bge-m3";
/// Pinned HF commit for the Xenova/bge-m3 FP16 (~1.08 GB) and INT8 (~568 MB) models.
/// Update intentionally after verifying equivalent embedding quality vs FP32.
const XENOVA_REPO_REVISION: &str = "4de13258303883538bd53b696b452bf8099f0858";

pub(super) struct ModelFiles {
    pub onnx_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

/// Returns `true` when the primary ONNX model file already exists in the
/// hf-hub snapshot cache, meaning `repo.get()` will return immediately
/// without fetching from the network.
///
/// hf-hub 0.5.x layout when constructed with `ApiBuilder::with_cache_dir(p)`:
/// `{p}/models--{owner}--{name}/snapshots/{revision}/{filename}`
///
/// Note: this differs from Python `huggingface_hub`, which appends a `hub/`
/// segment when `HF_HOME` is set. The Rust crate treats `with_cache_dir`
/// as `HF_HUB_CACHE` directly — no `hub/` subdirectory is added.
fn is_model_cached(cache_dir: &Path, repo_id: &str, revision: &str, onnx_filename: &str) -> bool {
    let repo_dir = format!("models--{}", repo_id.replace('/', "--"));
    cache_dir
        .join(repo_dir)
        .join("snapshots")
        .join(revision)
        .join(onnx_filename)
        .exists()
}

pub(super) fn download_model_files(
    cache_dir: &Path,
    show_progress: bool,
    variant: ModelVariant,
) -> Result<ModelFiles> {
    let (repo_id, repo_revision) = match variant {
        ModelVariant::Fp32 => (REPO_ID, REPO_REVISION),
        ModelVariant::Fp16 | ModelVariant::Int8 => (XENOVA_REPO_ID, XENOVA_REPO_REVISION),
    };

    // Check the hf-hub snapshot directory for the primary ONNX file before
    // touching the network.  This lets us log a clear "from cache" message
    // rather than silence while hf-hub resolves files.
    let onnx_filename = match variant {
        ModelVariant::Fp32 => "onnx/model.onnx",
        ModelVariant::Fp16 => "onnx/model_fp16.onnx",
        ModelVariant::Int8 => "onnx/model_int8.onnx",
    };
    let cached = is_model_cached(cache_dir, repo_id, repo_revision, onnx_filename);
    if cached {
        info!(
            repo_id,
            revision = repo_revision,
            model_variant = %variant,
            "Model files found in local cache — no download needed"
        );
    } else {
        info!(
            repo_id,
            revision = repo_revision,
            model_variant = %variant,
            "Model files not in local cache — downloading from HuggingFace Hub"
        );
    }

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(show_progress)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build hf-hub API: {e}"))?;

    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        repo_revision.to_string(),
    ));

    let onnx_path = match variant {
        ModelVariant::Fp32 => {
            let path = repo
                .get("onnx/model.onnx")
                .map_err(|e| anyhow::anyhow!("Failed to get onnx/model.onnx: {e}"))?;
            repo.get("onnx/model.onnx_data")
                .map_err(|e| anyhow::anyhow!("Failed to get onnx/model.onnx_data: {e}"))?;
            repo.get("onnx/Constant_7_attr__value")
                .map_err(|e| anyhow::anyhow!("Failed to get onnx/Constant_7_attr__value: {e}"))?;
            path
        }
        ModelVariant::Fp16 => repo
            .get("onnx/model_fp16.onnx")
            .map_err(|e| anyhow::anyhow!("Failed to get onnx/model_fp16.onnx: {e}"))?,
        ModelVariant::Int8 => repo
            .get("onnx/model_int8.onnx")
            .map_err(|e| anyhow::anyhow!("Failed to get onnx/model_int8.onnx: {e}"))?,
    };

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to get tokenizer.json: {e}"))?;

    Ok(ModelFiles {
        onnx_path,
        tokenizer_path,
    })
}
