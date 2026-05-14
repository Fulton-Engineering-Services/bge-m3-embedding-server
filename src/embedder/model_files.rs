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

/// Paths to the ONNX model and tokenizer files resolved from the hf-hub cache.
pub(super) struct ModelFiles {
    /// Path to the ONNX model file (variant-specific).
    pub onnx_path: PathBuf,
    /// Path to the `tokenizer.json` file.
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

/// Downloads (or retrieves from the local hf-hub snapshot cache) the ONNX model
/// and tokenizer files for the given model variant.
///
/// `show_progress` enables hf-hub's download progress bar; pass `true` only for
/// the leader worker (worker 0) so progress is shown exactly once.
pub(super) fn download_model_files(
    cache_dir: &Path,
    show_progress: bool,
    variant: ModelVariant,
) -> Result<ModelFiles> {
    // Fail fast if the cache directory is structurally invalid (e.g. a path
    // component is a regular file or a non-directory device, the parent is
    // read-only, or the operator pointed `BGE_M3_CACHE_DIR` at something we
    // can never write to). `create_dir_all` is idempotent on an already-valid
    // directory, so this is a no-op on a healthy production setup. The check
    // is cheap and runs before any network syscall.
    //
    // Without this check, `hf_hub::ApiBuilder` defers cache validation until
    // mid-download — after a `metadata()` HTTP round-trip that has *no*
    // default ureq timeout. On a runner with a misconfigured cache dir AND
    // unreliable IPv6 connectivity (notably GitHub Actions), the connect
    // call to huggingface.co blocks indefinitely instead of letting the
    // doomed mkdir surface as the actual cause. That hang reaches all the
    // way up to `EmbedPool::spawn`'s init task, which never sees a ready
    // signal — manifesting as the spawn-tests timeout on CI.
    std::fs::create_dir_all(cache_dir).map_err(|e| {
        anyhow::anyhow!(
            "Cannot create or access model cache directory {}: {e}",
            cache_dir.display()
        )
    })?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelVariant;

    /// `download_model_files` must fail synchronously — without ever issuing a
    /// network request — when the cache directory is structurally impossible
    /// to create. `/dev/null/impossible` is the canonical bad-cache fixture
    /// used by `EmbedPool::spawn` tests: `/dev/null` is a character device on
    /// every Unix, so `mkdir /dev/null/impossible` reliably returns ENOTDIR.
    ///
    /// Regression guard: before the upfront `create_dir_all` validation,
    /// hf-hub's lazy cache layout meant this code path executed a metadata
    /// HTTP call to huggingface.co first and only attempted the doomed mkdir
    /// inside the download flow. On runners with no IPv6 connectivity and
    /// ureq's default `None` connect timeout, that call blocked indefinitely
    /// and the leader-failure spawn tests timed out on CI.
    #[cfg(unix)]
    #[test]
    fn download_model_files_fails_fast_on_unwritable_cache_dir() {
        let bad = Path::new("/dev/null/impossible");
        let started = std::time::Instant::now();
        let result = download_model_files(bad, false, ModelVariant::Fp32);
        let elapsed = started.elapsed();
        let Err(err) = result else {
            panic!("expected Err for an unwritable cache dir, got Ok");
        };
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "validation should fail without a network round-trip; took {elapsed:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("cache directory"),
            "error should mention the cache directory; got: {msg}"
        );
    }
}
