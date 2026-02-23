use fastembed::{SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding};
use std::path::Path;

pub struct Embedder {
    model: SparseTextEmbedding,
}

impl Embedder {
    pub fn new(cache_dir: &Path) -> anyhow::Result<Self> {
        let model = SparseTextEmbedding::try_new(
            SparseInitOptions::new(SparseModel::BGEM3)
                .with_cache_dir(cache_dir.to_path_buf()),
        )?;
        Ok(Self { model })
    }

    pub fn embed(&mut self, texts: Vec<String>) -> anyhow::Result<Vec<SparseEmbedding>> {
        Ok(self.model.embed(texts, None)?)
    }
}
