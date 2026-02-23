use anyhow::Result;
use fastembed::{
    EmbeddingModel, SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding,
    TextEmbedding, TextInitOptions,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{info, info_span, Instrument};

pub enum EmbedRequest {
    Dense {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    Sparse {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<SparseEmbedding>>>,
    },
}

#[derive(Clone)]
pub struct EmbedPool {
    tx: mpsc::Sender<EmbedRequest>,
}

impl EmbedPool {
    pub fn spawn(n: usize, cache_dir: PathBuf) -> (Self, JoinHandle<Result<()>>) {
        let capacity = n * 4;
        let (tx, rx) = mpsc::channel::<EmbedRequest>(capacity);
        let rx = Arc::new(Mutex::new(rx));

        let init_handle = tokio::task::spawn(
            async move {
                let mut handles = Vec::with_capacity(n);

                for id in 0..n {
                    let rx_clone = Arc::clone(&rx);
                    let cache_dir_clone = cache_dir.clone();

                    let handle = tokio::task::spawn_blocking(move || {
                        let span = info_span!("worker", id = id);
                        let _guard = span.enter();

                        info!("Loading dense model (worker {id})...");
                        let mut dense_model = TextEmbedding::try_new(
                            TextInitOptions::new(EmbeddingModel::BGEM3)
                                .with_cache_dir(cache_dir_clone.clone())
                                .with_show_download_progress(id == 0),
                        )
                        .map_err(|e| anyhow::anyhow!("Failed to load dense model: {e}"))?;

                        info!("Loading sparse model (worker {id})...");
                        let mut sparse_model = SparseTextEmbedding::try_new(
                            SparseInitOptions::new(SparseModel::BGEM3)
                                .with_cache_dir(cache_dir_clone)
                                .with_show_download_progress(false),
                        )
                        .map_err(|e| anyhow::anyhow!("Failed to load sparse model: {e}"))?;

                        info!("Worker {id} models loaded, entering request loop");

                        let rt = tokio::runtime::Handle::current();
                        loop {
                            let request = rt.block_on(async { rx_clone.lock().await.recv().await });

                            match request {
                                None => {
                                    info!("Worker {id} channel closed, shutting down");
                                    break;
                                }
                                Some(EmbedRequest::Dense { texts, reply }) => {
                                    let result = dense_model
                                        .embed(texts, None)
                                        .map_err(|e| anyhow::anyhow!("Dense embed error: {e}"));
                                    let _ = reply.send(result);
                                }
                                Some(EmbedRequest::Sparse { texts, reply }) => {
                                    let result = sparse_model
                                        .embed(texts, None)
                                        .map_err(|e| anyhow::anyhow!("Sparse embed error: {e}"));
                                    let _ = reply.send(result);
                                }
                            }
                        }

                        Ok::<(), anyhow::Error>(())
                    });

                    handles.push(handle);
                }

                for handle in handles {
                    handle
                        .await
                        .map_err(|e| anyhow::anyhow!("Worker task panicked: {e}"))??;
                }

                Ok(())
            }
            .instrument(info_span!("embed_pool")),
        );

        (Self { tx }, init_handle)
    }

    pub async fn dense(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Dense {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }

    pub async fn sparse(&self, texts: Vec<String>) -> Result<Vec<SparseEmbedding>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest::Sparse {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("EmbedPool channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Worker dropped reply sender"))?
    }
}
