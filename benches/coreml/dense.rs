//! Dense criterion benchmark groups.

use std::cell::RefCell;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use ort::value::TensorRef;

use crate::setup::{
    bench_model_variant, build_execution_providers, cache_dir, ep_name, load_bench_models,
    load_corpus, onnx_batch_size, tokenize_batch,
};

pub(crate) fn bench_embed_dense(
    session: &RefCell<ort::session::Session>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[impl AsRef<str>],
    batch_size: usize,
    variant: &str,
) -> Vec<Vec<f32>> {
    let mut all = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(batch_size) {
        let (ids, mask) = tokenize_batch(tokenizer, chunk);
        let ids_t = TensorRef::from_array_view(ids.view()).expect("ids tensor");
        let mask_t = TensorRef::from_array_view(mask.view()).expect("mask tensor");
        let mut sess = session.borrow_mut();
        let outputs = sess
            .run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
            })
            .expect("session.run failed");

        // FP32: sentence_embedding [batch, 1024] — pre-pooled CLS.
        // FP16/INT8: last_hidden_state [batch, seq, 1024] — CLS at position 0.
        match variant {
            "fp16" | "int8" => {
                let lhs = outputs["last_hidden_state"]
                    .try_extract_array::<f32>()
                    .expect("Failed to extract last_hidden_state");
                let cls = lhs.index_axis(ndarray::Axis(1), 0); // [batch, 1024]
                for i in 0..chunk.len() {
                    let row = cls.index_axis(ndarray::Axis(0), i);
                    let slice = row.as_slice().expect("contiguous");
                    let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let normalized: Vec<f32> = if norm > 0.0 {
                        slice.iter().map(|x| x / norm).collect()
                    } else {
                        slice.to_vec()
                    };
                    all.push(normalized);
                }
            }
            _ => {
                let emb = outputs["sentence_embedding"]
                    .try_extract_array::<f32>()
                    .expect("Failed to extract sentence_embedding");
                for i in 0..chunk.len() {
                    let row = emb.index_axis(ndarray::Axis(0), i);
                    let slice = row.as_slice().expect("contiguous");
                    let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let normalized: Vec<f32> = if norm > 0.0 {
                        slice.iter().map(|x| x / norm).collect()
                    } else {
                        slice.to_vec()
                    };
                    all.push(normalized);
                }
            }
        }
    }
    all
}

// NOTE(ARC-6): bench_dense and bench_sparse each load their own model instance.
// Criterion calls group functions independently and closures borrow &models, so
// each group needs its own owned session. The ~10s load cost is acceptable for
// benchmark setup and avoids Rc<RefCell<>> coupling between groups.
pub(crate) fn bench_dense(c: &mut Criterion) {
    let corpus = load_corpus();
    let cache = cache_dir();
    let eps = build_execution_providers(&cache);
    let label = ep_name();
    let onnx_bs = onnx_batch_size();
    let variant = bench_model_variant();

    eprintln!("[bench] Loading model with EP={label}, variant={variant} ...");
    let models = load_bench_models(&cache, eps);

    // Warmup — triggers CoreML model compilation on first run.
    eprintln!("[bench] Dense warmup inference (onnx_batch_size={onnx_bs}) ...");
    bench_embed_dense(
        &models.session,
        &models.tokenizer,
        &["warmup text for CoreML compilation"],
        onnx_bs,
        &variant,
    );
    eprintln!("[bench] Dense model ready.");

    let mut names: Vec<&String> = corpus.scenarios.keys().collect();
    names.sort();

    let mut group = c.benchmark_group("dense");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for name in &names {
        let scenario = &corpus.scenarios[*name];

        let single = &scenario.texts[0..1];
        group.bench_with_input(BenchmarkId::new("single", *name), &single, |b, texts| {
            b.iter(|| {
                bench_embed_dense(&models.session, &models.tokenizer, texts, onnx_bs, &variant);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("batch", *name),
            &scenario.texts,
            |b, texts| {
                b.iter(|| {
                    bench_embed_dense(&models.session, &models.tokenizer, texts, onnx_bs, &variant);
                });
            },
        );
    }

    group.finish();
}
