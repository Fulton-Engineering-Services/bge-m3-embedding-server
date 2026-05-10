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

//! Sparse criterion benchmark groups.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use ndarray::ArrayView1;
use ort::value::TensorRef;

use crate::setup::{
    bench_model_variant, build_execution_providers, cache_dir, ep_name, load_bench_models,
    load_corpus, load_sparse_weights, onnx_batch_size, SPECIAL_TOKENS,
};

pub(crate) fn bench_embed_sparse(
    session: &RefCell<ort::session::Session>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[impl AsRef<str>],
    batch_size: usize,
    weight: &ndarray::Array1<f32>,
    bias: f32,
    variant: &str,
) -> Vec<HashMap<usize, f32>> {
    let weight_view = weight.view();
    let mut all = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(batch_size) {
        let str_refs: Vec<&str> = chunk.iter().map(AsRef::as_ref).collect();
        let encodings = tokenizer
            .encode_batch(str_refs, true)
            .expect("Tokenization failed");
        let batch_len = encodings.len();
        let seq_len = encodings[0].get_ids().len();
        let mut ids_flat: Vec<i64> = Vec::with_capacity(batch_len * seq_len);
        let mut mask_flat: Vec<i64> = Vec::with_capacity(batch_len * seq_len);
        for enc in &encodings {
            ids_flat.extend(enc.get_ids().iter().map(|&id| i64::from(id)));
            mask_flat.extend(enc.get_attention_mask().iter().map(|&m| i64::from(m)));
        }
        let ids = ndarray::Array2::from_shape_vec((batch_len, seq_len), ids_flat)
            .expect("input_ids shape mismatch");
        let mask = ndarray::Array2::from_shape_vec((batch_len, seq_len), mask_flat)
            .expect("attention_mask shape mismatch");
        let ids_t = TensorRef::from_array_view(ids.view()).expect("ids tensor");
        let mask_t = TensorRef::from_array_view(mask.view()).expect("mask tensor");
        let mut sess = session.borrow_mut();
        let outputs = sess
            .run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
            })
            .expect("session.run failed");
        // FP32: token_embeddings [batch, seq, 1024].
        // FP16/INT8: last_hidden_state [batch, seq, 1024] — same shape, different key.
        let output_key = match variant {
            "fp16" | "int8" => "last_hidden_state",
            _ => "token_embeddings",
        };
        let token_emb = outputs[output_key]
            .try_extract_array::<f32>()
            .expect("Failed to extract token embeddings");
        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let attn = enc.get_attention_mask();
            let batch_hidden = token_emb.index_axis(ndarray::Axis(0), i);
            let mut token_weights: HashMap<usize, f32> = HashMap::new();
            for j in 0..ids.len() {
                if attn[j] == 0 {
                    continue;
                }
                let token_id = ids[j];
                if SPECIAL_TOKENS.contains(&token_id) {
                    continue;
                }
                let hidden = batch_hidden.index_axis(ndarray::Axis(0), j);
                let hidden_slice = hidden.as_slice().expect("contiguous");
                let hidden_view = ArrayView1::from(hidden_slice);
                let score = (hidden_view.dot(&weight_view) + bias).max(0.0);
                if score > 0.0 {
                    token_weights
                        .entry(token_id as usize)
                        .and_modify(|w| *w = w.max(score))
                        .or_insert(score);
                }
            }
            all.push(token_weights);
        }
    }
    all
}

pub(crate) fn bench_sparse(c: &mut Criterion) {
    let corpus = load_corpus();
    let cache = cache_dir();
    let eps = build_execution_providers(&cache);
    let label = ep_name();
    let onnx_bs = onnx_batch_size();
    let variant = bench_model_variant();

    eprintln!("[bench] Loading model with EP={label}, variant={variant} (for sparse) ...");
    let models = load_bench_models(&cache, eps);
    let (weight, bias) = load_sparse_weights();

    eprintln!("[bench] Sparse warmup inference (onnx_batch_size={onnx_bs}) ...");
    bench_embed_sparse(
        &models.session,
        &models.tokenizer,
        &["warmup text for CoreML compilation"],
        onnx_bs,
        &weight,
        bias,
        &variant,
    );
    eprintln!("[bench] Sparse model ready.");

    let mut names: Vec<&String> = corpus.scenarios.keys().collect();
    names.sort();

    let mut group = c.benchmark_group("sparse");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for name in &names {
        let scenario = &corpus.scenarios[*name];

        let single = &scenario.texts[0..1];
        group.bench_with_input(BenchmarkId::new("single", *name), &single, |b, texts| {
            b.iter(|| {
                bench_embed_sparse(
                    &models.session,
                    &models.tokenizer,
                    texts,
                    onnx_bs,
                    &weight,
                    bias,
                    &variant,
                );
            });
        });

        group.bench_with_input(
            BenchmarkId::new("batch", *name),
            &scenario.texts,
            |b, texts| {
                b.iter(|| {
                    bench_embed_sparse(
                        &models.session,
                        &models.tokenizer,
                        texts,
                        onnx_bs,
                        &weight,
                        bias,
                        &variant,
                    );
                });
            },
        );
    }

    group.finish();
}
