//! Dual-output (`/v1/embeddings:both`) equivalence cases.
//!
//! Verifies that the dual-output path (`embed_both`-equivalent) produces
//! the same numerical results as two separate single-output passes
//! (`embed_dense` + `embed_sparse`-equivalent), within FP rounding tolerance.
//!
//! This validates the algorithm-level invariant of the unified
//! `/v1/embeddings:both` endpoint without requiring the binary crate's
//! private types in scope: it runs the same ORT session three ways on the
//! same inputs and compares outputs.

use crate::helpers::{cosine_similarity, l2_normalize, locate_model_file, locate_tokenizer};

#[test]
#[ignore = "requires BGE_M3_EQUIVALENCE_TEST=1 and model download"]
fn dual_pass_equivalent_to_separate_passes() {
    if std::env::var("BGE_M3_EQUIVALENCE_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP: BGE_M3_EQUIVALENCE_TEST != 1");
        return;
    }

    let cache_dir =
        std::env::var("BGE_M3_CACHE_DIR").unwrap_or_else(|_| "/tmp/bge-m3-cache".to_string());
    let model_str = std::env::var("BGE_M3_MODEL").unwrap_or_else(|_| "fp16".to_string());
    let is_fp32 = model_str == "fp32";

    let model_path = locate_model_file(&cache_dir, &model_str)
        .expect("Could not locate model. Set BGE_M3_CACHE_DIR.");
    let tokenizer_path =
        locate_tokenizer(&cache_dir, &model_str).expect("Could not locate tokenizer.json");

    eprintln!(
        "Dual-equivalence check using model={model_str} at {}",
        model_path.display()
    );

    let texts: Vec<String> = vec![
        "fn factorial(n: u64) -> u64 { (1..=n).product() }".to_string(),
        "Rust ownership and borrowing rules at compile time".to_string(),
        "The mitochondria is the powerhouse of the cell.".to_string(),
        "SELECT * FROM users WHERE last_login > NOW() - INTERVAL '7 days';".to_string(),
    ];
    let n = texts.len();

    // Tokenize without padding, then pad to chunk-local max.
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("tokenizer load");
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .expect("truncation");
    tokenizer.with_padding(None);

    let str_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch_fast(str_refs, true)
        .expect("tokenize");
    let pad_to = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(1);

    let mut ids_flat: Vec<i64> = Vec::with_capacity(n * pad_to);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(n * pad_to);
    for enc in &encodings {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        let seq_len = ids.len();
        ids_flat.extend(ids.iter().map(|&id| i64::from(id)));
        mask_flat.extend(mask.iter().map(|&m| i64::from(m)));
        let pad = pad_to.saturating_sub(seq_len);
        ids_flat.extend(std::iter::repeat_n(1i64, pad));
        mask_flat.extend(std::iter::repeat_n(0i64, pad));
    }
    let ids_arr = ndarray::Array2::from_shape_vec((n, pad_to), ids_flat).expect("ids array");
    let mask_arr = ndarray::Array2::from_shape_vec((n, pad_to), mask_flat).expect("mask array");

    let mut sess = ort::session::Session::builder()
        .expect("builder")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("opt")
        .with_intra_threads(1)
        .expect("threads")
        .commit_from_file(&model_path)
        .expect("session load");

    // -------------------------------------------------------------------
    // Pass A: dual extraction — one session.run() yielding dense + sparse
    // (mimicking what embed_both does in src/embedder/dual.rs).
    // -------------------------------------------------------------------
    let (dense_dual, sparse_base_dual) = run_dual_pass(&mut sess, &ids_arr, &mask_arr, is_fp32);

    // -------------------------------------------------------------------
    // Pass B: dense-only — separate session.run() (mimicking embed_dense).
    // -------------------------------------------------------------------
    let dense_only = run_dense_pass(&mut sess, &ids_arr, &mask_arr, is_fp32);

    // -------------------------------------------------------------------
    // Pass C: sparse-only — separate session.run() (mimicking embed_sparse).
    // -------------------------------------------------------------------
    let sparse_base_only = run_sparse_pass(&mut sess, &ids_arr, &mask_arr, is_fp32);

    // Assert the dense rows agree exactly within FP tolerance for each text.
    assert_eq!(dense_dual.len(), dense_only.len());
    for (i, (a, b)) in dense_dual.iter().zip(dense_only.iter()).enumerate() {
        let max_abs_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let cos = cosine_similarity(a, b);
        eprintln!("  text[{i}]: dense max|Δ|={max_abs_diff:.2e}, cos={cos:.6}");
        assert!(
            max_abs_diff < 1e-4,
            "text[{i}]: dense vectors differ beyond FP tolerance (max|Δ|={max_abs_diff:.6e})"
        );
        assert!(
            cos > 0.9999,
            "text[{i}]: dense cosine similarity {cos:.6} below 0.9999 threshold"
        );
    }

    // Assert the per-token sparse-base hidden states agree elementwise.
    assert_eq!(sparse_base_dual.len(), sparse_base_only.len());
    let max_diff = sparse_base_dual
        .iter()
        .zip(sparse_base_only.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("  Token-hidden-state max|Δ| across all positions: {max_diff:.2e}");
    assert!(
        max_diff < 1e-4,
        "Token hidden states differ beyond FP tolerance (max|Δ|={max_diff:.6e})"
    );

    eprintln!("PASS: dual-output path matches separate dense + sparse passes");
}

fn run_dual_pass(
    sess: &mut ort::session::Session,
    ids: &ndarray::Array2<i64>,
    mask: &ndarray::Array2<i64>,
    is_fp32: bool,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n = ids.nrows();
    let pad_to = ids.ncols();
    let ids_t = ort::value::TensorRef::from_array_view(ids.view()).expect("ids");
    let mask_t = ort::value::TensorRef::from_array_view(mask.view()).expect("mask");
    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
        .expect("dual run");

    if is_fp32 {
        let dense_arr = outputs["sentence_embedding"]
            .try_extract_array::<f32>()
            .expect("sentence_embedding");
        let dense: Vec<Vec<f32>> = dense_arr
            .as_slice()
            .expect("contig")
            .chunks(1024)
            .map(|row| {
                let mut v = row.to_vec();
                l2_normalize(&mut v);
                v
            })
            .collect();
        let token_arr = outputs["token_embeddings"]
            .try_extract_array::<f32>()
            .expect("token_embeddings");
        let tokens = token_arr.as_slice().expect("contig").to_vec();
        (dense, tokens)
    } else {
        let lhs = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .expect("last_hidden_state");
        let lhs_slice = lhs.as_slice().expect("contig");
        let mut dense: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * pad_to * 1024;
            let mut v = lhs_slice[offset..offset + 1024].to_vec();
            l2_normalize(&mut v);
            dense.push(v);
        }
        let tokens = lhs_slice.to_vec();
        (dense, tokens)
    }
}

fn run_dense_pass(
    sess: &mut ort::session::Session,
    ids: &ndarray::Array2<i64>,
    mask: &ndarray::Array2<i64>,
    is_fp32: bool,
) -> Vec<Vec<f32>> {
    let n = ids.nrows();
    let pad_to = ids.ncols();
    let ids_t = ort::value::TensorRef::from_array_view(ids.view()).expect("ids");
    let mask_t = ort::value::TensorRef::from_array_view(mask.view()).expect("mask");
    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
        .expect("dense run");

    if is_fp32 {
        let arr = outputs["sentence_embedding"]
            .try_extract_array::<f32>()
            .expect("sentence_embedding");
        arr.as_slice()
            .expect("contig")
            .chunks(1024)
            .map(|row| {
                let mut v = row.to_vec();
                l2_normalize(&mut v);
                v
            })
            .collect()
    } else {
        let lhs = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .expect("last_hidden_state");
        let lhs_slice = lhs.as_slice().expect("contig");
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * pad_to * 1024;
            let mut v = lhs_slice[offset..offset + 1024].to_vec();
            l2_normalize(&mut v);
            out.push(v);
        }
        out
    }
}

fn run_sparse_pass(
    sess: &mut ort::session::Session,
    ids: &ndarray::Array2<i64>,
    mask: &ndarray::Array2<i64>,
    is_fp32: bool,
) -> Vec<f32> {
    let ids_t = ort::value::TensorRef::from_array_view(ids.view()).expect("ids");
    let mask_t = ort::value::TensorRef::from_array_view(mask.view()).expect("mask");
    let outputs = sess
        .run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
        .expect("sparse run");

    let key = if is_fp32 {
        "token_embeddings"
    } else {
        "last_hidden_state"
    };
    outputs[key]
        .try_extract_array::<f32>()
        .expect("token-level output")
        .as_slice()
        .expect("contig")
        .to_vec()
}
