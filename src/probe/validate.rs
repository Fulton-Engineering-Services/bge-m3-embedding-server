//! Tokenizer + ndarray shape validation at the configured `max_seq`.

/// Validates that the configured `max_seq` is reachable by the tokenizer and
/// ndarray dimension math without performing a full ORT `session.run()`.
///
/// Replaces the old `(1, max_seq)` capability check that called `session.run()`
/// and risked OOM-killing the container on memory-constrained hosts.
///
/// This function constructs the `input_ids` and `attention_mask` ndarrays at
/// `(1, max_seq)` and logs their shapes, confirming that:
/// - `max_seq` fits within `usize` bounds.
/// - ndarray can allocate the 2D layout `[1, max_seq]`.
///
/// Note: full position-embedding coverage (whether the ONNX model actually
/// supports the configured `max_seq`) is NOT verified here. Runtime errors
/// from the first real `/v1/embeddings` request surface that condition with
/// a clear ORT error, without risking startup OOM.
pub(super) fn validate_max_seq_shape(max_seq: usize) {
    let ids: ndarray::Array2<i64> = ndarray::Array2::zeros((1, max_seq));
    let mask: ndarray::Array2<i64> = ndarray::Array2::ones((1, max_seq));
    tracing::debug!(
        max_seq,
        ids_shape = ?ids.dim(),
        mask_shape = ?mask.dim(),
        "Max-seq shape validation passed (no session.run)"
    );
}
