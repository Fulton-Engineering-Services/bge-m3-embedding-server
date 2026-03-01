use ndarray::Array1;
use std::sync::OnceLock;

static WEIGHTS_BYTES: &[u8] = include_bytes!("sparse_linear.safetensors");

static SPARSE_LINEAR: OnceLock<(Array1<f32>, f32)> = OnceLock::new();

/// Returns the sparse-linear projection weights used by BGE-M3 sparse embedding.
///
/// The safetensors file contains a weight vector `[1024]` and a scalar bias.
/// Parsed once on first call and cached for the lifetime of the process.
pub(crate) fn sparse_linear() -> &'static (Array1<f32>, f32) {
    SPARSE_LINEAR.get_or_init(|| {
        let tensors = safetensors::SafeTensors::deserialize(WEIGHTS_BYTES)
            .expect("embedded sparse_linear.safetensors must be valid");

        let weight_view = tensors
            .tensor("weight")
            .expect("sparse_linear must contain 'weight' tensor");
        let bias_view = tensors
            .tensor("bias")
            .expect("sparse_linear must contain 'bias' tensor");

        let weight: Vec<f32> = weight_view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let bias = f32::from_le_bytes([
            bias_view.data()[0],
            bias_view.data()[1],
            bias_view.data()[2],
            bias_view.data()[3],
        ]);

        assert_eq!(weight.len(), 1024, "sparse_linear weight must be [1024]");
        (Array1::from(weight), bias)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_linear_loads_correct_shape() {
        let (weight, bias) = sparse_linear();
        assert_eq!(weight.len(), 1024);
        assert!(bias.abs() < 100.0, "bias should be a small number");
    }

    #[test]
    fn sparse_linear_is_idempotent() {
        let a = sparse_linear();
        let b = sparse_linear();
        assert!(std::ptr::eq(a, b), "should return the same cached ref");
    }
}
