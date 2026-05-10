//! Least-squares cost-model fit for the startup probe.

use tracing::warn;

/// One measured data point from the probe sweep.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DataPoint {
    pub batch: usize,
    pub seq: usize,
    pub rss_delta: usize,
}

/// Fits `peak = a * (batch * seq) + b * (batch * seq^2)` via ordinary least
/// squares (no intercept — workspace at batch=0 is 0 by definition).
///
/// The design matrix `X` has columns `[batch*seq, batch*seq^2]` and the
/// response `y` is `rss_delta` for each observation.
///
/// **Normalization**: columns are scaled to `[0, 1]` before solving
/// (`ξ1 = x1 / max(x1)`, `ξ2 = x2 / max(x2)`).  Without this, `x2` at
/// `max_seq=8192` exceeds `x1` by ~8000×, making the Gram matrix effectively
/// rank-1 under the naïve det threshold and causing the fit to silently fall
/// back to conservative defaults despite valid data.
///
/// Normal equations solved in normalized space via 2×2 matrix inverse
/// (Cramer's rule), then unscaled: `a = α / x1_max`, `b = β / x2_max`.
///
/// Returns `None` when:
/// - Fewer than 2 data points (under-determined system).
/// - `x1_max` or `x2_max` is zero (degenerate data).
/// - The normalized Gram matrix is nearly singular
///   (det < 1e-6 of max diagonal²).
/// - Either coefficient is negative (physically impossible workspace).
//
// cast_precision_loss: batch (≤ 16), seq (≤ 8192), and rss_delta (≤ ~28 GB) are
//   all well within f64's 2^52 mantissa (~4.5 PB). Coefficients are computed via
//   ordinary least squares where sub-integer precision in the inputs is irrelevant.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn fit_cost_model(data: &[DataPoint]) -> Option<(f64, f64)> {
    if data.len() < 2 {
        return None;
    }

    // Compute scale factors so both design-matrix columns lie in [0, 1].
    // Without normalization the x2 column (batch*seq²) at max_seq=8192 is
    // ~8000× larger than x1 (batch*seq), making the Gram matrix near-singular
    // under the det threshold even with 16 well-distributed data points.
    let x1_max = data
        .iter()
        .map(|dp| (dp.batch * dp.seq) as f64)
        .fold(0.0_f64, f64::max);
    let x2_max = data
        .iter()
        .map(|dp| (dp.batch * dp.seq * dp.seq) as f64)
        .fold(0.0_f64, f64::max);

    if x1_max == 0.0 || x2_max == 0.0 {
        return None;
    }

    // Build normalized Gram matrix: n1 = x1/x1_max, n2 = x2/x2_max ∈ [0,1].
    // Variable names use single-letter prefixes to avoid clippy::similar_names
    // on the longer accumulator names (g11, g12, g22, gy1, gy2).
    let mut g11 = 0.0_f64; // sum(n1²)
    let mut g12 = 0.0_f64; // sum(n1*n2)
    let mut g22 = 0.0_f64; // sum(n2²)
    let mut gy1 = 0.0_f64; // sum(n1*y)
    let mut gy2 = 0.0_f64; // sum(n2*y)

    for dp in data {
        let n1 = (dp.batch * dp.seq) as f64 / x1_max;
        let n2 = (dp.batch * dp.seq * dp.seq) as f64 / x2_max;
        let y = dp.rss_delta as f64;

        g11 += n1 * n1;
        g12 += n1 * n2;
        g22 += n2 * n2;
        gy1 += n1 * y;
        gy2 += n2 * y;
    }

    // 2×2 determinant in normalized space.
    // With n1, n2 ∈ [0,1], max_diag ≤ N and det is directly comparable.
    let det = g11 * g22 - g12 * g12;
    let max_diag_sq = g11.max(g22).powi(2);
    if det.abs() < 1e-6 * max_diag_sq {
        // Nearly singular — likely all data points at the same shape or
        // concentrated along one direction in design space.
        return None;
    }

    // Cramer's rule in normalized space → normalized coefficients.
    let alpha = (g22 * gy1 - g12 * gy2) / det; // coefficient of n1
    let beta = (g11 * gy2 - g12 * gy1) / det; // coefficient of n2

    // Unscale: a = alpha / x1_max, b = beta / x2_max.
    let a_raw = alpha / x1_max;
    let b_raw = beta / x2_max;

    // ----- Negative-coefficient handling (rc8) -----
    //
    // OLS can return a negative `a_raw` when the data has a sharp
    // discontinuity in y across the seq axis — for example, when ORT
    // switches attention kernels between seq=2048 and seq=4096 (small
    // memory-frugal fused kernel below the threshold; full O(N²) score
    // matrix above). The two-coefficient quadratic model `y = a·N + b·N²`
    // cannot describe a step function — the fitter has to drive `a` strongly
    // negative to subtract the quadratic prediction back out at low seq
    // where y ≈ 0.
    //
    // In that regime, `b_raw` is fine (the high-seq points fit a clean
    // quadratic) but `a_raw` is non-physical. The rc7 production data was
    // exactly this: `a_raw ≈ -109,000`, `b_raw ≈ 117`. Returning `None`
    // and falling back to conservative defaults under-budgets ORT
    // workspace by ~12× at high seq, which causes batched real-traffic
    // OOMs (see CLAUDE.md gotcha "rc7 production capacity at max_seq=8192").
    //
    // Fix: when `a_raw` is negative, raise it to 0 and let the existing
    // `.clamp(4_096.0, ...)` lower bound floor it at 4 KiB/token. That
    // produces a fitted `b` that correctly predicts high-seq workspace
    // (so the bin-packer rejects oversize batches) and an `a` that
    // slightly over-predicts low-seq workspace (which is the safe
    // direction — bin-packer might split low-seq batches more
    // aggressively than ideal, but never accepts unsafe ones).
    //
    // A negative `b_raw` still fails fast: that would require a quadratic
    // model to predict workspace *decreasing* as seq grows, which is
    // genuinely non-physical and signals a measurement bug.
    if b_raw < 0.0 {
        return None;
    }
    let a_raw = a_raw.max(0.0);

    // Clamp to sane operational ranges.
    // a: [4 KiB, 256 KiB] per token-position
    let a = a_raw.clamp(4_096.0, 262_144.0);
    // b: [0.01, 50_000] bytes per token-position^2
    let b = b_raw.clamp(0.01, 50_000.0);

    // Log if clamping was significant.
    let a_clamped = (a - a_raw).abs() > 0.01 * a_raw.abs();
    let b_clamped = (b - b_raw).abs() > 0.01 * b_raw.abs();
    if a_clamped || b_clamped {
        warn!(
            a_raw = format!("{a_raw:.0}"),
            b_raw = format!("{b_raw:.4}"),
            a_clamped = format!("{a:.0}"),
            b_clamped = format!("{b:.4}"),
            "Probe: fitted coefficients were clamped to sane range"
        );
    }

    Some((a, b))
}
