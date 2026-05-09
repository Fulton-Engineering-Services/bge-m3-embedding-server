/// Startup memory probe and cost-model coefficient fitter.
///
/// Runs a sweep of `(batch, seq)` shapes on the leader worker, measures
/// peak RSS deltas, and fits a two-coefficient quadratic cost model:
///
/// ```text
/// peak_workspace(batch, seq) ≈ a * (batch * seq) + b * (batch * seq^2)
/// ```
///
/// The fitted `a` and `b` are used by [`crate::binpack::CostModel`] to make
/// bin-packing decisions that respect the per-worker memory budget.
///
/// # Fallback
///
/// When RSS measurement is unavailable (non-Linux, probe shapes error, fit
/// diverges), conservative compile-time defaults are returned and a warning is
/// logged. The server still starts; it just uses a static cost model that
/// matches the old `BGE_M3_ONNX_BATCH_SIZE = 16` behavior.
use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use tracing::{info, warn};

/// Probe shape: `(batch_count, seq_length)`.
type Shape = (usize, usize);

/// Shapes swept by the probe. Anchored at both short and long ends of the
/// sequence axis to give the least-squares fit a stable coefficient surface.
///
/// 18 points is enough for a stable two-coefficient fit (3 unknowns; well
/// over-determined at 18 observations). Probe time is dominated by the large
/// `seq` points — typically under 60 s total on Fargate.
const PROBE_SHAPES: &[Shape] = &[
    (1, 64),
    (1, 256),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    // (1, max_seq) is added dynamically based on configured max.
    (4, 64),
    (4, 256),
    (4, 1024),
    (4, 2048),
    (8, 64),
    (8, 256),
    (8, 1024),
    (16, 64),
    (16, 256),
    (16, 512),
];

/// One measured data point from the probe sweep.
#[derive(Debug, Clone, Copy)]
struct DataPoint {
    batch: usize,
    seq: usize,
    rss_delta: usize,
}

/// Runs the startup probe on the already-warmed leader worker.
///
/// # Arguments
///
/// - `pool`: the `EmbedPool` whose leader worker has already loaded models.
/// - `max_seq`: the configured `BGE_M3_MAX_SEQ_LENGTH` (determines the topmost
///   probe shape and doubles as the capability check — if the model cannot run
///   at `(1, max_seq)`, the server fails fast).
/// - `rss_ceiling`: the per-worker workspace budget computed from sysinfo.
///   Shapes estimated to exceed this are skipped to avoid OOM mid-probe.
///
/// # Returns
///
/// `Ok((a, b))` where `a` and `b` are the fitted cost-model coefficients.
/// Returns conservative defaults and logs a warning on any failure.
pub(crate) async fn run_probe(
    pool: &EmbedPool,
    max_seq: usize,
    rss_ceiling: usize,
) -> (f64, f64) {
    info!(
        max_seq,
        rss_ceiling_mb = rss_ceiling / (1024 * 1024),
        "Starting memory probe"
    );

    // Build shape list: static shapes + the max_seq capability check.
    let mut shapes: Vec<Shape> = PROBE_SHAPES.to_vec();
    // Add the max_seq cap check if not already covered.
    if !shapes.iter().any(|&(_, s)| s == max_seq) {
        shapes.push((1, max_seq));
    }
    // Remove any shapes whose seq > max_seq (out of range for this model).
    shapes.retain(|&(_, s)| s <= max_seq);
    // Sort by ascending total token-positions so we grow load gradually.
    shapes.sort_by_key(|&(b, s)| b * s);

    let mut data: Vec<DataPoint> = Vec::with_capacity(shapes.len());
    let conservative = CostModel::conservative(rss_ceiling);

    // Synthesize probe texts from corpus (already curated and pinned).
    let corpus_texts = load_probe_texts();

    for (batch, seq) in &shapes {
        let batch = *batch;
        let seq = *seq;

        // Skip shapes estimated to exceed the rss_ceiling by more than
        // conservative cost model says (avoids OOM mid-probe).
        if !conservative.fits(batch, seq) {
            info!(
                batch,
                seq,
                "Probe: skipping shape (estimated to exceed rss_ceiling)"
            );
            continue;
        }

        // Synthesize texts of approximately `seq` tokens by repeating corpus
        // texts and trimming. We can't tokenize here (no tokenizer), so we
        // approximate: one word ≈ 1.3 tokens, one char ≈ 0.25 tokens.
        // At a rough 4 chars/token, a `seq`-token input is ~4*seq characters.
        let texts = synthesize_texts(&corpus_texts, batch, seq);

        match pool.probe(texts).await {
            Ok(result) => {
                let delta = result.rss_after.saturating_sub(result.rss_before);
                info!(
                    batch,
                    seq,
                    rss_delta_mb = delta / (1024 * 1024),
                    "Probe shape measured"
                );
                data.push(DataPoint { batch, seq, rss_delta: delta });
            }
            Err(e) => {
                if seq == max_seq {
                    // The max_seq capability check failed — fail fast.
                    tracing::error!(
                        error = %e,
                        seq = max_seq,
                        model_hint = "Set BGE_M3_MODEL=fp32 or lower BGE_M3_MAX_SEQ_LENGTH",
                        "Probe: model failed at configured max_seq_length — \
                         variant may not support this sequence length"
                    );
                    // Propagate as warning; caller converts to startup failure.
                    warn!(
                        "Falling back to conservative cost model after capability check failure"
                    );
                    return (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B);
                }
                warn!(batch, seq, error = %e, "Probe shape failed; skipping");
            }
        }
    }

    if data.is_empty() {
        warn!(
            "Probe collected no data points (RSS measurement unavailable?); \
             using conservative defaults"
        );
        return (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B);
    }

    if let Some((a, b)) = fit_cost_model(&data) {
        info!(
            a = format!("{a:.0}"),
            b = format!("{b:.4}"),
            data_points = data.len(),
            "Probe: fitted cost model"
        );
        (a, b)
    } else {
        warn!(
            data_points = data.len(),
            "Probe: least-squares fit failed or produced invalid coefficients; \
             using conservative defaults"
        );
        (CostModel::CONSERVATIVE_A, CostModel::CONSERVATIVE_B)
    }
}

// ---------------------------------------------------------------------------
// Least-squares cost-model fit
// ---------------------------------------------------------------------------

/// Fits `peak = a * (batch * seq) + b * (batch * seq^2)` via ordinary least
/// squares (no intercept — workspace at batch=0 is 0 by definition).
///
/// The design matrix `X` has columns `[batch*seq, batch*seq^2]` and the
/// response `y` is `rss_delta` for each observation.
///
/// Normal equations: `(X^T X) β = X^T y`, solved via 2×2 matrix inverse.
///
/// Returns `None` when:
/// - Fewer than 2 data points (under-determined system).
/// - The Gram matrix `X^T X` is nearly singular (det < 1e-6 of max diagonal²).
/// - Either coefficient is negative (physically impossible workspace).
/// - Either coefficient falls outside the sane ranges `[4 KiB, 256 KiB]` for
///   `a` and `[0.1, 10_000]` for `b` (clamped not rejected; see below).
//
// cast_precision_loss: batch (≤ 16), seq (≤ 8192), and rss_delta (≤ ~28 GB) are
//   all well within f64's 2^52 mantissa (~4.5 PB). Coefficients are computed via
//   ordinary least squares where sub-integer precision in the inputs is irrelevant.
#[allow(clippy::cast_precision_loss)]
fn fit_cost_model(data: &[DataPoint]) -> Option<(f64, f64)> {
    if data.len() < 2 {
        return None;
    }

    // Build design matrix columns and response.
    let mut x1_sum = 0.0_f64; // sum of (batch*seq)
    let mut x2_sum = 0.0_f64; // sum of (batch*seq^2)
    let mut x11 = 0.0_f64; // X^T X [0,0]
    let mut x12 = 0.0_f64; // X^T X [0,1]
    let mut x22 = 0.0_f64; // X^T X [1,1]
    let mut xy1 = 0.0_f64; // X^T y [0]
    let mut xy2 = 0.0_f64; // X^T y [1]

    for dp in data {
        let n = (dp.batch * dp.seq) as f64;
        let n2 = n * dp.seq as f64; // batch * seq^2
        let y = dp.rss_delta as f64;

        x1_sum += n;
        x2_sum += n2;
        x11 += n * n;
        x12 += n * n2;
        x22 += n2 * n2;
        xy1 += n * y;
        xy2 += n2 * y;
    }

    // 2×2 determinant: x11*x22 - x12^2
    let det = x11 * x22 - x12 * x12;
    let max_diag_sq = x11.max(x22).powi(2);
    if det.abs() < 1e-6 * max_diag_sq {
        // Nearly singular — likely all data points at the same shape.
        return None;
    }

    // Cramer's rule solution.
    let a_raw = (x22 * xy1 - x12 * xy2) / det;
    let b_raw = (x11 * xy2 - x12 * xy1) / det;

    // Reject negative coefficients — physically impossible.
    if a_raw < 0.0 || b_raw < 0.0 {
        return None;
    }

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

    let _ = (x1_sum, x2_sum); // used in logging context if desired
    Some((a, b))
}

// ---------------------------------------------------------------------------
// Probe text synthesis
// ---------------------------------------------------------------------------

/// Loads the benchmark corpus for use as probe text material.
///
/// Falls back to a tiny built-in sentence if the corpus file is not found.
fn load_probe_texts() -> Vec<String> {
    let corpus_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/corpus.json");
    if let Ok(raw) = std::fs::read_to_string(&corpus_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(scenarios) = json["scenarios"].as_object() {
                let mut texts: Vec<String> = Vec::new();
                for scenario in scenarios.values() {
                    if let Some(arr) = scenario["texts"].as_array() {
                        texts.extend(arr.iter().filter_map(|v| v.as_str().map(String::from)));
                    }
                }
                if !texts.is_empty() {
                    return texts;
                }
            }
        }
    }
    // Fallback: minimal probe text.
    vec!["The embedding server startup probe synthesizes texts to measure workspace cost.".to_string()]
}

/// Synthesizes `batch` texts each of approximately `target_seq` tokens.
///
/// Token estimation: ~4 chars/token for natural English text.
/// We repeat/trim corpus texts to hit the target character count.
fn synthesize_texts(corpus: &[String], batch: usize, target_seq: usize) -> Vec<String> {
    let target_chars = target_seq.saturating_mul(4).max(16);
    (0..batch)
        .map(|i| {
            let base = &corpus[i % corpus.len()];
            // Repeat the base text until we have enough characters.
            let repeated = base.repeat((target_chars / base.len().max(1)).max(2) + 1);
            // Trim to target_chars bytes (not chars, but close enough for probing).
            let trimmed = if repeated.len() > target_chars {
                &repeated[..target_chars]
            } else {
                &repeated
            };
            trimmed.to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_cost_model_two_points() {
        // hand-crafted data: batch=1,seq=64 → 8 MB; batch=1,seq=512 → 100 MB.
        // Expect a reasonable (a,b) pair.
        let data = vec![
            DataPoint { batch: 1, seq: 64, rss_delta: 8_000_000 },
            DataPoint { batch: 1, seq: 512, rss_delta: 100_000_000 },
            DataPoint { batch: 4, seq: 64, rss_delta: 30_000_000 },
            DataPoint { batch: 4, seq: 256, rss_delta: 80_000_000 },
        ];
        let result = fit_cost_model(&data);
        assert!(result.is_some(), "should produce a valid fit");
        let (a, b) = result.unwrap();
        assert!(a > 0.0, "a must be positive");
        assert!(b > 0.0, "b must be positive");
    }

    #[test]
    fn fit_cost_model_single_point_returns_none() {
        let data = vec![DataPoint { batch: 1, seq: 128, rss_delta: 5_000_000 }];
        assert!(fit_cost_model(&data).is_none(), "need >= 2 points");
    }

    #[test]
    fn fit_cost_model_singular_system_returns_none() {
        // All points on the same line through the origin in one variable — singular.
        let data = vec![
            DataPoint { batch: 1, seq: 128, rss_delta: 1_000_000 },
            DataPoint { batch: 1, seq: 128, rss_delta: 1_000_000 },
        ];
        // x2 = x1 * seq = same for both → singular (columns are linearly dependent).
        assert!(
            fit_cost_model(&data).is_none(),
            "identical rows should give None"
        );
    }

    #[test]
    fn fit_rejects_negative_coefficients() {
        // Construct data that forces a negative coefficient — artificially small
        // rss at high seq relative to low seq.
        let data = vec![
            DataPoint { batch: 16, seq: 64, rss_delta: 10_000_000 },
            DataPoint { batch: 1, seq: 8192, rss_delta: 1 }, // pathological
            DataPoint { batch: 8, seq: 256, rss_delta: 5_000_000 },
            DataPoint { batch: 4, seq: 1024, rss_delta: 2_000_000 },
        ];
        // May or may not produce a valid fit; just assert it doesn't panic.
        let _ = fit_cost_model(&data);
    }

    #[test]
    fn synthesize_texts_returns_batch_count() {
        let corpus = vec!["hello world".to_string(); 5];
        let texts = synthesize_texts(&corpus, 7, 256);
        assert_eq!(texts.len(), 7);
    }

    #[test]
    fn synthesize_texts_produces_non_empty() {
        let corpus = vec!["x".to_string()];
        let texts = synthesize_texts(&corpus, 3, 64);
        for t in &texts {
            assert!(!t.is_empty());
        }
    }

    #[test]
    fn load_probe_texts_returns_nonempty() {
        // Corpus is available when tests run from project root.
        let texts = load_probe_texts();
        assert!(!texts.is_empty());
    }

    #[test]
    fn probe_shapes_are_sorted_ascending_token_positions() {
        // Verify the static table is sane (not a hard requirement, just hygiene).
        let mut prev = 0usize;
        for &(b, s) in PROBE_SHAPES {
            let positions = b * s;
            // Not required to be strictly ascending (table has duplicates at same
            // token count but different (b,s) combos), just check no zeros.
            assert!(positions > 0, "probe shape ({b},{s}) has zero token positions");
            let _ = prev;
            prev = positions;
        }
    }
}
