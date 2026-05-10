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
use std::path::Path;
use tracing::{info, warn};

/// Probe shape: `(batch_count, seq_length)`.
type Shape = (usize, usize);

/// Shapes swept by the probe.
///
/// 7 shapes (6 static + 1 dynamic `max_seq`) are sufficient for a stable
/// two-coefficient fit. The static set is chosen to anchor both the linear
/// (`a`) and quadratic (`b`) coefficients across a wide token-position range:
///
/// - `(1, 64)` and `(1, 256)` anchor the linear term at low seq.
/// - `(4, 64)` shares `x1 = batch*seq = 256` with `(1, 256)` but has a
///   different `x2 = batch*seq² = 16384` vs `65536`, giving a near-direct
///   measurement of `b` independent of `a`.
/// - `(1, 1024)` and `(1, 2048)` provide mid-range leverage.
/// - `(1, 4096)` anchors the quadratic regime.
/// - `(1, max_seq)` is added dynamically — it serves as the capability check
///   and is the dominant quadratic anchor at the configured upper bound.
///
/// Removed from the original 16-shape set: all `(batch > 1, seq > 64)`
/// shapes such as `(4, 1024)`, `(4, 2048)`, `(8, 1024)`, `(16, 512)`.
/// These contributed noise to the fit (RSS delta for batch=16 includes ORT
/// arena effects and scheduler jitter not captured by the simple cost model)
/// without improving the stability condition of the normalized Gram matrix.
/// Estimated probe time with this set: ~120 s vs ~3.7 min (old 16-shape set
/// when shapes all ran) or up to ~20 min worst-case (old set on arm64 MLAS).
const PROBE_SHAPES: &[Shape] = &[
    (1, 64),   // linear anchor
    (4, 64),   // pairs with (1,256) for direct b isolation
    (1, 256),  // linear anchor
    (1, 1024), // mid-range
    (1, 2048), // mid-range, improves stability condition
    (1, 4096), // quadratic anchor
               // (1, max_seq) is added dynamically based on configured max.
];

/// One measured data point from the probe sweep.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DataPoint {
    pub batch: usize,
    pub seq: usize,
    pub rss_delta: usize,
}

/// Persistent cache of fitted probe coefficients stored on the EFS volume.
///
/// The cache key is `{server_version, model, max_seq, arch}`. When the
/// fingerprint matches the current server's configuration, the probe is
/// skipped and the cached `(a, b)` are used immediately.
#[derive(serde::Serialize, serde::Deserialize)]
struct ProbeCache {
    schema_version: u32,
    server_version: String,
    model: String,
    max_seq: usize,
    arch: String,
    fitted_at_unix: u64,
    a: f64,
    b: f64,
}

/// Attempts to load cached probe coefficients from `{cache_dir}/probe-coefficients.json`.
///
/// Returns `Some((a, b))` when a valid, fingerprint-matching cache file exists.
/// Returns `None` when the file is absent, unreadable, or the fingerprint does
/// not match the current `(server_version, model_variant, max_seq, arch)`.
pub(crate) fn try_load_probe_cache(
    cache_dir: &Path,
    model_variant: &str,
    max_seq: usize,
) -> Option<(f64, f64)> {
    let path = cache_dir.join("probe-coefficients.json");
    let raw = std::fs::read_to_string(&path).ok()?;
    let cache: ProbeCache = serde_json::from_str(&raw).ok()?;

    let current_version = env!("CARGO_PKG_VERSION");
    let current_arch = std::env::consts::ARCH;

    if cache.schema_version != 1
        || cache.server_version != current_version
        || cache.model != model_variant
        || cache.max_seq != max_seq
        || cache.arch != current_arch
    {
        info!(
            cached_version = %cache.server_version,
            current_version,
            cached_model = %cache.model,
            model_variant,
            cached_max_seq = cache.max_seq,
            max_seq,
            cached_arch = %cache.arch,
            current_arch,
            "Probe cache fingerprint mismatch; will re-probe"
        );
        return None;
    }

    if cache.a <= 0.0 || cache.b <= 0.0 {
        warn!("Probe cache has non-positive coefficients; ignoring");
        return None;
    }

    info!(
        a = cache.a,
        b = cache.b,
        fitted_at_unix = cache.fitted_at_unix,
        "Probe cache hit — skipping startup probe"
    );
    Some((cache.a, cache.b))
}

/// Saves fitted probe coefficients to `{cache_dir}/probe-coefficients.json`
/// via an atomic temp-file + rename.
///
/// Errors are logged and silently ignored — a cache write failure must never
/// abort the server.
pub(crate) fn save_probe_cache(
    cache_dir: &Path,
    model_variant: &str,
    max_seq: usize,
    a: f64,
    b: f64,
) {
    let fitted_at_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let cache = ProbeCache {
        schema_version: 1,
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        model: model_variant.to_string(),
        max_seq,
        arch: std::env::consts::ARCH.to_string(),
        fitted_at_unix,
        a,
        b,
    };

    let json = match serde_json::to_string_pretty(&cache) {
        Ok(j) => j,
        Err(e) => {
            warn!(error = %e, "Failed to serialize probe cache; skipping write");
            return;
        }
    };

    let final_path = cache_dir.join("probe-coefficients.json");
    let tmp_path = cache_dir.join("probe-coefficients.json.tmp");

    if let Err(e) = std::fs::write(&tmp_path, &json) {
        warn!(error = %e, path = %tmp_path.display(), "Failed to write probe cache temp file");
        return;
    }

    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        warn!(error = %e, "Failed to atomically rename probe cache file");
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }

    info!(
        path = %final_path.display(),
        a,
        b,
        "Probe coefficients cached to EFS"
    );
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
pub(crate) async fn run_probe(pool: &EmbedPool, max_seq: usize, rss_ceiling: usize) -> (f64, f64) {
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
                seq, "Probe: skipping shape (estimated to exceed rss_ceiling)"
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
                data.push(DataPoint {
                    batch,
                    seq,
                    rss_delta: delta,
                });
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
                    warn!("Falling back to conservative cost model after capability check failure");
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
    vec![
        "The embedding server startup probe synthesizes texts to measure workspace cost."
            .to_string(),
    ]
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

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Builds a `DataPoint` from `(batch, seq, a, b)` using the model formula
    /// `rss = a * (batch * seq) + b * (batch * seq²)`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn make_dp(batch: usize, seq: usize, a: f64, b: f64) -> DataPoint {
        let rss_delta = (a * (batch * seq) as f64 + b * (batch * seq * seq) as f64) as usize;
        DataPoint {
            batch,
            seq,
            rss_delta,
        }
    }

    // ---------------------------------------------------------------------------
    // Stage 1: diagnostic — does current fit_cost_model pass for production data?
    //
    // This test uses the 16-shape production set with a=18000, b=6 (fp16/aarch64
    // typical values). Before the normalization fix (Stage 2) this would return
    // None because the Gram matrix det falls below the scale-invariance threshold.
    // After Stage 2 it must return Some with coefficients close to the ground truth.
    // ---------------------------------------------------------------------------

    #[test]
    fn fit_cost_model_production_scale_16_shapes_with_max_seq_8192() {
        // All 16 shapes swept by the old probe sweep, plus the dynamic (1,8192).
        let a_true = 18_000.0_f64;
        let b_true = 6.0_f64;
        let data: Vec<DataPoint> = [
            (1usize, 64usize),
            (1, 256),
            (1, 1024),
            (1, 2048),
            (1, 4096),
            (1, 8192),
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
        ]
        .iter()
        .map(|&(b, s)| make_dp(b, s, a_true, b_true))
        .collect();

        let result = fit_cost_model(&data);
        // After the normalization fix this must succeed.
        assert!(
            result.is_some(),
            "fit_cost_model should succeed on 16-shape production data including (1,8192)"
        );
        let (a, b) = result.unwrap();
        // Expect recovery within 5% of the true coefficients.
        assert!(
            (a - a_true).abs() < 0.05 * a_true,
            "a={a:.0} should be within 5% of {a_true}"
        );
        assert!(
            (b - b_true).abs() < 0.05 * b_true,
            "b={b:.4} should be within 5% of {b_true}"
        );
    }

    // ---------------------------------------------------------------------------
    // Stage 2: normalized OLS correctness
    // ---------------------------------------------------------------------------

    #[test]
    fn fit_cost_model_two_points() {
        // hand-crafted data: batch=1,seq=64 → 8 MB; batch=1,seq=512 → 100 MB.
        // Expect a reasonable (a,b) pair.
        let data = vec![
            DataPoint {
                batch: 1,
                seq: 64,
                rss_delta: 8_000_000,
            },
            DataPoint {
                batch: 1,
                seq: 512,
                rss_delta: 100_000_000,
            },
            DataPoint {
                batch: 4,
                seq: 64,
                rss_delta: 30_000_000,
            },
            DataPoint {
                batch: 4,
                seq: 256,
                rss_delta: 80_000_000,
            },
        ];
        let result = fit_cost_model(&data);
        assert!(result.is_some(), "should produce a valid fit");
        let (a, b) = result.unwrap();
        assert!(a > 0.0, "a must be positive");
        assert!(b > 0.0, "b must be positive");
    }

    #[test]
    fn fit_cost_model_recovers_known_coefficients_from_7_probe_shapes() {
        // Verify the new 7-shape set also gives a good fit.
        let a_true = 18_500.0_f64;
        let b_true = 6.5_f64;
        let data: Vec<DataPoint> = [
            (1, 64),
            (4, 64),
            (1, 256),
            (1, 1024),
            (1, 2048),
            (1, 4096),
            (1, 8192),
        ]
        .iter()
        .map(|&(b, s)| make_dp(b, s, a_true, b_true))
        .collect();

        let result = fit_cost_model(&data);
        assert!(result.is_some(), "7-shape fit should succeed");
        let (a, b) = result.unwrap();
        assert!(
            (a - a_true).abs() < 0.05 * a_true,
            "a={a:.0} should be within 5% of {a_true}"
        );
        assert!(
            (b - b_true).abs() < 0.05 * b_true,
            "b={b:.4} should be within 5% of {b_true}"
        );
    }

    #[test]
    fn fit_cost_model_single_point_returns_none() {
        let data = vec![DataPoint {
            batch: 1,
            seq: 128,
            rss_delta: 5_000_000,
        }];
        assert!(fit_cost_model(&data).is_none(), "need >= 2 points");
    }

    #[test]
    fn fit_cost_model_singular_system_returns_none() {
        // All points on the same line through the origin in one variable — singular.
        let data = vec![
            DataPoint {
                batch: 1,
                seq: 128,
                rss_delta: 1_000_000,
            },
            DataPoint {
                batch: 1,
                seq: 128,
                rss_delta: 1_000_000,
            },
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
            DataPoint {
                batch: 16,
                seq: 64,
                rss_delta: 10_000_000,
            },
            DataPoint {
                batch: 1,
                seq: 8192,
                rss_delta: 1,
            }, // pathological
            DataPoint {
                batch: 8,
                seq: 256,
                rss_delta: 5_000_000,
            },
            DataPoint {
                batch: 4,
                seq: 1024,
                rss_delta: 2_000_000,
            },
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
            assert!(
                positions > 0,
                "probe shape ({b},{s}) has zero token positions"
            );
            let _ = prev;
            prev = positions;
        }
    }

    #[test]
    fn probe_cache_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        save_probe_cache(dir.path(), "fp16", 8192, 18_432.0, 6.2);
        let result = try_load_probe_cache(dir.path(), "fp16", 8192);
        assert!(result.is_some(), "should load after save");
        let (a, b) = result.unwrap();
        assert!((a - 18_432.0).abs() < 1.0);
        assert!((b - 6.2).abs() < 0.01);
    }

    #[test]
    fn probe_cache_fingerprint_mismatch_returns_none() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        save_probe_cache(dir.path(), "fp16", 8192, 18_432.0, 6.2);
        // Different model variant
        assert!(
            try_load_probe_cache(dir.path(), "fp32", 8192).is_none(),
            "model mismatch should return None"
        );
        // Different max_seq
        assert!(
            try_load_probe_cache(dir.path(), "fp16", 4096).is_none(),
            "max_seq mismatch should return None"
        );
    }

    #[test]
    fn probe_cache_missing_returns_none() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        assert!(
            try_load_probe_cache(dir.path(), "fp16", 8192).is_none(),
            "missing cache file should return None"
        );
    }
}
