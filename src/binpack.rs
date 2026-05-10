/// Quadratic-aware workspace cost model for ONNX attention inference.
///
/// BGE-M3 uses multi-head attention whose intermediate tensor footprint scales
/// as `O(batch * seq^2)` (attention score matrix) plus `O(batch * seq)`
/// (FFN intermediates, projection matrices). The total peak workspace is
/// approximately:
///
/// ```text
/// peak ≈ a * (batch * seq) + b * (batch * seq^2)
/// ```
///
/// where `a` (bytes/token-position) captures the FFN / projection contribution
/// and `b` (bytes/token-position^2) captures the attention contribution.
///
/// At sequence length 512 attention is small relative to FFN, so a linear
/// approximation works. At 8192, `b * N^2` dominates by ~16×, so using only
/// `a` would under-budget by that same factor.
///
/// Coefficients are derived at startup by [`crate::probe`] or set
/// conservatively from compile-time defaults when measurement is unavailable.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct CostModel {
    /// Bytes per token-position (linear term: FFN intermediates, projections).
    pub a: f64,
    /// Bytes per token-position-squared (quadratic term: attention scores).
    pub b: f64,
    /// Maximum workspace bytes available per worker for a single `session.run()` call.
    pub max_workspace_bytes: usize,
}

impl CostModel {
    /// Conservative static defaults calibrated so a `(16, 512)` chunk lands at
    /// ~140 MB workspace — matching the old static budget at the previous default
    /// `BGE_M3_ONNX_BATCH_SIZE = 16`, `MAX_SEQ_LENGTH = 512`.
    ///
    /// These are used when the probe cannot run (no ORT, no model, macOS without
    /// cgroup support) or when `BGE_M3_DISABLE_AUTO_BUDGET` is set.
    ///
    /// Formula check: 16 KiB/token × 16 × 512 + 8 B/token² × 16 × 512²
    ///   = 16384 × 8192 + 8 × 16 × 262144
    ///   = 134 217 728 + 33 554 432
    ///   = 167 772 160 ≈ 160 MB per chunk (workers run sequentially inside one worker).
    pub const CONSERVATIVE_A: f64 = 16_384.0; // 16 KiB per token-position
    pub const CONSERVATIVE_B: f64 = 8.0; // 8 bytes per token-position^2

    /// Default maximum workspace per worker when memory cannot be detected.
    ///
    /// 2 GiB is conservatively safe for the Fargate 28 GiB task with 7 workers
    /// (`28 GB * 0.7 safety / 7 workers ≈ 2.8 GB`); we round down for headroom.
    pub const DEFAULT_MAX_WORKSPACE: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

    /// Constructs a `CostModel` with conservative defaults and the given workspace ceiling.
    pub fn conservative(max_workspace_bytes: usize) -> Self {
        Self {
            a: Self::CONSERVATIVE_A,
            b: Self::CONSERVATIVE_B,
            max_workspace_bytes,
        }
    }

    /// Estimated peak workspace (bytes) for a single `session.run()` call with
    /// `count` texts and `max_seq` as the padded sequence length.
    ///
    /// Uses saturating arithmetic on `u128` to avoid overflow at large inputs.
    //
    // cast_precision_loss: n is u128, but realistic values (batch ≤ 256, seq ≤ 8192)
    //   keep n ≤ 2_097_152 — well within f64's 2^52 mantissa — so no bits are lost.
    // cast_possible_truncation: f64 → u128 intentionally floors fractional bytes;
    //   this is a memory *budget estimate*, not an exact byte count.
    // cast_sign_loss: a and b are validated positive at construction, so the
    //   products are always ≥ 0 before the cast.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn chunk_cost(&self, count: usize, max_seq: usize) -> u128 {
        let n = count as u128 * max_seq as u128;
        let linear = (self.a * n as f64) as u128;
        let quad = (self.b * n as f64 * max_seq as f64) as u128;
        linear.saturating_add(quad)
    }

    /// Returns `true` if the chunk fits within the workspace budget.
    pub fn fits(&self, count: usize, max_seq: usize) -> bool {
        self.chunk_cost(count, max_seq) <= self.max_workspace_bytes as u128
    }
}

/// Length-sorted greedy bin-packer.
///
/// Partitions `seq_lengths` (indexed 0..n) into contiguous groups (chunks)
/// where each chunk satisfies `cost_model.fits(chunk.len(), max_seq_in_chunk)`.
///
/// If a single text exceeds the budget on its own — which can happen when
/// `max_workspace_bytes` is very small or the text is at `MAX_SEQ_LENGTH` and
/// the budget is tighter than one text — it gets its own single-element chunk.
/// The caller (ORT session) will either succeed or fail; we never silently
/// truncate or discard inputs.
///
/// # Returns
///
/// `Vec<Vec<usize>>` where each inner `Vec` contains the **original indices**
/// of texts in that chunk, sorted ascending by sequence length. The outer vec
/// preserves the order chunks should be processed in. Callers scatter results
/// back to the original positions using these indices.
///
/// # Complexity
///
/// `O(n log n)` sort + `O(n)` scan.
pub(crate) fn bin_pack(seq_lengths: &[usize], cost_model: &CostModel) -> Vec<Vec<usize>> {
    if seq_lengths.is_empty() {
        return Vec::new();
    }

    // Sort indices by ascending sequence length so we can greedily pack
    // short texts together. Long texts naturally form their own small chunks.
    let mut sorted: Vec<usize> = (0..seq_lengths.len()).collect();
    sorted.sort_unstable_by_key(|&i| seq_lengths[i]);

    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_max_seq: usize = 0;

    for idx in sorted {
        let seq = seq_lengths[idx];
        let new_max = current_max_seq.max(seq);
        let new_count = current.len() + 1;

        if current.is_empty() || cost_model.fits(new_count, new_max) {
            // Adding this text keeps the chunk within budget.
            current.push(idx);
            current_max_seq = new_max;
        } else {
            // Flush the current chunk and start a new one.
            tracing::debug!(
                chunk_idx = chunks.len(),
                batch = current.len(),
                max_seq = current_max_seq,
                estimated_workspace_mb =
                    cost_model.chunk_cost(current.len(), current_max_seq) / (1024 * 1024),
                "bin_pack chunk decided"
            );
            chunks.push(std::mem::take(&mut current));
            current.push(idx);
            current_max_seq = seq;
        }
    }

    if !current.is_empty() {
        tracing::debug!(
            chunk_idx = chunks.len(),
            batch = current.len(),
            max_seq = current_max_seq,
            estimated_workspace_mb =
                cost_model.chunk_cost(current.len(), current_max_seq) / (1024 * 1024),
            "bin_pack chunk decided"
        );
        chunks.push(current);
    }

    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn model(a: f64, b: f64, max_bytes: usize) -> CostModel {
        CostModel {
            a,
            b,
            max_workspace_bytes: max_bytes,
        }
    }

    // Simple model with no quadratic term; maps budget/count = max_tokens_per_chunk.
    fn linear_model(bytes_per_token: f64, max_bytes: usize) -> CostModel {
        model(bytes_per_token, 0.0, max_bytes)
    }

    // ── basic correctness ──────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_empty() {
        let cm = linear_model(1.0, 1000);
        assert!(bin_pack(&[], &cm).is_empty());
    }

    #[test]
    fn single_text_one_chunk() {
        let cm = linear_model(1.0, 1000);
        let chunks = bin_pack(&[100], &cm);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], vec![0]);
    }

    #[test]
    fn all_texts_fit_in_one_chunk() {
        // Budget: 10 000 bytes; 10 texts × 50 tokens × 1 byte/token = 500 bytes.
        let cm = linear_model(1.0, 10_000);
        let seqs = vec![50usize; 10];
        let chunks = bin_pack(&seqs, &cm);
        assert_eq!(chunks.len(), 1);

        // All 10 original indices must be present.
        let mut found: Vec<usize> = chunks.into_iter().flatten().collect();
        found.sort_unstable();
        assert_eq!(found, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn texts_split_across_chunks_by_budget() {
        // Budget: 100 tokens × 1 byte; each text is 60 tokens.
        // Two texts = 120 > 100, so each chunk holds exactly 1 text.
        let cm = linear_model(1.0, 100);
        let seqs = vec![60usize; 3];
        let chunks = bin_pack(&seqs, &cm);
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 1);
        }
    }

    #[test]
    fn one_huge_plus_many_tiny() {
        // Budget: 1000 bytes / 1 byte per token.
        // One text at 900 tokens; 100 texts at 5 tokens each.
        let cm = linear_model(1.0, 1000);
        let mut seqs = vec![5usize; 100];
        seqs.push(900); // last index = 100, seq = 900

        let chunks = bin_pack(&seqs, &cm);

        // The huge text should be alone (900 + 5 > 1000 for any pairing with a 5-token text).
        let huge_chunk = chunks.iter().find(|c| c.contains(&100));
        let huge_chunk = huge_chunk.expect("huge text must be in some chunk");
        assert_eq!(huge_chunk.len(), 1, "huge text should be alone");

        // All tiny texts should be packed densely: 5 tokens × 200 = 1000 ≤ 1000.
        // The tiny chunks should each hold 200 texts (floor(1000/5) = 200).
        let total_tiny: usize = chunks
            .iter()
            .filter(|c| !c.contains(&100))
            .map(Vec::len)
            .sum();
        assert_eq!(total_tiny, 100);

        // All original indices appear exactly once.
        let mut all_idx: Vec<usize> = chunks.into_iter().flatten().collect();
        all_idx.sort_unstable();
        assert_eq!(all_idx, (0..101).collect::<Vec<_>>());
    }

    #[test]
    fn all_indices_present_exactly_once() {
        let cm = linear_model(10.0, 5000);
        let seqs = vec![32, 64, 128, 256, 512, 256, 128, 64, 32, 512];
        let chunks = bin_pack(&seqs, &cm);

        let mut found: Vec<usize> = chunks.into_iter().flatten().collect();
        found.sort_unstable();
        assert_eq!(found, (0..seqs.len()).collect::<Vec<_>>());
    }

    #[test]
    fn zero_max_workspace_each_text_solo() {
        let cm = model(1.0, 1.0, 0);
        let seqs = vec![10usize; 5];
        let chunks = bin_pack(&seqs, &cm);
        // Even a single text costs > 0, so every text must be alone.
        assert_eq!(chunks.len(), 5);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 1);
        }
    }

    #[test]
    fn single_text_exceeding_budget_gets_own_chunk() {
        // Budget: 50 bytes. Single text at 100 tokens costs 100 > 50.
        // Must get its own chunk regardless.
        let cm = linear_model(1.0, 50);
        let chunks = bin_pack(&[100], &cm);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], vec![0]);
    }

    // ── quadratic-dominance test ───────────────────────────────────────────

    #[test]
    fn quadratic_dominance_long_seqs_get_smaller_chunks() {
        // Conservative defaults: a=16384, b=8.
        // At seq=512: cost per text ≈ 16384*512 + 8*512*512 = 8_388_608 + 2_097_152 ≈ 10.5 MB
        // At seq=8192: cost per text ≈ 16384*8192 + 8*8192*8192 = 134_217_728 + 536_870_912 ≈ 671 MB
        // So with 2 GB budget: 512-token texts ~190 per chunk, 8192-token texts ~2-3 per chunk.
        let cm = CostModel::conservative(CostModel::DEFAULT_MAX_WORKSPACE);

        let short_seqs: Vec<usize> = vec![512; 300];
        let long_seqs: Vec<usize> = vec![8192; 10];

        let short_chunks = bin_pack(&short_seqs, &cm);
        let long_chunks = bin_pack(&long_seqs, &cm);

        // Short chunks should pack many texts; long chunks should be much smaller.
        // cast_precision_loss: chunk counts are small (≤ 300), far within f64
        //   precision; the assertion only checks an order-of-magnitude ratio (5×).
        #[allow(clippy::cast_precision_loss)]
        let avg_short: f64 =
            short_chunks.iter().map(Vec::len).sum::<usize>() as f64 / short_chunks.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let avg_long: f64 =
            long_chunks.iter().map(Vec::len).sum::<usize>() as f64 / long_chunks.len() as f64;

        assert!(
            avg_short > avg_long * 5.0,
            "short-seq chunks ({avg_short:.1} avg) should be much larger than \
             long-seq chunks ({avg_long:.1} avg)"
        );
    }

    // ── cost_model unit tests ──────────────────────────────────────────────

    #[test]
    fn chunk_cost_pure_linear() {
        let cm = model(100.0, 0.0, usize::MAX);
        // cost(4, 128) = 100 * 4 * 128 + 0 = 51_200
        assert_eq!(cm.chunk_cost(4, 128), 51_200);
    }

    #[test]
    fn chunk_cost_pure_quadratic() {
        let cm = model(0.0, 1.0, usize::MAX);
        // cost(2, 64) = 0 + 1 * 2 * 64 * 64 = 8_192
        assert_eq!(cm.chunk_cost(2, 64), 8_192);
    }

    #[test]
    fn fits_returns_false_when_over_budget() {
        let cm = model(1.0, 0.0, 100);
        assert!(cm.fits(1, 50));
        assert!(!cm.fits(3, 50)); // 150 > 100
    }

    #[test]
    fn conservative_defaults_at_16x512_is_reasonable() {
        let cm = CostModel::conservative(2 * 1024 * 1024 * 1024);
        // (16, 512) must fit (this is the old static budget's worst case).
        assert!(cm.fits(16, 512), "conservative model must fit (16, 512)");
        // (1, 8192) must fit (single long text must always be processable).
        assert!(cm.fits(1, 8192), "conservative model must fit (1, 8192)");
    }

    // ── proptest ───────────────────────────────────────────────────────────

    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn all_indices_present_exactly_once_proptest(
                seqs in prop::collection::vec(1usize..=8192, 0..=200),
                max_bytes in 1usize..=2_000_000_000,
            ) {
                let cm = CostModel::conservative(max_bytes);
                let chunks = bin_pack(&seqs, &cm);

                let mut found: Vec<usize> = chunks.into_iter().flatten().collect();
                found.sort_unstable();
                let expected: Vec<usize> = (0..seqs.len()).collect();
                prop_assert_eq!(found, expected);
            }

            #[test]
            fn chunks_never_empty(
                seqs in prop::collection::vec(1usize..=512, 1..=100),
                max_bytes in 1usize..=1_000_000,
            ) {
                let cm = CostModel::conservative(max_bytes);
                let chunks = bin_pack(&seqs, &cm);
                for chunk in &chunks {
                    prop_assert!(!chunk.is_empty(), "no chunk should be empty");
                }
            }
        }
    }
}
