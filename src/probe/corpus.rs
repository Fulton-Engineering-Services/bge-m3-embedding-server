//! Probe text synthesis helpers.
//!
//! The probe sweeps `(batch, seq)` shapes by submitting synthesized texts to
//! the leader worker. Texts come from the curated benchmark corpus; we
//! repeat/trim corpus entries to hit the target token count for each shape.

/// Loads the benchmark corpus for use as probe text material.
///
/// Falls back to a tiny built-in sentence if the corpus file is not found.
pub(super) fn load_probe_texts() -> Vec<String> {
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
pub(super) fn synthesize_texts(corpus: &[String], batch: usize, target_seq: usize) -> Vec<String> {
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
