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

// -----------------------------------------------------------------------
// REPO_REVISION drift detection (ARC-3)
// -----------------------------------------------------------------------

fn extract_const_str(path: &str, const_name: &str) -> String {
    let prefix = format!("const {const_name}");
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            let start = trimmed.find('"').expect("missing opening quote");
            let end = trimmed[start + 1..]
                .find('"')
                .expect("missing closing quote");
            return trimmed[start + 1..start + 1 + end].to_string();
        }
    }
    panic!("{const_name} not found in {path}");
}

#[test]
fn repo_revision_consistent_across_all_copies() {
    let embedder = extract_const_str("src/embedder/model_files.rs", "REPO_REVISION");
    let bench = extract_const_str("benches/coreml/setup.rs", "REPO_REVISION");
    let example = extract_const_str("examples/fp16_eval.rs", "REPO_REVISION");

    assert_eq!(
        embedder, bench,
        "REPO_REVISION mismatch: src/embedder/model_files.rs ({embedder}) != benches/coreml/setup.rs ({bench})"
    );
    assert_eq!(
        embedder, example,
        "REPO_REVISION mismatch: src/embedder/model_files.rs ({embedder}) != examples/fp16_eval.rs ({example})"
    );
    assert_eq!(embedder.len(), 40, "REPO_REVISION should be a 40-char SHA");
    assert!(
        embedder.chars().all(|c| c.is_ascii_hexdigit()),
        "REPO_REVISION should be hexadecimal"
    );
}

#[test]
fn xenova_repo_revision_consistent_across_all_copies() {
    let embedder = extract_const_str("src/embedder/model_files.rs", "XENOVA_REPO_REVISION");
    let bench = extract_const_str("benches/coreml/setup.rs", "XENOVA_REPO_REVISION");

    assert_eq!(
        embedder, bench,
        "XENOVA_REPO_REVISION mismatch: \
         src/embedder/model_files.rs ({embedder}) != benches/coreml/setup.rs ({bench})"
    );
    assert_eq!(embedder.len(), 40);
    assert!(embedder.chars().all(|c| c.is_ascii_hexdigit()));
}

// -----------------------------------------------------------------------
// Benchmark corpus shape validation (TST-5)
// -----------------------------------------------------------------------

#[test]
fn benchmark_corpus_has_expected_shape() {
    let content = std::fs::read_to_string("benches/fixtures/corpus.json")
        .expect("corpus.json must be readable from project root");
    let corpus: serde_json::Value =
        serde_json::from_str(&content).expect("corpus.json must be valid JSON");

    assert!(corpus.get("metadata").is_some(), "must have 'metadata' key");
    assert!(
        corpus.get("scenarios").is_some(),
        "must have 'scenarios' key"
    );

    let sources = &corpus["metadata"]["sources"];
    assert_eq!(sources["knowledgebase_chunks"]["count"], 50);
    assert_eq!(sources["coordinator_vector_store"]["count"], 75);
    assert_eq!(sources["codekeeper_symbols"]["count"], 50);
    assert_eq!(sources["boundary_cases"]["count"], 9);

    let scenarios = corpus["scenarios"]
        .as_object()
        .expect("scenarios must be object");
    let expected: &[(&str, usize)] = &[
        ("document_chunks", 50),
        ("tool_descriptions", 75),
        ("code_symbols", 50),
        ("boundary_cases", 9),
    ];
    for &(name, count) in expected {
        let texts = scenarios
            .get(name)
            .and_then(|s| s.get("texts"))
            .and_then(|t| t.as_array())
            .unwrap_or_else(|| panic!("scenarios.{name}.texts must be an array"));
        assert_eq!(texts.len(), count);
    }

    let total: usize = scenarios
        .values()
        .filter_map(|s| s.get("texts").and_then(|t| t.as_array()).map(Vec::len))
        .sum();
    assert_eq!(total, 184, "total corpus texts should be 184");
}
