# PR #9 Review Findings — CoreML EP + Direct ORT Migration

**Date**: 2026-03-01
**Branch**: feat/fastembed-removal → main
**Scope**: 21 files, +4840 -835 lines
**Reviewers**: Security, Correctness, Architecture, Tests
**Overall**: Approve with Comments (0 Critical, 4 High, 10 Medium, 9 Low, 3 Info)

---

## High Priority

### SEC-1: Model files downloaded from HuggingFace Hub without integrity verification
- **File**: `src/embedder.rs:55-82`
- **Evidence**: `hf_hub::api::sync::Api` downloads ONNX model files over HTTPS with no SHA-256 hash verification against a pinned manifest
- **Impact**: Compromised HuggingFace repo or MITM could substitute adversarial weights affecting search/memory retrieval
- **Recommendation**: Pin expected SHA-256 hashes and verify post-download; or document known-good hashes for manual audit
- **Status**: Fixed — pinned HF repo to immutable commit revision `5617a9f61b028005a4858fdac845db406aefb181` in all 3 files (src/embedder.rs, benches/coreml.rs, examples/fp16_eval.rs) using `Repo::with_revision()`

### COR-1 / SEC-5: Unchecked bias tensor data length — panic on malformed safetensors
- **File**: `src/weights/mod.rs:29-34` (also `benches/coreml.rs:196-201`, `examples/fp16_eval.rs:84-89`)
- **Evidence**: `bias_view.data()[0..3]` indexed directly without length guard. Weight has `assert_eq!(weight.len(), 1024)` but bias has none.
- **Impact**: Corrupted/updated safetensors would panic at startup (process-fatal with `panic = "abort"`)
- **Recommendation**: Add `assert_eq!(bias_data.len(), 4, "sparse_linear bias must be scalar F32")` in all three locations
- **Status**: Fixed — added length assertion in all three locations

### ARC-1: Architecture documentation describes old fastembed design
- **File**: `docs/architecture.md:155-167`, `docs/bge-m3-model.md:13`
- **Evidence**: References `TextEmbedding`/`SparseTextEmbedding` (removed fastembed types). Mermaid diagram shows removed types. `bge-m3-model.md` says server "wraps fastembed-rs."
- **Impact**: Misleading for contributors; CLAUDE.md was updated but docs/ tree lags behind
- **Recommendation**: Update docs to reflect single-session design, add weights module to layout table, remove fastembed references
- **Status**: Fixed — updated architecture.md and bge-m3-model.md

### TST-1: Core inference functions have zero unit test coverage
- **File**: `src/embedder.rs:88-282`
- **Evidence**: `load_tokenizer()`, `load_session()`, `embed_dense()`, `embed_sparse()`, `load_models()` have no unit tests. `EmbedPool::with_fixed_responses` bypasses all inference code.
- **Impact**: Regressions in L2 normalization, zero-norm guard, special-token filtering, or max-pooling pass all 72 tests
- **Recommendation**: Extract pure-math functions (L2 norm, sparse projection, max-pooling) into testable units without ORT dependency
- **Status**: Fixed — extracted `normalize_l2`, `sparse_project`, and `sparse_maxpool` as pure helpers; refactored `embed_dense`/`embed_sparse` to call them; added 11 unit tests covering normal cases, edge cases (zero-norm, negative scores), special-token filtering, attention-mask, and sorted output

### TST-2: Benchmark as integration test has no CI path
- **File**: `benches/coreml.rs`
- **Evidence**: Only code exercising real ORT inference is the benchmark, requires `ORT_LIB_LOCATION` with CoreML EP — unavailable in CI
- **Recommendation**: Document the CI gap; consider self-hosted macOS runner for end-to-end validation
- **Status**: Fixed — added TODO comment documenting CI gap and requirements

---

## Medium Priority

### SEC-2: ORT pinned to release candidate (`=2.0.0-rc.11`)
- **File**: `Cargo.toml:18`
- **Impact**: RCs don't receive security patch backports; not in OSS CVE databases for cargo-deny
- **Recommendation**: Track ort 2.0 stable; migrate when it ships
- **Status**: Deferred — will address when ort 2.0 stable ships

### SEC-3: Bundled sparse_linear.safetensors has no provenance documentation
- **File**: `src/weights/`
- **Impact**: Auditor cannot confirm weights are genuine BGE-M3 extraction
- **Recommendation**: Document extraction commands, source checkpoint SHA, bundled file SHA
- **Status**: Fixed — added comprehensive provenance comment to `src/weights/mod.rs` documenting source checkpoint, SHA-256, tensor shapes, and file size

### COR-2: `encodings[0]` index without empty-batch guard
- **File**: `src/embedder.rs:150,216`
- **Impact**: Theoretical panic if `encode_batch` returns empty vec for non-empty input
- **Recommendation**: Add debug-assert if `encodings.is_empty()`
- **Status**: Fixed — extracted `tokenize_to_arrays` shared helper with `debug_assert!(!encodings.is_empty())`; both `embed_dense` and `embed_sparse` now call it

### COR-3: `stats()` in fp16_eval panics on empty slice
- **File**: `examples/fp16_eval.rs:358-363`
- **Impact**: Misleading output (`inf`/`NaN`) if all sparse overlaps are NaN
- **Recommendation**: Add early return for empty input
- **Status**: Fixed — added early return `(NaN, NaN, NaN)` for empty slice

### COR-4: `loaded_workers` not decremented on clean channel-close exit
- **File**: `src/embedder.rs:428`
- **Impact**: Health may report `ok` instead of `fail` after shutdown — low-risk accounting bug
- **Recommendation**: Decrement `loaded_workers` when worker exits with models still loaded
- **Status**: Fixed — added `loaded_workers.fetch_sub(1, ...)` in the `Ok(None)` (channel-closed) branch when models are still loaded

### COR-5: fp16_eval tokenization inconsistency (batched dense vs single sparse)
- **File**: `examples/fp16_eval.rs:199-211`
- **Impact**: Eval tool inconsistency, not production code
- **Recommendation**: Use batched tokenization for both paths
- **Status**: Fixed — changed sparse tokenization to `encode_batch(vec![text.as_str()], true)`

### ARC-2: embed_dense and embed_sparse duplicate tokenize→tensor→run pipeline
- **File**: `src/embedder.rs:135-282`
- **Recommendation**: Extract shared `tokenize_to_tensors` helper
- **Status**: Fixed — merged with COR-2; both functions now call `tokenize_to_arrays`

### ARC-3: Benchmark fully duplicates production model loading + inference logic
- **File**: `benches/coreml.rs:116-338`
- **Recommendation**: Promote pure embedding functions for bench/test access
- **Status**: Documented — cannot eliminate; binary crate has no `[lib]` section so bench/examples can't import. Added NOTE(ARC-3) documenting intentional duplication and legitimate behavioral differences (RefCell, .expect(), custom EP configs)

### ARC-4: fp16_eval re-implements load_tokenizer, load_session, sparse weights (3rd copy)
- **File**: `examples/fp16_eval.rs:96-137`
- **Recommendation**: Same root cause as ARC-3; three-way duplication is fragile
- **Status**: Documented — same binary-crate constraint as ARC-3. Added NOTE(ARC-4) documenting intentional duplication and FP16 fallback logic unique to this example

### ARC-5: docs module table omits weights module; config table lacks BGE_M3_ONNX_BATCH_SIZE
- **File**: `docs/architecture.md`
- **Recommendation**: Update documentation tables
- **Status**: Fixed — already addressed in high-priority ARC-1 documentation update

### TST-3: Weights module tests lack invalid-input/shape-mismatch cases
- **File**: `src/weights/mod.rs:41-58`
- **Recommendation**: Add test for truncated/corrupted bytes
- **Status**: Fixed — added `bundled_file_is_valid_safetensors` and `bundled_file_size_matches` (pinned at 4,236 bytes) tests

### TST-4: `bias.abs() < 100.0` assertion is vacuous
- **File**: `src/weights/mod.rs:49`
- **Recommendation**: Assert actual known bias value or at minimum non-zero + finite
- **Status**: Fixed — replaced with precise known-value check `(*bias - 0.045_196_53).abs() < 1e-6`

### TST-5: Handler tests can't exercise validate_input when pool is ready
- **File**: `src/handler.rs`
- **Recommendation**: Use `with_fixed_responses` in ready state for input validation tests
- **Status**: Fixed — added 4 handler tests using `with_fixed_responses` + `ready=true` that confirm `InvalidRequest` (not `ServiceUnavailable`) for empty-input and over-batch on both dense and sparse endpoints

### TST-6: Benchmark corpus lacks boundary edge cases
- **File**: `benches/fixtures/corpus.json`
- **Recommendation**: Add 1-3 token texts and 512-token boundary texts
- **Status**: Fixed — added `boundary_cases` scenario with 6 short texts (1-16 chars) and 3 near-512-token-limit texts (~2000-2100 chars)

---

## Low Priority

| ID | Reviewer | Title | File |
|----|----------|-------|------|
| SEC-4 | Security | `target-cpu=native` non-reproducible binaries | `.cargo/config.toml:5` |
| SEC-6 | Security | deny.toml stale fastembed comment | `deny.toml:22` |
| COR-6 | Correctness | Unused `fp16_onnx_bytes` memory waste | `examples/fp16_eval.rs:396-401` |
| COR-7 | Correctness | Bench `.unwrap()` without message | `benches/coreml.rs:229-230` |
| ARC-6 | Architecture | Benchmark loads models twice | `benches/coreml.rs:344-465` |
| ARC-7 | Architecture | coreml-profile underdocumented for bench use case | `Cargo.toml:31-33` |
| ARC-8 | Architecture | target-cpu=native applies to all dev builds | `.cargo/config.toml:4-5` |
| TST-7 | Tests | bench_embed_sparse discards output (no black_box) | `benches/coreml.rs:273-338` |
| TST-9 | Tests | fp16_eval exits 0 on Phase A failure | `examples/fp16_eval.rs:533-543` |

---

## Positive Observations

- `unsafe_code = "forbid"` enforced at workspace level
- Single-session design eliminates duplicate model loads, validated as correct
- `WorkerGuard` RAII handles clean exit and panic unwind
- Cold-start leader-first ordering prevents hf-hub file-lock contention
- `Config::from_lookup` with closure enables parallel-safe unit tests
- `cargo deny` + `Cargo.lock` checksums provide supply-chain protection
- Tokenizer `max_length=512` prevents unbounded allocation (DoS mitigation)
- `REPO_ID` hardcoded (no SSRF vector)
- Sparse projection math confirmed correct against BGE-M3 specification
