# PR #9 Re-Review Findings — CoreML EP + Direct ORT Migration

**Date**: 2026-03-01
**Branch**: feat/fastembed-removal → main
**Scope**: 23 files, +5377 -837 lines
**Reviewers**: Security, Correctness, Architecture, Tests
**Overall**: Approve with Comments (0 Critical, 4 High, 17 Medium, 14 Low, 6 Info)
**Context**: Second review pass after all original findings were addressed

---

## High Priority

### SEC-1: `ort` pinned to RC with no build-time guard for prebuilt binary download
- **Reviewer**: Security
- **File**: `Cargo.toml:18`
- **Evidence**: `ort = { version = "=2.0.0-rc.11", ... }` — exact pin to RC. If `ORT_LIB_LOCATION` is not set, `ort-sys` may download a prebuilt binary at build time from an external URL whose provenance is outside this project's control.
- **Impact**: Supply-chain vector for new developers running `cargo build` without documentation; RC releases may not receive security advisories through `cargo deny`
- **Recommendation**: Add `build.rs` warning or assertion when `ORT_LIB_LOCATION` is absent, or upgrade to stable once available
- **Status**: Fixed — added `build.rs` with `cargo:warning` when `ORT_LIB_LOCATION` is not set

### COR-1: `tokenize_to_arrays` index-out-of-bounds panic in release builds
- **Reviewer**: Correctness
- **File**: `src/embedder.rs:209-215`
- **Evidence**: `debug_assert!(!encodings.is_empty())` is compiled out in `--release`. If `encode_batch` returns empty, `encodings[0]` panics rather than propagating an error.
- **Impact**: Unrecoverable panic kills worker thread; if all workers hit same trigger, service enters `fail` state
- **Recommendation**: Replace `debug_assert!` with runtime `anyhow::bail!`
- **Status**: Fixed — promoted to runtime `anyhow::bail!`

### ARC-1: `docs/request-flow.md` still references removed fastembed API calls
- **Reviewer**: Architecture
- **File**: `docs/request-flow.md:53, 73`
- **Evidence**: Sequence diagrams reference `TextEmbedding::embed()` and `SparseTextEmbedding::embed()` — types that no longer exist. The pr9-review-findings.md records ARC-1 as "Fixed" but `request-flow.md` was missed.
- **Impact**: Contributors form incorrect mental model of inference dispatch
- **Recommendation**: Update both diagrams to reflect single-session ORT design
- **Status**: Fixed — updated sequence diagrams and prose

### TST-1: `normalize_l2` missing negative-component and single-element test cases
- **Reviewer**: Tests
- **File**: `src/embedder.rs` (test section)
- **Evidence**: Three existing tests don't cover sign preservation (`[-3.0, 4.0]`), single-element vectors, or post-normalization norm-equals-1 assertions. This is the only safety net for dense embedding correctness.
- **Impact**: Undetected regression in normalization would silently corrupt all dense embeddings
- **Recommendation**: Add 3 targeted tests for sign preservation, single-element, and output-norm-is-1
- **Status**: Fixed — added 3 new tests

---

## Medium Priority

### SEC-2: Bundled `sparse_linear.safetensors` has no runtime integrity check
- **Reviewer**: Security
- **File**: `src/weights/mod.rs:18`
- **Evidence**: Size pin (4,236 bytes) is weaker than SHA-256 verification. Documented hash `a2601321...` is not enforced at runtime.
- **Recommendation**: Add compile-time or startup SHA-256 check against documented hash
- **Status**: Open

### SEC-3: No rate limiting on embedding endpoints
- **Reviewer**: Security
- **File**: `src/handler.rs:69,111`
- **Evidence**: Expensive ONNX inference with no per-client throttling on internal LAN service
- **Recommendation**: Add Tower concurrency limit or document network-level controls as mitigation
- **Status**: Open

### SEC-4: `hf-hub` uses native-tls; Docker containers may have minimal CA bundles
- **Reviewer**: Security
- **File**: Cargo.lock (`hf-hub` dep chain)
- **Evidence**: TLS delegates to system keychain; Linux containers may lack full CA bundle
- **Recommendation**: Audit or document CA bundle assumption for container deployments
- **Status**: Open

### SEC-5: `BGE_M3_CACHE_DIR` not validated for path safety
- **Reviewer**: Security
- **File**: `src/config.rs:85`, `src/embedder.rs:62-92`
- **Evidence**: Environment variable used without path normalization. Low exploitability (operator-controlled).
- **Recommendation**: Document as accepted risk given operator-controlled environment
- **Status**: Open

### COR-2: `chunks_exact(4)` silently discards trailing bytes in weight parsing
- **Reviewer**: Correctness
- **File**: `src/weights/mod.rs`
- **Evidence**: No explicit `data().len() % 4 == 0` assertion before `chunks_exact`
- **Recommendation**: Add byte-length assertion before `chunks_exact`
- **Status**: Open

### COR-3: `fp16_eval.rs` embed functions panic on empty scenario text lists
- **Reviewer**: Correctness
- **File**: `examples/fp16_eval.rs`
- **Evidence**: Same `encodings[0]` pattern as COR-1 but in example code
- **Recommendation**: Add early return for empty input
- **Status**: Open

### COR-4: FP16 fallback assumes ORT auto-promotes `last_hidden_state` to F32
- **Reviewer**: Correctness
- **File**: `examples/fp16_eval.rs`
- **Evidence**: Undocumented assumption about ORT auto-casting FP16 tensors
- **Recommendation**: Document assumption or add contextual error message
- **Status**: Open

### ARC-2: Three-way code duplication (binary crate constraint)
- **Reviewer**: Architecture
- **File**: Multiple (`src/embedder.rs`, `benches/coreml.rs`, `examples/fp16_eval.rs`)
- **Evidence**: `download_model_files`, `load_tokenizer`, `load_session`, sparse weight deserialization duplicated 3x
- **Recommendation**: Track `[lib]` crate split; add `REPO_REVISION` drift-detection CI check
- **Status**: Documented (NOTE(ARC-3), NOTE(ARC-4))

### ARC-3: `REPO_REVISION` constant duplicated across 3 files with no drift detection
- **Reviewer**: Architecture
- **File**: `src/embedder.rs:75`, `benches/coreml.rs:121`, `examples/fp16_eval.rs:45`
- **Evidence**: Pin `5617a9f61b028005a4858fdac845db406aefb181` in 3 independent copies
- **Recommendation**: Grep-based CI check asserting unique value
- **Status**: Open

### ARC-4: `docs/pr9-review-findings.md` committed as permanent doc
- **Reviewer**: Architecture
- **File**: `docs/pr9-review-findings.md`
- **Evidence**: Review artifact mixed with reference docs; "Status: Fixed" entries become stale
- **Recommendation**: Move to `docs/decisions/` ADR format or preserve in PR comments
- **Status**: Open

### ARC-5: `run_worker` takes 7 positional arguments
- **Reviewer**: Architecture
- **File**: `src/embedder.rs:444`
- **Evidence**: Growing arg list with Clippy suppression; spans two concerns (identity/lifecycle + execution policy)
- **Recommendation**: Group policy args into `WorkerConfig` struct
- **Status**: Open

### ARC-6: `FastPrediction` hardcoded with no production config escape hatch
- **Reviewer**: Architecture
- **File**: `src/embedder.rs:397-413`
- **Evidence**: No env var to opt out of `FastPrediction` on low-RAM Macs
- **Recommendation**: Expose `BGE_M3_COREML_COMPUTE_UNITS` env var or document the fixed choice
- **Status**: Open

### TST-2: `sparse_project` tests missing zero-weight and negative-bias edge cases
- **Reviewer**: Tests
- **File**: `src/embedder.rs` tests
- **Recommendation**: Add targeted edge-case tests
- **Status**: Open

### TST-3: `sparse_maxpool` missing all-masked-out input test
- **Reviewer**: Tests
- **File**: `src/embedder.rs` tests
- **Recommendation**: Add test with `mask = [0, 0, 0]` for non-special IDs
- **Status**: Open

### TST-4: Weights module missing SHA-256 integrity test
- **Reviewer**: Tests
- **File**: `src/weights/mod.rs`
- **Evidence**: Documented SHA-256 not enforced at test time
- **Recommendation**: Promote documented hash to enforced test assertion
- **Status**: Open

### TST-5: Benchmark corpus `boundary_cases` not verified by CI-runnable assertion
- **Reviewer**: Tests
- **File**: `benches/fixtures/corpus.json`
- **Recommendation**: Lightweight JSON-parse unit test for corpus shape
- **Status**: Open

### TST-6: `sparse_linear_loads_correct_shape` doesn't verify weight vector statistical properties
- **Reviewer**: Tests
- **File**: `src/weights/mod.rs`
- **Recommendation**: Assert `weight.iter().all(|w| w.is_finite())` and non-zero check
- **Status**: Open

---

## Low Priority

| ID | Reviewer | Title | File | Status |
|----|----------|-------|------|--------|
| SEC-6 | Security | `hf-hub` revision pinned but hash not verified by cargo supply chain | `src/embedder.rs:43` | Open |
| SEC-7 | Security | `coreml-profile` feature emits per-op dispatch data to stderr | `src/embedder.rs:372-373` | Open |
| SEC-8 | Security | `deny.toml` uses `unmaintained = "workspace"` not `"deny"` | `deny.toml:5` | Open |
| COR-5 | Correctness | `WorkerGuard::drop` remaining count variable name slightly misleading | `src/embedder.rs:409` | Open |
| COR-6 | Correctness | `weight_correlation` returns `1.0` for single shared index | `examples/fp16_eval.rs` | Open |
| COR-7 | Correctness | `benches/coreml.rs` `tokenize_batch` same `encodings[0]` panic pattern | `benches/coreml.rs` | Open |
| ARC-7 | Architecture | `tokenize_to_arrays` returns `Vec<Encoding>` unused by dense path | `src/embedder.rs:228-260` | Open |
| ARC-8 | Architecture | Sparse weight loading re-implemented in bench | `benches/coreml.rs:195-215` | Open |
| ARC-9 | Architecture | `deny.toml` retains `CDLA-Permissive-2.0` possibly no longer needed | `deny.toml:22` | Open |
| ARC-10 | Architecture | `BGE_M3_ONNX_BATCH_SIZE` documented in 4 places with varying descriptions | Multiple files | Open |
| TST-L1 | Tests | `sparse_maxpool_basic` value assertion order could mislead on failure | `src/embedder.rs` tests | Open |
| TST-L2 | Tests | Handler validation tests duplicate 5-line AppState boilerplate 4x | `src/handler.rs` tests | Open |
| TST-L3 | Tests | `sparse_linear_is_idempotent` couples to `OnceLock` pointer repr | `src/weights/mod.rs` tests | Open |
| TST-L4 | Tests | `onnx_batch_size_uses_platform_default` redundant with defaults test | `src/config.rs` tests | Open |

---

## Positive Observations

- Single-session unification eliminates duplicate model loads — halves idle-timeout memory
- `unsafe_code = "forbid"` enforced at workspace level
- Commit-hash pinning (`REPO_REVISION`) for model supply-chain integrity
- `WorkerGuard` RAII handles clean exit and panic unwind
- Pure-math extraction (`normalize_l2`, `sparse_project`, `sparse_maxpool`) — testable without ORT
- 11 focused unit tests with precise epsilon-based assertions
- Handler validation tests explicitly discriminate `InvalidRequest` vs `ServiceUnavailable`
- Config tests use closure pattern avoiding `env::set_var` race conditions
- Safetensors format for bundled weights (no pickle/execution surface)
- Provenance documentation with SHA-256, tensor shapes, source checkpoint
- `OnceLock` for process-lifetime weight caching, verified by pointer-equality test
- Dramatic dependency tree reduction — removing fastembed eliminates `image`, `rav1e`, `libfuzzer-sys`
- `target-cpu=native` scoped only to `aarch64-apple-darwin` target triple
- Cold-start leader-first ordering prevents hf-hub file-lock contention
- Input validation: `max_length=512` prevents unbounded allocation (DoS mitigation)
- `onnx_batch_size` clamped to min 1 — prevents `texts.chunks(0)` infinite iterator
