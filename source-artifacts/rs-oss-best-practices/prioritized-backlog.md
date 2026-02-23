All 6 layer 2 researchers complete. Synthesizing the final report now.

---

## Final Research Report: OSS Best Practices Audit — `bge-m3-axum-fastembed-rs`

---

### Research Tree

```
bge-m3-axum-fastembed-rs OSS Audit
├── 1. Project Configuration & Cargo.toml ✓
├── 2. CI/CD & Release Automation ✓
├── 3. Code Quality & Linting ✓
├── 4. Error Handling & Observability ✓
│   └── A1: Worker post-readiness panic [drilled]
│       ├── A1.1: Panic detection mechanics ✓
│       ├── A1.2: Health endpoint degradation ✓
│       └── A1.3: Worker restart / supervision ✓
├── 5. Testing Strategy ✓
│   ├── E1: Router-level integration tests ✓
│   ├── E2: Model-gated test strategy ✓
│   └── E3: main.rs startup testability ✓
├── 6. Documentation & Developer Experience ✓
└── 7. Security & Supply Chain ✓
```

---

### Executive Summary

The project is a clean, well-structured Rust service with several genuine strengths: committed `Cargo.lock`, working `cargo-deny` with a thorough license allowlist, `-D warnings` enforced in CI, a good README with architecture diagram, no unsafe code, and a coherent OpenAI-compatible API design.

The audit found **one correctness defect**, **one tier of high-impact security gaps**, and **two tiers of OSS maturity gaps**. The correctness defect is the most urgent: Tokio explicitly swallows panics when a `JoinHandle` is dropped (confirmed in Tokio source: *"we assume they are not interested in the panic and swallow it"*), meaning a post-readiness worker death is invisible — the `AtomicBool` flag is write-once and `/health` returns 200 indefinitely even with a dead pool. The fix is well-defined and contained to `~50 lines` across three files.

---

### Categorized Opportunity List

---

#### Category A — Reliability & Observability

**A1 — Silent worker death with false-healthy health signal** *(Correctness Defect)*
> Layer 2 drill produced a fully-designed fix.

The root mechanics (confirmed via Tokio source): `drop(worker_handles)` at `embedder.rs:137` detaches all workers permanently. Any post-readiness panic is caught by Tokio's harness and discarded. The `Arc<Mutex<Receiver>>` topology means a single worker death doesn't close the channel — surviving workers continue, clients see occasional 500s, health stays 200.

**Recommended fix** (Option A — degrade + delegate restart to Docker; Option B/supervisor ruled out due to ~1GB-per-worker reload spike risk; Option C/catch_unwind ruled out because ONNX model state is corrupted after panic):

Three targeted changes:

1. **`src/embedder.rs`** — Add `Arc<AtomicUsize> live_workers` field to `EmbedPool`. Add a RAII `WorkerGuard` whose `Drop` decrements the counter on any exit including panic unwind. Initialize to `n` before spawning.

2. **`src/state.rs`** — Add `live_workers: Arc<AtomicUsize>` and `total_workers: usize` to `AppState`.

3. **`src/handler.rs`** — Replace the write-once `AtomicBool` health check with a three-state model:

| State | `live_workers` | `/health` HTTP | body `status` |
|-------|---------------|----------------|--------------|
| Loading | 0 (pre-init) | 503 | `"loading"` |
| All alive | N == total | 200 | `"ok"` |
| Partially alive | 1..N-1 | 200 | `"warn"` + worker counts |
| All dead | 0 (post-init) | 503 | `"fail"` + worker counts |

Also update `check_ready()` to return `AppError::ServiceUnavailable` (503) when `live_workers == 0`, instead of letting it fall through to the pool channel-closed path which returns 500.

The Docker `HEALTHCHECK --retries=3 --interval=10s` + `restart: unless-stopped` already in `compose.yml` handles container restart automatically once `/health` returns 503. Total recovery time: ~30s detection + ~15-30s ONNX init from cache = ~60s worst case.

---

**A2 — No `#[instrument]` on handler functions** *(Medium impact, low effort)*
Neither `dense_embeddings` nor `sparse_embeddings` has `#[instrument]`. Operators cannot correlate slow responses with `batch_size` in distributed traces. Add `#[instrument(skip(state), fields(batch_size = %texts.len()))]` to both handlers.

**A3 — Tracing subscriber text-only — no JSON output** *(Medium impact, low effort)*
`tracing-subscriber` has only `features = ["env-filter"]`. Log aggregators (Loki, Datadog, CloudWatch) cannot parse human-readable output. Add `"json"` to features and a `BGE_M3_LOG_FORMAT=json` env var that switches the subscriber format.

**A4 — `anyhow` errors throughout worker pool — error types unclassifiable** *(Medium impact, medium effort)*
Channel-closed errors (pool crashed → 503) and ONNX inference errors (transient → 500) both produce `AppError::Internal`. A small `EmbedError` enum with typed variants would enable correct status mapping. Deferred until ARC-2 is resolved.

**A5 — Design-doc observability items unimplemented** *(Low impact, low effort)*
`tower-http` `RequestId` layer not wired; model load duration not logged. The design doc specified both. Two low-effort additions to `main.rs`.

---

#### Category B — Security & Supply Chain

**B1 — GitHub Actions use mutable version tags** *(High impact, low effort)*
All 13 action references (`@v4`, `@stable`, `@v2`, etc.) are mutable tags. The release pipeline holds `contents: write` + `packages: write`. A compromised tag executes arbitrary code with those permissions — the same attack vector as CVE-2025-30066 (`tj-actions/changed-files`).
Fix: pin all actions to commit SHAs using `pin-github-action` tool, then add `github-actions` ecosystem to Dependabot to keep pins current.

**B2 — Dockerfile runs as root** *(High impact, low effort)*
No `USER` instruction anywhere in the 35-line Dockerfile. Any RCE gives host kernel surface. Fix: add `RUN useradd -u 1001 -r bge && USER bge` before `CMD`.

**B3 — Workflow `permissions:` blocks absent/too broad** *(High impact, low effort)*
`ci.yml` has no `permissions:` block (inherits repo default). `release.yml` grants `contents: write` + `packages: write` at the workflow level, meaning quality-gate jobs (check, test, deny) receive write access they don't need. Fix: add `permissions: contents: read` to `ci.yml`; scope write permissions to only the build/release jobs in `release.yml`.

**B4 — No SBOM, no image signing, no SLSA provenance** *(High impact, medium effort)*
Release pipeline builds multi-arch images to GHCR but produces no attestations. `id-token: write` permission is absent (required for OIDC signing). Fix: add `attestations: true` to `docker/build-push-action`, or add `actions/attest-build-provenance` after manifest push.

**B5 — Dockerfile base image not pinned by digest** *(Medium impact, low effort)*
`FROM ubuntu:24.04` is mutable. Non-reproducible builds. Fix: pin with `@sha256:<digest>`.

**B6 — No `#![forbid(unsafe_code)]`** *(Medium impact, trivial)*
No unsafe code currently exists, but no compile-time guarantee prevents future introduction. One-line addition to `src/main.rs`.

**B7 — `deny.toml`: `yanked = "warn"` not `"deny"`** *(Medium impact, trivial)*
Yanked crates (which may indicate security issues) don't block CI. Change to `yanked = "deny"`.

**B8 — No `dependabot.yml`** *(Medium impact, low effort)*
No automated PR-based updates for Cargo or GitHub Actions dependencies. CVE patches in transitive deps require manual `cargo update`. Add `.github/dependabot.yml` covering both `cargo` and `github-actions` ecosystems (weekly schedule).

**B9 — `curl` in runtime Docker image** *(Low impact, low effort)*
Installed solely for the `HEALTHCHECK` command, unnecessarily expanding attack surface. Replace with an inline shell command against the server's TCP port, or rely on the orchestrator's probe against the exposed `/health` endpoint.

---

#### Category C — CI/CD & Automation

**C1 — No MSRV CI job** *(High impact, low effort)*
No `rust-version` field in `Cargo.toml` and no CI job validating it. Consumers have no contractual Rust version guarantee. Fix: add `rust-version = "1.75"` (approximate based on axum/tokio requirements) to `Cargo.toml`; add a matrix entry `dtolnay/rust-toolchain@"1.75"` in the test job.

**C2 — No code coverage** *(High impact, low effort)*
34 unit tests, `main.rs` (89 lines of critical startup logic) has 0 tests and no way to surface this gap. Fix: add `cargo-llvm-cov` step with Codecov upload; add badge to README.

**C3 — No Rust channel matrix** *(Medium impact, low effort)*
CI only runs against `stable`. Beta failures give advance warning before they hit stable. Fix: add `strategy.matrix.rust: [stable, beta]` to the test job; add MSRV slot once C1 is addressed.

**C4 — No CI concurrency cancellation** *(Low impact, trivial)*
Successive pushes queue redundant CI runs. Fix: add `concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }` to `ci.yml`.

**C5 — `--no-tests=pass` silences empty test suite** *(Low impact, trivial)*
Deleting all tests would produce a green CI run. Change to `--no-tests=warn` or remove the flag.

---

#### Category D — Code Quality & Linting

**D1 — No `[lints]` table in `Cargo.toml`** *(High impact, low effort)*
The Rust 1.74+ idiomatic lint policy mechanism is absent. Current lint config exists only as CI flags. Fix:
```toml
[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
```

**D2 — No crate-level lint attributes** *(High impact, low effort)*
`src/main.rs` has zero `#![...]` attributes. IDE and local `cargo clippy` runs use default lint set only. Fix: add `#![warn(clippy::all, clippy::pedantic)]` and `#![forbid(unsafe_code)]` (or rely on D1's `[lints]` table).

**D3 — No `rustfmt.toml`** *(Medium impact, low effort)*
Format config is implicit. Toolchain upgrades can silently reformat the entire codebase. Fix: add `rustfmt.toml` with at minimum `edition = "2021"`, `imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`.

**D4 — CI clippy doesn't enable `clippy::pedantic`** *(Medium impact, low effort)*
High-value lints (`must_use_candidate`, `needless_pass_by_value`, `explicit_iter_loop`) not enforced. Fix: add `-W clippy::pedantic` to the CI clippy invocation; add selective `#[allow]` with justification comments where genuinely noisy.

**D5 — 4 bare `.unwrap()` calls in test code** *(Low impact, trivial)*
Inconsistent with the rest of the codebase which uses `.expect("reason")`. Fix: replace with `.expect("descriptive message")`.

---

#### Category E — Testing Strategy

**E1 — No router-level integration tests** *(High impact, medium effort)*
> Layer 2 drill produced concrete, compilable test code.

Zero tests exercise the full HTTP path through the Axum router. Key untested paths: 422 JSON rejection, 413 body limit, 415 missing content-type, 405 wrong method, and the complete `AppError::IntoResponse` chain.

**Prerequisite**: Extract `pub(crate) fn build_router(state: Arc<AppState>) -> Router` from `main()` (6-line change — the router construction is already a pure function).

**Required dev-dependencies** (declare explicitly; already transitive via axum):
```toml
[dev-dependencies]
tower = { version = "0.5", features = ["util"] }
http-body-util = "0.1"
```

**Tests unlocked** (11 new tests in `src/main.rs`, no model loading required):
- `GET /health` → 503 when not ready, 200 when ready
- `POST /v1/embeddings` → 503 when not ready, 422 wrong type, 422 missing field, 400 syntax error, 415 missing content-type, 413 oversized body, 405 wrong method
- `POST /v1/sparse-embeddings` → 503 when not ready, 422 wrong type

Note: The tests must go in `src/main.rs` (not `tests/`), because `build_router` and `EmbedPool::closed_for_test` are `pub(crate)`.

---

**E2 — Happy-path response mapping untested** *(High impact, medium effort)*
> Layer 2 drill produced a concrete implementation sketch.

The token counting formula, `*i as u32` index cast, `object: "embedding"` / `"list"` literals, and `model: "bge-m3"` constant are all untested because `closed_for_test()` makes every happy-path call error before reaching the mapping code.

**Recommended fix**: Add a second `#[cfg(test)]` constructor to `EmbedPool`:

```rust
pub(crate) fn with_fixture_response(
    dense_fixture: Vec<Vec<f32>>,
    sparse_fixture: Vec<fastembed::SparseEmbedding>,
) -> Self
```

This mirrors the existing `closed_for_test()` idiom exactly. Spawns a lightweight background task (not `spawn_blocking`) that returns pre-canned vectors for every request regardless of input, enabling testing of all response-mapping logic without any ONNX code path.

**Important**: Before writing the fixture construction, verify `fastembed::SparseEmbedding.indices` field type — the handler casts `*i as u32`, suggesting the source type may be `Vec<i64>`. The test must use the actual source type.

For tests genuinely requiring ONNX inference (e.g., verifying BGE-M3 produces 1024-dimensional dense vectors): use `#[ignore]` and document in CLAUDE.md as `cargo nextest run --run-ignored` with `BGE_M3_CACHE_DIR` populated.

---

**E3 — `main.rs` startup logic 100% untested** *(High impact, medium effort)*
> Layer 2 drill produced exact function signatures and test cases.

Four `process::exit(1)` call sites in the readiness probe task cannot be observed or asserted by any test.

**Recommended refactoring** (zero behavioral change, three extractions):

1. `build_router(state: Arc<AppState>) -> Router` — already covered by E1
2. `async fn run_readiness_probe(init_handle: JoinHandle<anyhow::Result<()>>, state: Arc<AppState>) -> anyhow::Result<()>` — replaces `process::exit(1)` with `Err(...)` returns; `main()` calls it and handles the error
3. Change `tracing_subscriber::fmt().init()` → `.try_init().ok()` to be safe under repeated test invocations

**Tests unlocked** (4 new tests, no model loading required, using `closed_for_test()`):
- `init_handle` returns `Err` → probe returns error, `ready` stays false
- `init_handle` panics → probe returns error
- `init_handle` succeeds but dense warm-up fails → probe returns error
- Probe failure → `ready` is never set to `true`

---

**E4 — No `[dev-dependencies]`** *(Medium impact, trivial)*
Structural barrier to adding test utilities. Add the section proactively (triggered by E1 needing `tower` + `http-body-util`).

**E5 — No property-based tests for `TextInput`/`validate_input`** *(Medium impact, low effort)*
`serde(untagged)` has historically surprising behavior; char-count limit at 32,768 is a security control. Add `proptest` to dev-dependencies and fuzz the `TextInput` deserialization and `validate_input` boundaries.

**E6 — No benchmark suite** *(Low impact, medium effort)*
No performance regression detection for ONNX Runtime version upgrades. Add a `criterion` benchmark for worker pool throughput at batch sizes 1/16/64/256.

---

#### Category F — Project Configuration

**F1 — No MSRV declaration** *(High impact, low effort)*
No `rust-version` field. `cargo add` resolver can't assist; CI can't enforce. Add `rust-version = "1.75"` (verify actual minimum).

**F2 — No `[profile.release]`** *(Medium impact, low effort)*
Shipped binary uses Cargo defaults: no LTO, 16 codegen units, no strip, `panic = "unwind"`. For a compute-bound inference server:
```toml
[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
panic = "abort"
```

**F3 — `tokio = { features = ["full"] }`** *(Low impact, low effort)*
Enables rarely-needed features. Replace with `["rt-multi-thread", "macros", "net", "signal"]`.

**F4 — `[package]` metadata incomplete** *(Low impact, trivial)*
Missing `description`, `repository`, `keywords`, `categories`, `readme`. Add all five.

---

#### Category G — Documentation & Developer Experience

**G1 — No `SECURITY.md`** *(High impact, low effort)*
No vulnerability disclosure policy or contact channel. Required for OpenSSF Best Practices badge compliance.

**G2 — No `CONTRIBUTING.md`** *(Medium impact, low effort)*
No dev setup, test instructions, or PR process guidance for contributors.

**G3 — No `CHANGELOG.md`** *(Medium impact, low effort)*
Release notes are GitHub auto-generated text. Add structured Keep a Changelog format; automate with `git-cliff` (which reads the existing Conventional Commits-style history).

**G4 — No OpenAPI spec** *(Medium impact, medium effort)*
Consumers must reverse-engineer JSON shapes from README prose. Add an OpenAPI 3.x spec via `utoipa` crate, enabling auto-generated client stubs and contract testing.

**G5 — No README badges** *(Low impact, trivial)*
No CI status, license, or Docker image badges. Add GitHub Actions badge, license badge, GHCR badge.

**G6 — Sparse doc comment coverage** *(Low impact, low effort)*
`models.rs` is well-documented; `embedder.rs`, `handler.rs`, and `config.rs` are not. No `#![warn(missing_docs)]` enforcement.

**G7/G8 — No `CODE_OF_CONDUCT.md` / no issue+PR templates** *(Low impact, low effort)*
GitHub community profile is incomplete.

---

### Prioritized Backlog View

| Priority | Item | Category | Effort |
|----------|------|----------|--------|
| 🔴 P0 | Worker death: RAII counter + three-state health + check_ready 503 | A1 | Medium |
| 🔴 P1 | GitHub Actions SHA pinning | B1 | Low |
| 🔴 P1 | Workflow `permissions:` blocks | B3 | Low |
| 🔴 P1 | Dockerfile: add `USER` instruction | B2 | Low |
| 🟠 P2 | Extract `build_router()` + router-level tests (11 tests) | E1/E3 | Medium |
| 🟠 P2 | `EmbedPool::with_fixture_response()` + happy-path tests | E2 | Medium |
| 🟠 P2 | Extract `run_readiness_probe()` + startup failure tests (4 tests) | E3 | Medium |
| 🟠 P2 | `[lints]` table in `Cargo.toml` | D1 | Low |
| 🟠 P2 | `#[instrument]` on handlers | A2 | Low |
| 🟠 P2 | JSON logging support | A3 | Low |
| 🟠 P2 | MSRV declaration + CI job | F1/C1 | Low |
| 🟠 P2 | `SECURITY.md` | G1 | Low |
| 🟡 P3 | Dependabot for Cargo + Actions | B8/C CI | Low |
| 🟡 P3 | SBOM + image signing | B4 | Medium |
| 🟡 P3 | Code coverage (llvm-cov + Codecov) | C2 | Low |
| 🟡 P3 | `rustfmt.toml` | D3 | Low |
| 🟡 P3 | `[profile.release]` | F2 | Low |
| 🟡 P3 | `CONTRIBUTING.md`, `CHANGELOG.md` | G2/G3 | Low |
| 🟢 P4 | Dockerfile digest pinning, curl removal | B5/B9 | Low |
| 🟢 P4 | `clippy::pedantic` in CI | D4 | Low |
| 🟢 P4 | Property-based tests for `TextInput` | E5 | Low |
| 🟢 P4 | Rust channel matrix (stable + beta) | C3 | Low |
| 🟢 P4 | OpenAPI spec | G4 | Medium |
| 🟢 P5 | Everything else (tokio/full, badges, metadata, etc.) | F/G | Trivial |

`★ Insight ─────────────────────────────────────`
The layer 2 drill on A1 confirmed something architecturally interesting: Tokio's decision to silently swallow panics on dropped `JoinHandle`s is an intentional API contract (stated explicitly in source comments), not a bug. The correct Rust idiom for observable worker lifecycle is RAII `Drop` guards + `AtomicUsize` counters — not `catch_unwind`, which is especially dangerous here because the ONNX FFI state is corrupted after a panic and cannot safely continue. The pattern `let _guard = WorkerGuard(counter)` in a `spawn_blocking` closure ensures the decrement fires on any exit including panic unwind, because `Drop` runs during stack unwinding.
`─────────────────────────────────────────────────`

`★ Insight ─────────────────────────────────────`
The E1/E3 drill revealed that the single biggest unlock for the test suite is a 6-line refactor: extracting `build_router()` from `main()`. This one change enables 11 new router-level tests (422/413/415/405 rejections, health state transitions, middleware chain verification) plus unlocks the readiness probe extraction. The existing `EmbedPool::closed_for_test()` constructor was already designed for exactly this kind of use — the infrastructure is in place, it just hasn't been connected to the router layer.
`─────────────────────────────────────────────────`