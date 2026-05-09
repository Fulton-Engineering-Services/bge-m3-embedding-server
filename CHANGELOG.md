# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.15.0-rc1] - 2026-05-09

### Added
- **Unified dense + sparse embeddings endpoint** (`POST /v1/embeddings:both`) — runs **one** ONNX `session.run()` per chunk and projects both dense (CLS-pooled, L2-normalized) and sparse (SPLADE-style) outputs from the same forward pass. Numerically equivalent to calling `/v1/embeddings` and `/v1/sparse-embeddings` separately on the same inputs, within FP rounding tolerance, but at near-zero marginal GPU cost. Existing `/v1/embeddings` and `/v1/sparse-embeddings` endpoints are unchanged.
- `EmbedPool::both(texts)` Rust API, `Both` job variant in the worker pool, and `embed_both` shared-pass implementation.
- `DualRequest` / `DualResponse` / `DualEmbeddingData` request/response types.
- Handler-level validation tests, router-level tests, and an opt-in (`BGE_M3_EQUIVALENCE_TEST=1`) integration test verifying that the dual-pass path matches separate dense + sparse passes within FP tolerance.
- OpenAPI documentation for the new endpoint plus `DualRequest`/`DualResponse`/`DualEmbeddingData` schemas.

## [0.14.0] - 2026-05-09

### Added
- **Long-context embeddings** — supports up to 8192 tokens (BGE-M3's full positional range), configurable via `BGE_M3_MAX_SEQ_LENGTH`
- **Memory-aware auto-budget** — startup probe on Linux fits a quadratic cost model `a × N + b × N²` to measured RSS deltas; derives `max_workspace_bytes` from detected cgroup/`/proc` memory minus a safety factor
- **Length-aware bin-packing** (`src/binpack.rs`) — groups texts into ONNX `session.run()` calls that fit within the workspace ceiling; padding is per-chunk, not global
- `BGE_M3_DISABLE_AUTO_BUDGET`, `BGE_M3_AVAILABLE_MEMORY_BYTES`, `BGE_M3_MEMORY_SAFETY_FACTOR`, `BGE_M3_TOKEN_BUDGET`, `BGE_M3_COST_MODEL_A/B` configuration knobs
- `tuning` block in the `/health` response exposing fitted cost-model coefficients, memory source, and workspace ceiling
- `CONTRIBUTING.md` and `.github/dco.yml` (Developer Certificate of Origin enforcement)
- DCO, MSRV, and corrected license badges in README

### Changed
- `BGE_M3_ONNX_BATCH_SIZE` is **deprecated**; if set, a warning is logged and the value is translated to `BGE_M3_TOKEN_BUDGET` for backward compatibility
- Default `BGE_M3_MAX_SEQ_LENGTH` increased from 512 to 8192

### Fixed
- cgroup v1 sentinel value (near `i64::MAX`) is now correctly detected as "unlimited"

## [0.13.0] - 2026-04-10

### Changed
- Dependency refresh batch (Dependabot): `tokio` → 1.50, `hf-hub` → 0.5, `safetensors` → 0.7, `uuid` → 1.22; CI actions: `Swatinem/rust-cache` → 2.9.1, `softprops/action-gh-release` → 2.6.1, `docker/setup-buildx-action` → 4.0.0, `docker/login-action` → 4.0.0, `docker/build-push-action` → 7.0.0
- INT8 quantized model variant (`BGE_M3_MODEL=int8`) refinements following 0.11.0 introduction

## [0.12.0] - 2026-03-14

### Changed
- Default model variant changed to **FP16** (`Xenova/bge-m3`, ~1.08 GB/session) for reduced RAM
- `coreml-ep.md` and `model-variants.md` documentation updates

## [0.11.0] - 2026-03-03

### Added
- **INT8 quantized model support** (`BGE_M3_MODEL=int8`, `Xenova/bge-m3`) — ~568 MB/session (74% memory reduction). Dense cosine similarity vs FP32: mean=0.976, p5=0.969, min=0.963
- Replaces `docs/apple-silicon.md` with four focused reference docs: `architecture.md`, `coreml-ep.md`, `model-variants.md`, `performance.md`

### Fixed
- ORT custom build `scripts/build-ort.sh`: corrected PATH/`CMAKE_PREFIX_PATH` handling so Homebrew-provided `protoc` and `protobuf` cannot be discovered during the ONNX Runtime build

## [0.10.0] - 2026-03-03

### Added
- `BGE_M3_MODEL` environment variable for selecting between `fp32` (BAAI/bge-m3) and `fp16` (Xenova/bge-m3) model variants
- Apple Silicon native install scripts (`scripts/install-bge-m3-apple.sh`, `scripts/ai.bge-m3.server.plist`) — builds a release binary and bootstraps a LaunchAgent on port 8089

## [0.9.0] - 2026-03-03

### Changed
- Repository renamed from `bge-m3-axum-fastembed-rs` to `bge-m3-embedding-server`

## [0.6.0] - 2026-02-23

### Added
- Criterion 0.5 benchmark suite (`benches/embeddings.rs`) for pure-compute hot paths: `text_input_deser` (single_string 30 ns, array_16 556 ns) and `dense_request_deser` (single_input 91 ns, array_input/64 1794 ns)
- Hand-crafted OpenAPI 3.1 specification (`openapi.yaml`) covering all three endpoints with full request/response schemas, `TextInput` `oneOf`, and component definitions
- README API Reference section linking to `openapi.yaml`

### Changed
- MSRV raised from 1.75 → 1.88 (`aligned v0.4.3` requires `edition2024`/Rust 1.85; `ort v2.0.0-rc.11` requires Rust 1.88)

## [0.5.0] - 2026-02-23

### Added
- `X-Request-ID` header on every response via `tower-http` `SetRequestIdLayer` + `PropagateRequestIdLayer` (UUID v4)
- Model load timing: dense and sparse model load duration logged as structured `elapsed_ms` field
- SBOM generation (`anchore/sbom-action`, SPDX format) attached to each GitHub Release
- Docker image signing with `sigstore/cosign` (keyless OIDC) in the release workflow
- MSRV CI job: `cargo check` on Rust 1.75 (matching `rust-version` in `Cargo.toml`)
- Code coverage CI job: `cargo llvm-cov` → Codecov upload (`fail_ci_if_error: false`)
- Beta channel test matrix: `test` job now runs on both `stable` and `beta` (`continue-on-error` for beta)
- Property-based tests for `TextInput` deserialization using `proptest` (4 tests, 1024 generated cases)
- Doc comments on `Config`, `AppError`, `AppState`, `validate_input`, `check_ready`
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- GitHub issue templates (`bug_report.yml`, `feature_request.yml`) and PR template

### Changed
- `.unwrap()` → `.expect("...")` in all test code (main.rs, handler.rs)
- Test count: 53 → 59 unit tests

## [0.4.0] - 2026-02-23

### Added
- `BGE_M3_LOG_FORMAT=json` environment variable for structured JSON log output
- `EmbedPool::with_fixed_responses()` test fixture for happy-path handler tests
- Happy-path dense and sparse embedding handler tests (53 tests total, up from 51)
- `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md` community health files
- README status badges (CI, Release, License, Docker)
- `dependabot.yml` for weekly automated dependency updates (github-actions + cargo)

### Changed
- Dockerfile: non-root `USER bge`, `curl` removed from runtime, HEALTHCHECK uses bash TCP check
- Dockerfile: base image `ubuntu:24.04` pinned to SHA256 digest
- GitHub Actions: all action refs SHA-pinned for supply chain security

## [0.3.0] - 2026-02-23

### Added
- Worker lifecycle tracking with `RAII WorkerGuard` — live worker count exposed via `EmbedPool::live_worker_count()`
- Health endpoint reflects degraded state (`"warn"`) when some workers have exited
- Router-level integration tests via extracted `build_router()` function
- `run_readiness_probe()` extracted for unit testability
- `AppState::total_workers` field for health endpoint reporting
- `[lints.clippy] pedantic` enforced with all warnings treated as errors
- `[profile.release]` hardening: `lto = "thin"`, `codegen-units = 1`, `strip = "symbols"`, `panic = "abort"`
- `rustfmt.toml` for consistent formatting
- `deny.toml` for supply chain security (`cargo deny check`)
- CI workflow hardening: job-level permissions, concurrency cancellation, `--no-tests=warn`

### Changed
- Test count increased from 34 to 51 unit tests

## [0.2.1] - 2026-02-22

### Fixed
- CI workflow now uses `--no-tests=warn` instead of `--no-tests=pass`

## [0.2.0] - 2026-02-22

### Added
- Robust embedding service with worker pool pattern
- Dense embeddings endpoint (`POST /v1/embeddings`) — OpenAI-compatible
- Sparse embeddings endpoint (`POST /v1/sparse-embeddings`) — BGE-M3 SPLADE-style
- Health endpoint (`GET /health`) with readiness probe
- Multi-arch Docker images (linux/amd64, linux/arm64) via GHCR
- Automated release workflow via GitHub Actions

[Unreleased]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.9.1...v0.10.0
[0.9.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.6.0...v0.9.0
[0.6.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/releases/tag/v0.2.0
