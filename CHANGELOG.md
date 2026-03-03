# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.6.0...v0.9.0
[0.6.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/releases/tag/v0.2.0
