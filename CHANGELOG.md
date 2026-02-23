# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/releases/tag/v0.2.0
