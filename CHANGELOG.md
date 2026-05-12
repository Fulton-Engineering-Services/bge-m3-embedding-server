# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CUDA and TensorRT execution providers** — opt-in via `BGE_M3_EP=cuda` or `BGE_M3_EP=tensorrt`
  using new `cuda` / `tensorrt` Cargo features (`ort/cuda`, `ort/tensorrt`). Activates NVIDIA
  GPU inference on Linux. On macOS, CoreML is always used and `BGE_M3_EP` is ignored.
- `BGE_M3_GPU_VRAM_BUDGET_BYTES` environment variable — VRAM workspace ceiling (default 10 GiB)
  when a GPU EP is active. The host-RAM startup probe is bypassed when `BGE_M3_EP` is `cuda` or
  `tensorrt`.
- `Dockerfile.cuda` — multi-stage CUDA image based on `nvidia/cuda:12.6.0-cudnn-*-ubuntu24.04`
  (linux/amd64 only). Uses the `download-ort` feature to fetch a CUDA+TRT-enabled ORT binary at
  build time.
- Release workflow extended: publishes `<version>-cuda` / `latest-cuda` GHCR tags alongside the
  CPU (`latest`) multi-arch image.

### Changed
- `BGE_M3_WORKERS` is automatically clamped to `1` with a `WARN` log line when `BGE_M3_EP` is
  `cuda` or `tensorrt`. The GPU is a serial inference resource; multiple sessions on the same GPU
  waste VRAM without throughput benefit.

## [0.15.0] - 2026-05-10

### Added
- **Unified dense + sparse embeddings endpoint** (`POST /v1/embeddings:both`) — runs **one** ONNX `session.run()` per chunk and projects both dense (CLS-pooled, L2-normalized) and sparse (SPLADE-style) outputs from the same forward pass. Numerically equivalent to calling `/v1/embeddings` and `/v1/sparse-embeddings` separately on the same inputs, within FP rounding tolerance, but at near-zero marginal GPU cost. Existing `/v1/embeddings` and `/v1/sparse-embeddings` endpoints are unchanged.
- `EmbedPool::both(texts)` Rust API, `Both` job variant in the worker pool, and `embed_both` shared-pass implementation.
- `DualRequest` / `DualResponse` / `DualEmbeddingData` request/response types.
- Handler-level validation tests, router-level tests, and an opt-in (`BGE_M3_EQUIVALENCE_TEST=1`) integration test verifying that the dual-pass path matches separate dense + sparse passes within FP tolerance.
- OpenAPI documentation for the new endpoint plus `DualRequest`/`DualResponse`/`DualEmbeddingData` schemas.

### Fixed
- **Sequential worker spawning** — `EmbedPool::spawn` now loads workers sequentially (leader first, then followers one at a time) so each worker's per-model RSS delta is measured in isolation. Parallel spawning caused `/proc/self/statm` to capture cumulative multi-worker allocation on the first RSS read, producing inflated `per_worker_workspace` values that broke the probe cost-model fit.
- **Median aggregation of per-worker RSS deltas** — `EmbedPool` now stores the median (not the max) of per-worker RSS deltas. Median is robust to a single outlier from page-cache settling or ORT arena jitter; `fetch_max` would return an inflated outlier and re-introduce the same measurement contamination on a different code path.
- **ECS Managed Instances cgroup-v2 path-walk** — `detect_available_memory()` now reads `/proc/self/cgroup`, extracts the unified-hierarchy cgroup path, and walks ancestors until it finds a numeric limit. On Bottlerocket without `--cgroupns=private`, `/sys/fs/cgroup/memory.max` resolves to the host root where the value is `"max"`; the path-walk reaches the task's actual cgroup limit without falling back to host RAM.
- **Probe RSS-cap absolute guard** — `run_probe` reads `current_rss` before each shape and skips it if `current_rss + 4 × chunk_cost(batch, seq) > 87.5% × cgroup_limit`, preventing the ORT session arena from accumulating past the container ceiling mid-sweep.
- **Arena warm-up before probe sweep** — `run_probe` now runs a `(1, 64)` `session.run()` and discards the result before the measurement sweep begins. ORT lazily initialises its memory arena on the first call, contributing a ~1 GiB constant offset to `rss_after - rss_before`. The warm-up flushes this out of per-shape deltas so the OLS fitter receives meaningful signal.
- **`OwnedSemaphorePermit` for probe serialisation** — `spawn_probe_task` uses `Arc<Semaphore>::acquire_many_owned` and moves the resulting `OwnedSemaphorePermit` into the spawned task closure. Using `acquire_many` and binding the permit to a local variable in the parent function caused the permit to be dropped immediately after `tokio::spawn` returned (synchronously), before the spawned task started — leaving the semaphore un-drained and allowing real traffic to contaminate per-shape RSS measurements.
- **Restored 7-shape probe set** — `PROBE_SHAPES` contains 6 static shapes plus a dynamic `(1, max_seq)` shape added at runtime. The three OOM-protection layers (arena warm-up, conservative `fits()` gate, absolute-RSS guard) make the full shape set safe to sweep without risking mid-probe container kills.

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

[Unreleased]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/compare/v0.14.0...v0.15.0
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
