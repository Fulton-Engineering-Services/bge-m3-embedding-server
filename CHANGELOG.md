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
- `BGE_M3_GPU_COUNT` environment variable — number of GPU devices on the instance.
  Auto-detected on Linux from `/proc/driver/nvidia/gpus/`; defaults to `1` on macOS and on
  Linux without an NVIDIA driver. Used to clamp `BGE_M3_WORKERS` and to pin each worker to a
  distinct CUDA device (`device_id = worker_index % gpu_count`).
- `BGE_M3_TRT_WARMUP_SHAPES` environment variable — pre-compiles TensorRT engine files for a
  configurable set of `(batch, seq)` shapes during worker startup. Default is a 24-shape 2D
  grid `{1, 2, 4, 8, 16, 32} × {128, 512, 2048, 8192}` in batch-major order. Multi-worker runs
  automatically stride-partition the grid so each GPU compiles a disjoint subset in parallel,
  reducing cold-compile wall-clock time roughly proportionally to GPU count.
- `BGE_M3_WARMUP_ONLY` environment variable — when `1`, the server initialises normally,
  compiles + fsyncs all configured TRT engine files, then exits 0. No HTTP listener is bound.
  Designed as an ECS init container that pre-populates the shared TRT engine cache before the
  main container starts, so `/health` reaches `ok` in seconds rather than 90–180 minutes on a
  cold cache. Heartbeat logging (including the per-device GPU heartbeat) still fires during
  warmup so operators can monitor VRAM and temperature in CloudWatch while engines compile.
- TensorRT engine-cache durability — every regular file under `{BGE_M3_CACHE_DIR}/trt-engines/`
  plus the directory itself is fsynced after each successful warmup compile, so an ECS
  `OutOfMemoryError` SIGKILL (`exitCode 137`) cannot strand a half-written engine plan in the
  kernel page cache. Cache state is logged at every container start as `trt cache: found N
  cached engines at <path>` (warm) or `trt cache: empty (will compile)` (cold).
- TensorRT timing cache at `{BGE_M3_CACHE_DIR}/trt-timing` — persists per-tactic kernel timings
  so the TRT builder can skip tactic-selection on subsequent engine builds. Enabled
  unconditionally alongside the engine cache.
- TRT warm-cache fast path — `trt_prewarm` runs at most 4 dimensional-extreme `(batch, seq)`
  shapes (one per `min_batch`, `max_batch`, `min_seq`, `max_seq`); if all four are under
  `CACHE_HIT_THRESHOLD_MS = 5000 ms` the remaining warmup shapes are skipped as `fully_cached:
  true`, eliminating ~24–72 s of redundant cache-hit loads on every warm restart. Zero false
  positives: an in-range hit on all four extremes mathematically guarantees every shard shape is
  within the cached profile range.
- NVML-backed GPU heartbeat (GPU builds only) — one additional `INFO` log event per CUDA device
  per heartbeat tick, with `gpu_device`, `vram_used_mb`, `vram_total_mb`,
  `vram_utilization_pct`, `gpu_utilization_pct`, `gpu_temp_c`, and `gpu_temp_f`. Driver/library
  unavailability is logged once at `WARN` and silently skipped — never fatal. CPU builds compile
  the GPU stats module as a zero-cost stub with no `nvml-wrapper` dependency.
- Build-variant log tagging — every JSON log line now starts with `"bge_module":"server"` and
  `"build":"cuda"` (when the `cuda`/`tensorrt` features are enabled, e.g. via `Dockerfile.cuda`)
  or `"build":"cpu"` (default `Dockerfile`, MLAS EP). Use `filter build = "cuda"` in CloudWatch
  Logs Insights to slice a mixed CPU/CUDA fleet.
- `BGE_M3_HEARTBEAT_SECS` environment variable — periodic heartbeat log interval (default
  `60`s; `0` disables). Each tick logs RSS, live/loaded workers, queue depth, available
  permits, and probe status.
- X-* HTTP header passthrough — embedding handlers extract a configurable allowlist of `X-*`
  request headers (e.g. `X-Request-ID`, `X-Trace-ID`) and propagate them onto the
  `embedding request complete` log event, so router-level correlation IDs survive the
  upstream/CPU/GPU split.
- Client-disconnect / abandoned-request logging — workers perform two best-effort
  `oneshot::Sender::is_closed()` checks: pre-dispatch (`request abandoned by client before
  dispatch`) skips inference for requests that were cancelled while queued, and
  post-completion (`request abandoned by client during inference`) logs `inference_ms_so_far`
  + `chunks` so operators can size the router's hedge budget against actual GPU wall time.
- `Dockerfile.cuda` — multi-stage CUDA + TensorRT image based on
  `nvidia/cuda:12.6.0-cudnn-*-ubuntu24.04` (linux/amd64 only). Downloads the **upstream
  Microsoft ORT GPU prebuilt** (`onnxruntime-linux-x64-gpu-{VERSION}.tgz` from the official
  `microsoft/onnxruntime` GitHub release), verifies SHA-256, and links ORT **dynamically** via
  `ORT_PREFER_DYNAMIC_LINK=1` so `dladdr` resolves the runtime path from
  `libonnxruntime.so.1.24.2` (not `argv[0]`). All provider `.so` files
  (`libonnxruntime_providers_shared.so`, `libonnxruntime_providers_tensorrt.so`,
  `libonnxruntime_providers_cuda.so`) plus `libonnxruntime.so.1.24.2` are installed to
  `/usr/local/bin/` next to the binary; the `libonnxruntime.so.1` SONAME symlink is recreated
  with `RUN ln -s` (Docker COPY dereferences symlinks). The runtime stage installs
  `libnvinfer10`, `libnvinfer-plugin10`, `libnvonnxparsers10` from NVIDIA's CUDA APT repo for
  the standard `TensorrtExecutionProvider`. **Production builds do NOT use the `download-ort`
  Cargo feature** — that feature is reserved for CI environments where `ORT_LIB_LOCATION` is
  unavailable.
- Release workflow extended: publishes `<version>-cuda` / `latest-cuda` GHCR tags alongside the
  CPU (`latest`) multi-arch image.

### Changed
- **`BGE_M3_WORKERS` is automatically clamped to `BGE_M3_GPU_COUNT`** (not `1`) when
  `BGE_M3_EP` is `cuda` or `tensorrt`, with a `WARN` log line if the requested count exceeds
  it. Each worker is pinned to a distinct CUDA device (`device_id = worker_index % gpu_count`).
  On a single-GPU instance the default `BGE_M3_GPU_COUNT=1` preserves the previous behavior;
  on a multi-GPU instance, set `BGE_M3_WORKERS = BGE_M3_GPU_COUNT` for maximum parallel
  inference throughput. Multi-stream concurrency on a single GPU remains a future enhancement.
- `ort/tracing` Cargo feature is enabled in this crate so ORT's internal `info!`/`warn!`/
  `error!` calls — including `apply_execution_providers`'s "Successfully registered" /
  "Couldn't register" / "An error occurred when attempting to register" lines and the C-side
  `tracing_logger` callback — are forwarded into `tracing` and visible under `target=ort` in
  CloudWatch.
- `error_on_failure()` is set on the `tensorrt` and `cuda` execution-provider dispatches in
  `src/embedder/session.rs::execution_providers`, so a failed EP registration trips a clear
  startup error instead of silently falling back to MLAS/CPU. Net effect: missing provider
  `.so` files, missing TensorRT runtime libs, and CUDA driver problems are now hard worker-load
  failures, not silent CPU fallbacks.

### Fixed
- **Fail-fast model cache directory validation** — `download_model_files` now calls
  `std::fs::create_dir_all(cache_dir)` *before* constructing any `hf_hub::ApiBuilder`. A
  structurally invalid cache path (a path component that's a regular file or non-directory
  device, a read-only parent, a missing EFS access-point mount) now returns
  `"Cannot create or access model cache directory <path>: <io::Error>"` immediately, instead
  of stalling indefinitely on `hf-hub`'s mid-download metadata HTTP call (which has no default
  ureq connect timeout). The previous behaviour caused `EmbedPool::spawn` to park forever
  waiting for a ready signal on misconfigured cache paths combined with unreliable IPv6
  connectivity (notably GitHub Actions). Companion `EmbedPool::spawn` change: a new
  `await_worker_signal` helper resolves on either the worker's readiness mpsc message or its
  `JoinHandle` exit (biased toward the explicit ready signal), so a worker that panics or
  exits before dropping its `ready_tx` clone now surfaces an explicit "exited before signaling
  readiness" / "panicked before signaling ready" error instead of stalling.
- **TensorRT prewarm postcondition uses `engine_count_after > 0`, not
  `engine_count_delta > 0`** — ORT's TRT EP stores one profile-based `.engine` file per fused
  subgraph and rewrites that single file in place when a new shape expands the cached
  `[min, max]` profile, so `engine_count_delta == 0` is the normal steady-state for every
  shape after the first cold compile. The prior `engine_count_delta <= 0` ERROR rule produced
  false-positive errors on every shape after the first; the postcondition now correctly
  treats `engine_count_after == 0` (no engine file at all) as the persistence failure in both
  `prewarm_persistence_postcondition_failed` and the per-shape WARN in `runner.rs`.

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
