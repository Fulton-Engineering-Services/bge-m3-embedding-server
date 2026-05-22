# CLAUDE.md — bge-m3-embedding-server

Axum HTTP server serving BGE-M3 dense and sparse embeddings via ONNX Runtime.

## Build & Test

```bash
cargo build
cargo nextest run --all-features --no-tests=warn
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo deny check          # supply chain audit
hawkeye check             # license headers (.rs files only)

# Equivalence tests (requires model download)
BGE_M3_EQUIVALENCE_TEST=1 BGE_M3_CACHE_DIR=/tmp/bge-m3-cache \
  cargo test --test equivalence -- --ignored --nocapture
```

**Always run `cargo fmt --all` before pushing** — CI fails the format check even when all tests pass.

**`--features tls` requires `cmake` and a C toolchain** at build time because `aws-lc-sys` (pulled in by `axum-server/tls-rustls`) compiles a bundled AWS-LC C library. On Ubuntu: `sudo apt-get install -y cmake`. On macOS: `brew install cmake` or Xcode CLT (already ships `cmake`).

## Run Locally

```bash
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```

Models are downloaded to `BGE_M3_CACHE_DIR` on first run; subsequent runs reuse the cache.
Add `BGE_M3_DISABLE_AUTO_BUDGET=1` to skip the 2-minute Linux startup probe during development.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/embeddings` | Dense embeddings (OpenAI-compatible) |
| `POST` | `/v1/sparse-embeddings` | Sparse embeddings (SPLADE-style) |
| `POST` | `/v1/embeddings:both` | Dense + sparse in one forward pass (preferred over two calls) |
| `GET` | `/health` | `200 ok/warn/idle` when ready; `503 loading/fail` otherwise |
| `GET` | `/health/deep` | Runs a canary embed (batch=1, seq≈8); `200 ok` or `503 fail/loading`. Exercises the actual TRT path — use this for ECS and ALB health checks |
| `GET` | `/v1/models` | Fleet discovery |

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | ONNX model cache path |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_TLS_CERT_PATH` | unset | Path to TLS certificate PEM file. When set together with `BGE_M3_TLS_KEY_PATH` and the server is built with `--features tls`, the server binds HTTPS instead of HTTP. Both must be set or both must be absent — setting only one is a startup error. |
| `BGE_M3_TLS_KEY_PATH` | unset | Path to TLS private key PEM file. Must be set together with `BGE_M3_TLS_CERT_PATH`; see above. |
| `BGE_M3_WORKERS` | `2` | Worker threads (each with its own ORT session). For GPU EPs, set equal to `BGE_M3_GPU_COUNT`. |
| `BGE_M3_INTRA_THREADS` | `1` | ORT intra-op threads per worker. Raise to `floor(num_cpus / workers)` on under-utilized hosts. |
| `BGE_M3_MAX_BATCH` | `256` | Max texts per request |
| `BGE_M3_MAX_BODY_BYTES` | `33554432` | Maximum HTTP request body size in bytes. Raise for large embedding batches with long function bodies. |
| `BGE_M3_MAX_SEQ_LENGTH` | `8192` | Max tokenized sequence length `[1, 8192]` |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Seconds before models are unloaded; `0` disables |
| `BGE_M3_MODEL` | `fp16` | Model variant: `fp32` (BAAI, ~2.16 GB/session), `fp16` (Xenova, ~1.08 GB), `int8` (Xenova quantized, ~568 MB) |
| `BGE_M3_EP` | `cpu` | Execution provider: `cpu`, `cuda`, or `tensorrt`. CoreML is always used on macOS. |
| `BGE_M3_GPU_COUNT` | auto | GPU device count. Auto-detected from `/proc/driver/nvidia/gpus/`; set explicitly on multi-GPU ECS tasks. |
| `BGE_M3_GPU_VRAM_BUDGET_BYTES` | 10 GiB | VRAM workspace ceiling for GPU EPs. |
| `BGE_M3_TRT_MAX_WORKSPACE_BYTES` | unset (TRT default) | TRT EP kernel autotuner workspace cap (bytes). Set to `4294967296` (4 GiB) on L40S/H100 with 4 workers to prevent JIT allocation failures when VRAM is 87%+ saturated. |
| `BGE_M3_GPU_MEM_LIMIT_BYTES` | unset (all available) | CUDA EP device memory limit (bytes). Symmetric cap for the CUDA execution provider. |
| `BGE_M3_ADAPTIVE_WARMUP_ENABLED` | `0` | `1` to enable the in-process background JIT warmup loop. Detects unseen `(batch, seq)` shapes during inference and pre-compiles them when the GPU is idle. |
| `BGE_M3_ADAPTIVE_WARMUP_QUIET_SECS` | `3` | Seconds of full idle (queue empty, all workers free) required before the adaptive loop fires a warmup compile. `0` returns immediately (no sleep before first idle check). |
| `BGE_M3_ADAPTIVE_WARMUP_MAX_SHAPES_PER_HOUR` | `12` | Per-process cap on adaptive warmup compiles per hour. Prevents pathological traffic from compiling indefinitely. |
| `BGE_M3_ENGINE_PROPAGATION_ENABLED` | matches `adaptive_warmup_enabled` | Broadcast `(batch, seq)` shape notifications to peer workers after a new TRT engine plan is written to EFS. Peers run `trt_prewarm` for a ~1-3s fast disk-load instead of full JIT. Disable for debugging (keeps adaptive warmup active). |
| `BGE_M3_TRT_WARMUP_SHAPES` | 16-shape grid | Comma-separated `BxL` shapes for TRT engine pre-compilation. Shrink to `1x128` on workstations. |
| `BGE_M3_WARMUP_ONLY` | `0` | Exit after TRT engine compilation — use as an ECS init container to pre-populate the engine cache. |

### TRT Operational

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_PREWARM_STRICT` | `true` | When true, prewarm postcondition failure (`engine_count_after == 0` after fresh TRT compiles) causes the worker to refuse readiness and exit instead of logging WARN and continuing. Set to `0` to restore pre-PR-77 behavior. See Key Gotchas. |
| `BGE_M3_CIRCUIT_BREAKER_THRESHOLD` | `5` | Number of consecutive inference errors that trips the per-worker circuit breaker. When tripped, the worker drops its ORT session (clearing the CUDA arena) and decrements `loaded_workers`. `/health` transitions to `idle` (200) when all workers unload; `fail` (503) when all workers exit. Worker reloads on next request. Set to a large value to effectively disable. |
| `BGE_M3_TRT_CACHE_GC_ENABLED` | unset (disabled) | Requires `cache-gc` Cargo feature compiled in. When set to `1`, the leader worker (id=0) deletes all `_smXX.engine` plans (and aligned sidecars) whose SM suffix does not match the current host's SM at startup. **WARNING:** Never enable against a shared EFS cache in a mixed-SM ASG — it will delete plans needed by other instance types. Drain old-SM ASG first, run GC, then bring up new-SM ASG. |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_LOG_FORMAT` | auto | `json` (non-TTY default), `text`, or `pretty`. CloudWatch requires `json`. |
| `BGE_M3_HEARTBEAT_SECS` | `60` | Heartbeat log interval (RSS, workers, queue depth, probe status). `0` disables. |
| `RUST_LOG` | `info` | Tracing filter (e.g. `bge_m3_embedding_server::binpack=trace`). |

Every JSON log line begins with `"bge_module":"server"` and `"build":"cpu"` or `"build":"cuda"` for fleet filtering.

### Auto-Budget (Linux only)

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_DISABLE_AUTO_BUDGET` | unset | `1` skips the startup probe and uses conservative defaults. |
| `BGE_M3_MEMORY_SAFETY_FACTOR` | `0.7` | Fraction `[0.1, 1.0]` of detected workspace; 30% headroom for ORT fragmentation. |
| `BGE_M3_COST_MODEL_A` / `_B` | probe-derived | Override OLS-fitted linear/quadratic workspace coefficients. |
| `BGE_M3_AVAILABLE_MEMORY_BYTES` | detected | Override memory detection for testing or unusual runtimes. |

**Deprecated:** `BGE_M3_ONNX_BATCH_SIZE` — logs a WARN and maps to `BGE_M3_TOKEN_BUDGET`.

## Model Variants

- **FP16 (default):** Fleet default on Linux. On macOS CoreML, Cast nodes fragment the GPU subgraph — **use `fp32` on Apple Silicon** (6–10× faster).
- **FP32:** Required for macOS CoreML. Also required when Xenova exports lack 8192-position embeddings.
- **INT8:** ~74% memory reduction vs FP32; cosine similarity vs FP32 mean=0.976. Use MLAS only — CoreML fragments identically to FP16.

## Architecture

**Worker pool:** each worker is a `spawn_blocking` thread with its own ORT session. Requests dispatched via bounded `mpsc` channel.

- **Tokenize-once, bin-pack:** texts tokenized in one pass, grouped into `session.run()` calls using the quadratic cost model `a·BS + b·BS²` to fit `max_workspace_bytes`.
- **Startup probe (Linux):** sweeps 7 `(batch, seq)` shapes, fits OLS cost-model coefficients, caches to `{cache_dir}/probe-coefficients.json` (keyed by `version × model × max_seq × arch`).
- **Sparse projection:** token hidden states → `sparse_linear.safetensors` (4 KB) → ReLU → max-pool.
- **Cross-worker engine propagation:** After any worker writes a new TRT engine plan (adaptive_warmup or real-inference JIT), a `tokio::sync::broadcast` channel fans out the `(batch, seq)` shape to every peer. Each peer drains the channel between requests and runs `trt_prewarm` (~1-3s fast disk-load from EFS) against its own session. Per-worker `warmed_local: HashSet` ensures idempotency. See the homogeneous-SM constraint (L-5 in `adaptive_warmup.rs` docs): plans are SM-specific, so all GPUs in the pool must share the same compute capability.

Verify tuning after deploy: `curl http://localhost:8081/health | jq '{status,max_seq_length,tuning}'`
Healthy values: `a_bytes_per_token` ≈ 18000–20000 (fp16, amd64); `b_bytes_per_token_sq` ≈ 5–8; `model_rss_bytes_per_worker` ≈ 1.1 GB.

## Source Layout

- **No `mod.rs`** — use the `foo.rs + foo/` layout. Parent module files (`embedder.rs`, `handler.rs`, etc.) are facades: only `mod` declarations and `pub use` re-exports.
- **`main.rs` is 20–40 lines.** All logic lives in `lib.rs` and submodules so it stays testable.
- **File-size target:** 100–400 lines; hard ceiling ~500 production lines. Tests beyond ~150 lines move to `<file>/tests.rs`.
- **Config tests** use the `from_lookup()` closure — not `env::set_var` — to avoid global state under parallel tests.

## Docker

```bash
# CPU (linux/amd64 + linux/arm64)
docker build -t bge-m3-embedding-server .
docker run --rm -p 8081:8081 -v /path/to/cache:/cache bge-m3-embedding-server

# CUDA + TensorRT (linux/amd64 only)
docker build -f Dockerfile.cuda -t bge-m3-embedding-server:cuda .
docker run --rm --gpus all -p 8081:8081 -v /path/to/cache:/cache \
  -e BGE_M3_EP=cuda bge-m3-embedding-server:cuda
```

Port `8081`. Dockerfile `HEALTHCHECK` polls `/health` every 10 s with a 120 s start period.
Releases: `<version>`/`latest` (CPU multi-arch) and `<version>-cuda`/`latest-cuda` (CUDA amd64) on GHCR.

## Releasing

**Do not create git tags locally.** Bump version in `Cargo.toml`, commit, push to `main`. The Release workflow handles tagging, Docker builds, and GitHub Release.

## Key Gotchas

- **Workers load sequentially** — `/proc/self/statm` is process-wide; parallel init contaminates per-worker RSS deltas, breaking the probe OLS fit. Never parallelize `EmbedPool::spawn`.
- **Median, not max, for RSS** — `EmbedPool` stores the median of per-worker deltas to be robust to page-cache settling outliers. Do not revert to `fetch_max`.
- **Stale model cache** — silent worker failures ("Worker exited before signaling readiness"). Fix: clear `BGE_M3_CACHE_DIR`.
- **Xenova FP16/INT8 long-context** — Xenova exports may cap `max_position_embeddings` at 512. ORT errors on the first real request if `BGE_M3_MAX_SEQ_LENGTH` exceeds this. Use `fp32` or lower the seq length.
- **TRT cold-start: use `BGE_M3_WARMUP_ONLY=1`** — the default 16-shape grid takes ~90–180 min on first deploy. Run as an ECS init container to decouple compilation from service startup. Set `healthCheckGracePeriodSeconds ≥ 10800`.
- **TRT plans are compute-capability-specific** — `sm_75` (T4), `sm_86` (A10G), `sm_89` (L4/L40S), `sm_120` (Blackwell). Plans from one SM cannot run on another. Keep ASG instance families homogeneous when using a shared EFS cache. ORT namespaces filenames by `_smXX` so plans for different SMs coexist safely.
- **`ort/tracing` is required** — without it, ORT EP registration failures are completely silent (this masked a prior production outage). `Cargo.toml` enables it explicitly; check `target=ort` events in CloudWatch when a TRT regression is suspected.
- **`error_on_failure()` on GPU EPs** — both `cuda` and `tensorrt` dispatch use `.error_on_failure()` so missing provider `.so` files or CUDA driver problems cause a hard startup failure rather than a silent CPU fallback.
- **Dockerfile.cuda uses the Microsoft ORT tarball** — not pyke.io. ORT is dynamically linked (`ORT_PREFER_DYNAMIC_LINK=1`); provider `.so` files live in `/usr/local/bin/`. To upgrade ORT: bump `ARG ORT_VERSION` and `ARG ORT_MS_SHA256` in `Dockerfile.cuda`, update the `ln -s` version string, and bump `Cargo.toml` `ort = "=X.Y.Z"`.
- **ECS capacity at `max_seq=8192`** — per-worker high-water ~10.3 GB (fp16). Formula: `workers × 10.3 GB + OS_HEADROOM ≤ task_memoryMiB`.
- **TRT warmup grid MUST cover small batches (1 AND 2)** — single-text and two-text requests are the most common router traffic pattern. If `BGE_M3_TRT_WARMUP_SHAPES` omits batches 1 and/or 2, the first such request triggers in-band TRT JIT for an unseen shape; on the `/v1/embeddings:both` route the autotuner can request multi-terabyte tactic scratch allocations on the fused `value/MatMul + LayerNorm` foreign-node, the CUDA allocator fails, and TRT emits `failed to create engine from network`. `BGE_M3_TRT_MAX_WORKSPACE_BYTES` does NOT bound autotuner tactic scratch buffers — this is a TRT EP limitation, not a config issue. The default in-code grid is `{1,2,4,8,16,32} × {128,512,2048,8192}` (24 shapes). When overriding `BGE_M3_TRT_WARMUP_SHAPES`, always include rows for batches 1 AND 2 — the server emits a `trt_warmup_shape_coverage_gap` WARN at startup if either is missing.
- **TRT fatal engine build error → immediate worker exit** — `is_trt_engine_build_fatal` detects "failed to build engine" and "failed to create engine from network" patterns. When either fires during inference, the worker sends the error to the caller and returns `Err` from `run_worker`, causing `WorkerGuard` to decrement `live_workers`. ECS replaces the task once all workers exit. This is a harder signal than the circuit breaker: no retry, no reload — the CUDA driver state is considered unrecoverable.
- **Per-worker circuit breaker** — After N consecutive inference errors (default N=5, `BGE_M3_CIRCUIT_BREAKER_THRESHOLD`), the worker drops its ORT session and decrements `loaded_workers`. `/health` returns `idle` (200) when `loaded_workers == 0`. The worker reloads on the next request. The circuit breaker and TRT-fatal paths are complementary: non-fatal repeated failures trigger a soft model reload; fatal TRT state causes a hard worker exit.
- **`/health/deep` for ECS and ALB** — Point both ECS `healthCheck.command` and the ALB target group at `/health/deep` instead of `/health`. The deep check exercises the actual TRT session.run() path (batch=1, seq≈8) and returns 503 on inference failure, catching the silent-failure mode where `/health` returned 200 OK while all real embedding requests returned 500.
- **`BGE_M3_PREWARM_STRICT` is a breaking behavior change (default `true` since v0.14.0)** — Prior behavior: prewarm postcondition failure (`engine_count_after == 0`) logged WARN and let the worker continue. New default: worker returns `Err`, pool init propagates, process exits with code 1, and ECS retries the container. If your deployment uses a large warmup shape grid and TRT occasionally hits VRAM exhaustion on first boot, strict mode will cause ECS restart loops. Set `BGE_M3_PREWARM_STRICT=0` to restore the previous warn-and-continue behavior while tuning. After deploying, monitor `exitCode=1` in the ECS event log on first boot. **ECS recommendation:** configure a `restartPolicy` with `restartAttempts ≤ 3` and a `restartWindow` to prevent infinite restart loops under sustained VRAM pressure. Note: a future improvement would use exit code 2 for prewarm strict failures (vs code 1 for other crashes) to enable distinct CloudWatch alarm routing.
- **x_headers log field key rename (v0.14.0)** — The `x_headers` JSON log field changed key format from hyphenated (`x-request-id`) to underscored (`x_request_id`). Any CloudWatch Insights queries or SIEM filters referencing `x_headers."x-*"` (hyphenated) must be updated to `x_headers."x_*"` (underscore) after upgrading past v0.14.0. HTTP header forwarding behavior is unaffected; only the emitted log field keys changed.
