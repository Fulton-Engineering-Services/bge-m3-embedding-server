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
| `GET` | `/v1/models` | Fleet discovery |

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | ONNX model cache path |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Worker threads (each with its own ORT session). For GPU EPs, set equal to `BGE_M3_GPU_COUNT`. |
| `BGE_M3_INTRA_THREADS` | `1` | ORT intra-op threads per worker. Raise to `floor(num_cpus / workers)` on under-utilized hosts. |
| `BGE_M3_MAX_BATCH` | `256` | Max texts per request |
| `BGE_M3_MAX_SEQ_LENGTH` | `8192` | Max tokenized sequence length `[1, 8192]` |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Seconds before models are unloaded; `0` disables |
| `BGE_M3_MODEL` | `fp16` | Model variant: `fp32` (BAAI, ~2.16 GB/session), `fp16` (Xenova, ~1.08 GB), `int8` (Xenova quantized, ~568 MB) |
| `BGE_M3_EP` | `cpu` | Execution provider: `cpu`, `cuda`, or `tensorrt`. CoreML is always used on macOS. |
| `BGE_M3_GPU_COUNT` | auto | GPU device count. Auto-detected from `/proc/driver/nvidia/gpus/`; set explicitly on multi-GPU ECS tasks. |
| `BGE_M3_GPU_VRAM_BUDGET_BYTES` | 10 GiB | VRAM workspace ceiling for GPU EPs. |
| `BGE_M3_TRT_WARMUP_SHAPES` | 16-shape grid | Comma-separated `BxL` shapes for TRT engine pre-compilation. Shrink to `1x128` on workstations. |
| `BGE_M3_WARMUP_ONLY` | `0` | Exit after TRT engine compilation — use as an ECS init container to pre-populate the engine cache. |

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
- **`ort/tracing` is required** — without it, ORT EP registration failures are completely silent (this masked the 2026-05 outage). `Cargo.toml` enables it explicitly; check `target=ort` events in CloudWatch when a TRT regression is suspected.
- **`error_on_failure()` on GPU EPs** — both `cuda` and `tensorrt` dispatch use `.error_on_failure()` so missing provider `.so` files or CUDA driver problems cause a hard startup failure rather than a silent CPU fallback.
- **Dockerfile.cuda uses the Microsoft ORT tarball** — not pyke.io. ORT is dynamically linked (`ORT_PREFER_DYNAMIC_LINK=1`); provider `.so` files live in `/usr/local/bin/`. To upgrade ORT: bump `ARG ORT_VERSION` and `ARG ORT_MS_SHA256` in `Dockerfile.cuda`, update the `ln -s` version string, and bump `Cargo.toml` `ort = "=X.Y.Z"`.
- **ECS capacity at `max_seq=8192`** — per-worker high-water ~10.3 GB (fp16). Formula: `workers × 10.3 GB + OS_HEADROOM ≤ task_memoryMiB`.
