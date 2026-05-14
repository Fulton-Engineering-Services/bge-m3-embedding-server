# CLAUDE.md — bge-m3-embedding-server

Axum HTTP server serving BGE-M3 dense and sparse embeddings via direct ONNX Runtime integration.

## Use Cases

- Document indexing and hybrid search via `/v1/embeddings` (dense) and `/v1/sparse-embeddings` (SPLADE-style)
- **Hybrid ingestion pipelines** via `/v1/embeddings:both` — both output heads share one transformer forward pass (preferred over two separate calls)
- Drop-in replacement for hosted embedding APIs when local inference, data residency, or BGE-M3 sparse vectors are required

## Build & Test Commands

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

# CoreML dispatch profiling (per-op GPU/CPU/ANE decisions emitted to stderr at model load)
cargo build --features coreml-profile
```

## Run Locally

```bash
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```

On first run, model files are downloaded to `BGE_M3_CACHE_DIR`. Subsequent runs reuse the cache.
Watch logs for "ready". Set `BGE_M3_DISABLE_AUTO_BUDGET=1` to skip the Linux startup probe.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/embeddings` | Dense embeddings (OpenAI-compatible) |
| `POST` | `/v1/sparse-embeddings` | Sparse embeddings (BGE-M3 SPLADE-style) |
| `POST` | `/v1/embeddings:both` | Dense + sparse in a single forward pass |
| `GET` | `/health` | Readiness probe — `200 OK` when ready, `503` while loading |
| `GET` | `/v1/models` | Fleet discovery — returns `{"object":"list","data":[{"id":"bge-m3","object":"model","type":"bge-m3"}]}` |

### Health States

| Status code | `status` field | Meaning |
|-------------|---------------|---------|
| `503` | `loading` | Models still initializing at startup |
| `503` | `fail` | All worker threads have exited (fatal) |
| `200` | `idle` | Workers alive but models unloaded after idle timeout; will auto-reload on next request |
| `200` | `warn` | At least one worker exited but some remain |
| `200` | `ok` | All workers healthy and models loaded |

When `status=ok`, the `/health` response also includes:

```json
{
  "status": "ok",
  "workers": { "live": 7, "total": 7 },
  "max_seq_length": 8192,
  "tuning": {
    "a_bytes_per_token": 18432.0,
    "b_bytes_per_token_sq": 6.2,
    "max_workspace_bytes": 2500000000,
    "memory_source": "cgroup_v2",
    "available_bytes": 28991029248,
    "model_rss_bytes_per_worker": 1100000000
  }
}
```

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | Path where ONNX model files are cached |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Number of worker threads (each loads its own model instance; min 1). For GPU EPs, set this equal to `BGE_M3_GPU_COUNT` — each worker is pinned to a distinct GPU device for maximum parallel inference throughput. Workers are clamped to `BGE_M3_GPU_COUNT` in `EmbedPool::spawn` if the requested count exceeds it. |
| `BGE_M3_INTRA_THREADS` | `1` | Intra-op threads each ORT session may use per `session.run()` call (min 1). Default `1` keeps per-worker RSS predictable for the workspace probe. Raise to `floor(num_cpus / workers)` on under-utilized hosts (e.g. `4` on an 8 vCPU task with `workers=2`) to take CPU utilization from ~25% to ~100% under load. Re-run the probe after changing. |
| `BGE_M3_MAX_BATCH` | `256` | Maximum number of texts accepted per request (min 1) |
| `BGE_M3_MAX_SEQ_LENGTH` | `8192` | Maximum tokenized sequence length. Range `[1, 8192]`. Lower values reduce memory use; `8192` is the BGE-M3 model's published maximum. |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Seconds of inactivity before models are unloaded; `0` disables idle unloading |
| `BGE_M3_MODEL` | `fp16` | Model variant: `fp32` = `BAAI/bge-m3` (~2.16 GB/session); `fp16` = `Xenova/bge-m3` (~1.08 GB/session); `int8` = `Xenova/bge-m3` quantized (~568 MB/session). See model variant notes below. |
| `BGE_M3_EP` | `cpu` | Execution provider: `cpu` (MLAS, default), `cuda` (NVIDIA CUDA EP), or `tensorrt` (NVIDIA TensorRT EP). On macOS, CoreML is always used. `cuda`/`tensorrt` require the corresponding Cargo feature and a GPU-enabled ORT build; use `Dockerfile.cuda` and the `-cuda` image tag. |
| `BGE_M3_GPU_VRAM_BUDGET_BYTES` | unset | VRAM workspace ceiling (bytes) when `BGE_M3_EP` is `cuda` or `tensorrt`. Defaults to 10 GiB (suitable for A10G / L4 and larger). The host-RAM probe is bypassed when any GPU EP is active. |
| `BGE_M3_GPU_COUNT` | auto | Number of GPU devices on this instance. Auto-detected on Linux from `/proc/driver/nvidia/gpus/` entry count; defaults to `1` on macOS and on Linux without an NVIDIA driver. Workers are clamped to this value for GPU EPs; each worker is pinned to device `worker_index % gpu_count`. Set explicitly on multi-GPU ECS tasks: `BGE_M3_GPU_COUNT=8`. |
| `BGE_M3_TRT_WARMUP_SHAPES` | 16-shape 2D grid (see gotcha) | Comma-separated `BxL` shapes to pre-compile as TensorRT engine files during worker startup. Only used when `BGE_M3_EP=tensorrt`. Invalid tokens are skipped with a WARN; empty or all-invalid falls back to the default set. Each shape takes 30–170 s on first deploy; subsequent starts reuse the cached engines. With multiple workers (`BGE_M3_WORKERS > 1`), the shape list is automatically sharded across workers using stride partition so each GPU compiles a disjoint subset in parallel — total cold-compile time is reduced roughly proportionally to GPU count. Operators running on workstations should shrink the grid (e.g. `1x128`) to keep cold-start tractable. |
| `BGE_M3_WARMUP_ONLY` | `0` | Exit cleanly after TRT engine compilation. When `1`, the server initialises normally (loads ONNX, configures the TRT EP, runs pre-warm shape compilation), then exits 0 after all engines are compiled and fsynced to the cache. No TCP listener is bound. Use as an ECS init container to pre-populate the engine cache before starting the main container. On multi-GPU instances, set `BGE_M3_GPU_COUNT=N` alongside `BGE_M3_WARMUP_ONLY=1` so the init container spawns N workers and shards the warmup grid in parallel, reducing cold-compile time ~N×. A WARN is logged if set with a non-`tensorrt` EP (no-op, still exits 0). |

### Logging and Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_LOG_FORMAT` | auto | Log format. `json` = structured JSON (default in non-TTY / container environments); `text` or `pretty` = human-readable; unset = auto-detect via TTY check. **CloudWatch Logs Insights requires JSON.** |
| `BGE_M3_HEARTBEAT_SECS` | `60` | Interval between periodic heartbeat log events. Each event logs RSS, live/loaded workers, queue depth, available permits, and probe status. Set `0` to disable. |
| `RUST_LOG` | `info` | Standard tracing filter. Examples: `info`, `debug`, `bge_m3_embedding_server=debug`, `bge_m3_embedding_server::binpack=trace`. |

**Compile-time tags:** every JSON log line starts with two attributes in this order:

1. `"bge_module":"server"` — always the literal string `"server"`. Distinguishes this binary from the BGE router and other BGE-family services in shared log groups.
2. `"build"` — `"cuda"` when either the `cuda` or `tensorrt` feature is enabled (both are turned on by `Dockerfile.cuda`); `"cpu"` otherwise (default `Dockerfile`, MLAS EP). Use it in CloudWatch Insights to filter a mixed CPU/CUDA fleet, e.g. `filter build = "cuda"`.

Example: `{"bge_module":"server","build":"cpu","timestamp":"…","level":"INFO",…}`

The human-readable `text`/`pretty` formats do not include these tags.

**Key log events at INFO:** per-request completion (`route`, `batch_size`, `chunks`, `tokenize_ms`, `inference_ms`, `total_ms`, `worker_id`); periodic heartbeat (`rss_mb`, `live_workers`, `queue_depth`, `probe_status`); full startup sequence. `/health` and `/v1/models` are logged at DEBUG to suppress load-balancer noise.

**GPU heartbeat (GPU builds only):** on each heartbeat tick, one additional `INFO` event with `message: "gpu heartbeat"` is emitted per CUDA device, containing `gpu_device` (device index), `vram_used_mb`, `vram_total_mb`, `vram_utilization_pct`, `gpu_utilization_pct`, `gpu_temp_c` (GPU die temperature in °C), and `gpu_temp_f` (same temperature in °F, computed as `gpu_temp_c * 9 / 5 + 32`). NVML unavailability (driver absent, permission denied) is logged as a single `WARN` at startup and then silently skipped — it is never a fatal error. Per-device query failures are logged at `DEBUG` and skipped. CPU builds compile the GPU stats module as a zero-cost stub; no NVML dependency is present at all on CPU builds. The GPU heartbeat also fires during `BGE_M3_WARMUP_ONLY=1` runs on the same `BGE_M3_HEARTBEAT_SECS` interval, giving operators VRAM and temperature visibility in CloudWatch during TRT engine compilation.

**Useful CloudWatch Insights queries:**
```
# p99 request latency by route
fields route, total_ms
| filter ispresent(route) and @message like "embedding request complete"
| stats pct(total_ms, 99) as p99_ms by route
| sort p99_ms desc

# TRT cache state at every container start (warm vs cold)
fields @timestamp, cache_path, engine_count, profile_count, @message
| filter @message like "trt cache:"
| sort @timestamp desc

# p99 per-shape TRT engine compile time (warmup); cache_hit=true means loaded from disk
fields compile_ms, cache_hit, batch, seq
| filter @message like "engine compiled, cached, and fsynced"
| stats pct(compile_ms, 99) as p99_compile_ms, count() as runs by cache_hit

# TRT prewarm warm-cache fast-path: did the shard skip redundant cache-hit loads?
fields @timestamp, worker_id, checked, skipped, total, cache_hit_threshold_ms
| filter @message like "shard fully cached"
| sort @timestamp desc

# Requests the client abandoned (router hedge race, HTTP disconnect)
fields @timestamp, worker_id, route, inference_ms_so_far, chunks
| filter @message like "request abandoned by client"
| sort @timestamp desc

# Slow requests (> 5 s total)
fields @timestamp, route, batch_size, chunks, inference_ms, total_ms
| filter total_ms > 5000
| sort @timestamp desc
```

### Auto-Budget Tuning (Linux only)

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_DISABLE_AUTO_BUDGET` | unset | Set to `1` to skip the startup probe and use conservative defaults. |
| `BGE_M3_AVAILABLE_MEMORY_BYTES` | detected | Override memory detection (useful for testing and non-standard runtimes). |
| `BGE_M3_MEMORY_SAFETY_FACTOR` | `0.7` | Fraction `[0.1, 1.0]` of detected workspace to use; provides 30% headroom for ORT arena fragmentation. |
| `BGE_M3_TOKEN_BUDGET` | unset | Legacy: sets `max_workspace_bytes = token_budget × cost_per_position`. Use instead of `BGE_M3_ONNX_BATCH_SIZE`. |
| `BGE_M3_COST_MODEL_A` | probe-derived | Override linear coefficient `a` (bytes/token-position). Requires `BGE_M3_COST_MODEL_B` and `BGE_M3_AVAILABLE_MEMORY_BYTES`. |
| `BGE_M3_COST_MODEL_B` | probe-derived | Override quadratic coefficient `b` (bytes/token-position²). |

### Deprecated

| Variable | Notes |
|----------|-------|
| `BGE_M3_ONNX_BATCH_SIZE` | **Deprecated.** Replaced by quadratic cost model + auto-budget. If set, a `WARN` is logged and the value is translated to `BGE_M3_TOKEN_BUDGET` for backward compatibility. Will be removed in a future release. |

## Model Variant Notes

**FP16 (default):** Halves per-session memory vs FP32. On macOS with CoreML EP, FP16 is 6–10× slower than FP32 due to Cast nodes fragmenting the GPU subgraph — use `fp32` on Apple Silicon. On Linux (MLAS), FP16 is the fleet default for reduced RAM and consistent embeddings.

**FP32 (BAAI/bge-m3):** Recommended for macOS CoreML deployments for best latency. Also required if Xenova FP16/INT8 exports lack 8192-position positional embeddings — the startup probe will error at `max_seq_length` and log an actionable message.

**INT8 (Xenova/bge-m3):** Weights-only quantization; ~74% memory reduction vs FP32. Dense cosine similarity vs FP32: mean=0.976, p5=0.969, min=0.963. **Use MLAS (CPU EP) only** — DequantizeLinear nodes fragment CoreML execution identically to FP16.

## Architecture

**Worker pool** pattern: each worker is a `spawn_blocking` thread with its own ORT session (dense + sparse outputs). Work dispatched via bounded `mpsc` channel shared as `Arc<Mutex<Receiver>>`.

- **Tokenize-once, bin-pack:** texts tokenized in one pass, then grouped into `session.run()` calls fitting within `max_workspace_bytes` using the quadratic cost model `a·BS + b·BS²`.
- **Startup probe (Linux):** sweeps 7 `(batch, seq)` shapes, fits cost-model coefficients via OLS, caches to `{cache_dir}/probe-coefficients.json` (fingerprinted by `version × model × max_seq × arch`). Falls back to conservative defaults when RSS is unavailable or the fit is singular.
- **Sparse projection:** token hidden states → `sparse_linear.safetensors` (4 KB) → ReLU → max-pool.

## Source Layout Conventions

Standard Rust style for this crate. Follow the same rules when adding new code.

### File-size targets

- Leaf source files: aim for 100–400 lines. Hard ceiling ~500 lines of production code.
- Inline `#[cfg(test)] mod tests { ... }` is fine up to ~150 test lines. Beyond that, move
  the body to a sibling file: keep `#[cfg(test)] mod tests;` in the production file and put
  the test body in `<file>/tests.rs`.

### Module layout

- Use the `foo.rs + foo/` layout (no `mod.rs`). New modules: never create a `mod.rs`.
- Parent module files are facades: `mod` declarations and `pub use` re-exports only. No logic
  in `embedder.rs`, `probe.rs`, `handler.rs`, `bootstrap.rs`, etc. — those exist purely to
  expose their submodules' public surface and forward call sites.
- `main.rs` is a 20–40 line entry point. All real program logic lives in `lib.rs` and
  submodules so it stays unit- and integration-testable.

### When to split a file

Split a file when any of the following are true:
- It exceeds ~500 production lines.
- It contains two or more independent concerns (e.g. tokenization AND a worker loop, or
  cache I/O AND a numerical fitter).
- The same `mod tests` block has grown past ~150 lines.

When splitting, write a short comment in the parent facade naming each submodule so future
readers can navigate without opening every file.

### Reference

Examples to mirror: `tokio/src/runtime/`, `axum/src/routing/`, `hyper/src/proto/`.

## Verifying auto-budget tuning after deploy

```bash
curl http://localhost:8081/health | jq '{status,max_seq_length,tuning}'
```

Key diagnostic values when healthy: `tuning.a_bytes_per_token` ≈ 18000–20000 for fp16 on amd64; `tuning.b_bytes_per_token_sq` ≈ 5–8; `tuning.model_rss_bytes_per_worker` ≈ 1.1 GB. If `tuning` is absent, check whether `BGE_M3_DISABLE_AUTO_BUDGET=1` was set or the model variant lacks positional embeddings at the configured length.

## Docker

```bash
# Build (CPU — linux/amd64 + linux/arm64)
docker build -t bge-m3-embedding-server .

# Build (CUDA + TensorRT — linux/amd64 only)
docker build -f Dockerfile.cuda -t bge-m3-embedding-server:cuda .

# Run (CPU)
docker run --rm \
  -p 8081:8081 \
  -v /path/to/model-cache:/cache \
  bge-m3-embedding-server

# Run (GPU — requires NVIDIA Container Toolkit)
docker run --rm --gpus all \
  -p 8081:8081 \
  -v /path/to/model-cache:/cache \
  -e BGE_M3_EP=cuda \
  bge-m3-embedding-server:cuda
```

Port `8081`. Built-in `HEALTHCHECK` polls `/health` every 10 s with a 120 s start period (allows time for model download, ONNX init, and the startup probe).

The Release workflow publishes both `<version>` / `latest` (CPU, multi-arch) and `<version>-cuda` / `latest-cuda` (CUDA+TRT, amd64 only) tags to GHCR.

## Releasing

The Release workflow creates git tags automatically. **Do not create tags locally.**
To release: bump version in `Cargo.toml`, commit, push to `main`. The workflow handles tag creation, multi-arch Docker builds, and GitHub Release.

## Security Considerations

- **DoS / workspace bounds:** bin-packer + cost model is the admission control; per-chunk workspace is bounded. See `docs/decisions/long-context-security.md`.
- **TLS CA bundles:** `hf-hub` uses `native-tls`. Minimal Docker images may lack CA bundles — include `ca-certificates` or pre-download models.

## Gotchas

- **Workers load sequentially, not in parallel** — `/proc/self/statm` reports process-wide RSS; parallel session init contaminates per-worker deltas, breaking the probe cost-model fit. Never revert `EmbedPool::spawn` to parallel loading.
- **Median, not max, for RSS aggregation** — `EmbedPool` stores the median of per-worker RSS deltas. Median is robust to outliers from page-cache settling. Do not revert to `fetch_max`.
- **Physics-floor fail-fast** — if `per_worker_workspace` falls below `CostModel::conservative(0).chunk_cost(1, max_seq_length)`, the server exits (ECS restarts it). If the task loops, check `model_rss_per_worker_mb` in CloudWatch — if it's 2–8× higher than expected (~1.1 GiB for fp16), measurement contamination is the cause.
- **`probe_status=failed` + `max_workspace_bytes=0`** — means the workspace budget upstream is zero, not that ORT is broken. Check `per_worker_workspace_mb` in startup logs.
- **Stale model cache** — silent worker load failures ("Worker exited before signaling readiness"). Fix: clear `BGE_M3_CACHE_DIR`.
- **Cache-dir fail-fast (structurally invalid `BGE_M3_CACHE_DIR`).** `download_model_files` calls `std::fs::create_dir_all(cache_dir)` *before* constructing any `hf_hub::ApiBuilder`, so a misconfigured path (a path component that's a regular file or non-directory device, a read-only parent, a missing EFS access-point mount) surfaces immediately as `"Cannot create or access model cache directory <path>: <io::Error>"`. Without this check, `hf-hub` defers cache validation until mid-download — after a `metadata()` HTTP round-trip with no default ureq connect timeout — so on a runner with a misconfigured cache **and** unreliable IPv6 connectivity (the GitHub Actions failure mode) the connect to huggingface.co blocked indefinitely and the doomed `mkdir` never got a chance to surface as the actual cause; `EmbedPool::spawn`'s init future then parked forever waiting for a ready signal. The companion `EmbedPool::spawn` change uses `await_worker_signal` (a `tokio::select!` on the readiness mpsc receiver biased over the worker's `JoinHandle`) so a worker that panics or exits *before* dropping its `ready_tx` clone now surfaces an explicit "exited before signaling readiness" / "panicked before signaling ready" error instead of stalling — the init task always makes progress when the worker terminates, regardless of whether the worker had a chance to send its outcome. If you ever see a cold-start hang on `Cannot create or access model cache directory`, the regression is on the hf-hub side (lazy cache validation) and the fix path is to keep the upfront `create_dir_all`, not to debug network connectivity.
- **Config tests use `from_lookup()` closure** — not `env::set_var`, to avoid process-global state mutation under parallel test execution.
- **Always run `cargo fmt --all` before pushing** — CI fails `cargo fmt --all --check` even when all tests pass.
- **CoreML compiled MIL path** — `{BGE_M3_CACHE_DIR}/coreml/{hash}/N_dynamic_mlprogram/model/compiled_model.mlmodelc/model.mil` — useful for dispatch analysis and op tallying.
- **`BGE_M3_MODEL=fp16` CoreML precision** — loads a smaller ONNX (~1.08 GB) but CoreML compiles it to a FP32 graph internally. Memory savings are ORT session weight loading only, not CoreML runtime precision.
- **Xenova FP16/INT8 long-context** — Xenova exports may have been built with `max_position_embeddings=512`. If the model can't run at the configured `BGE_M3_MAX_SEQ_LENGTH`, ORT errors on the first real request. Use `BGE_M3_MODEL=fp32` or lower `BGE_M3_MAX_SEQ_LENGTH`.
- **Local Docker / macOS:** probe uses `/proc` + cgroups (Linux-only); native macOS LaunchAgent uses conservative defaults (no RSS measurement). In Docker on Apple Silicon, MLAS-only inference makes the probe sweep take several minutes — use `BGE_M3_DISABLE_AUTO_BUDGET=1` for dev; use the native LaunchAgent for production-realistic tuning.
- **Dockerfile builder is `ubuntu:24.04`** — the prebuilt ORT binary requires glibc ≥ 2.38. Debian Bookworm (glibc 2.36) fails with `undefined symbol: __isoc23_strtoul`. Rust installed via `rustup-init` with SHA-256 verification — never `curl | sh`.
- **ECS capacity at `max_seq=8192`** — per-worker high-water ~10.3 GB (fp16). Formula: `cfg_workers × 10.3 GB + OS_HEADROOM ≤ task_memoryMiB`. Workers=2 safe cap on 28 GiB; workers=4 fits at 56 GiB.
- **GPU EPs (cuda/tensorrt): BGE_M3_WORKERS is clamped to BGE_M3_GPU_COUNT** — each worker is pinned to a distinct CUDA device (`device_id = worker_index % gpu_count`). For best throughput on multi-GPU instances, set `BGE_M3_WORKERS = BGE_M3_GPU_COUNT`. On a single-GPU instance the default `BGE_M3_GPU_COUNT=1` preserves the previous single-worker behavior. With TRT EP, warmup shapes are automatically sharded across workers so each GPU compiles its own subset in parallel; the shared EFS cache makes all compiled engines available to all workers after warmup completes.
- **TensorRT cold-start compile (first deploy only):** The first `docker run` with `BGE_M3_EP=tensorrt` compiles engine files into `{BGE_M3_CACHE_DIR}/trt-engines/` for each warmup shape (30–170 s each). The default grid is a 2D `{1, 4, 16, 32} × {128, 512, 2048, 8192} = 16 shapes` composed in batch-major order; cold-cache total compile budget is **roughly 90–180 min on an NVIDIA L4** on first deploy, and the worker signals ready only after all shapes finish so `/health` returns `503` for the full window. Subsequent starts on the same instance reuse the cached engines and warm up in seconds. **Use `BGE_M3_WARMUP_ONLY=1` as an ECS init container to decouple compilation from service startup** — see the README ECS Init Container Pattern section. Operators on workstations should shrink `BGE_M3_TRT_WARMUP_SHAPES` (e.g. `1x128`) to keep iteration fast.
- **Per-warmup-shape skip by `.engine` filename is unsafe (do not implement).** ORT's TRT EP names engine files after the fused-subgraph identity and precision (`TensorrtExecutionProvider_TRTKernel_…_smXX.engine`), not after `(batch, seq)` — matching `.engine` basenames cannot prove a specific shape is compiled. On each prewarm start the server logs `trt prewarm: cache engine basenames …` for operator correlation.
- **Warm-cache fast path (implemented, zero false positives).** ORT's TRT EP stores engine profiles as per-dimension `[min, max]` ranges, **not** per-shape. A `session.run()` is a cache hit when every input dimension is within `[profile.min, profile.max]`; it is a cache miss (30–170 s compile) only when a dimension falls outside that range. After a full first-deploy warmup, the cached profile spans `batch ∈ [1, 32]` and `seq ∈ [128, 8192]` (for the default 16-shape grid), so every subsequent `session.run()` for a shard shape is a cache hit. Rather than pay the 16 × 1–3 s = 16–48 s of redundant cache-hit loads, `trt_prewarm` runs at most **4 dimensional-extreme shapes** (one per `min_batch`, `max_batch`, `min_seq`, `max_seq`), measures wall-clock time, and — if all are under `CACHE_HIT_THRESHOLD_MS = 5 000 ms` — skips the remaining shapes with `fully_cached: true`. The skip has zero false positives: each extreme independently bounds one dimension, so a cache hit on all four corners mathematically guarantees every shard shape is within the cached profile range. If any extreme is slow, the fast path is suppressed and all remaining shapes are compiled normally. Source: `trt_warmup.rs::coverage_check_shapes` + `trt_warmup.rs::trt_prewarm`; proof in module-level doc-comment and tests.
- **Multi-GPU warmup sharding reduces cold-start wall-clock ~N×:** With `BGE_M3_WORKERS=N` and `BGE_M3_GPU_COUNT=N`, the 16-shape grid is stride-partitioned across N workers so each GPU compiles only `ceil(16/N)` shapes concurrently. Example: 4 GPUs → each compiles 4 shapes in parallel → total cold-start drops from ~90–180 min to ~22–45 min. Sharding is automatic when workers > 1; no extra config beyond setting `BGE_M3_WORKERS` and `BGE_M3_GPU_COUNT`. Use this in the `BGE_M3_WARMUP_ONLY=1` init container for multi-GPU tasks.
- **TensorRT engine cache is per-EC2-host, not cross-host.** TRT plan files embed `(GPU compute capability, CUDA version, TRT version, ONNX model SHA, builder config)`. Within a homogeneous ASG (same instance family + same AMI) these are stable and the cache survives container restarts on the same host. ASGs that mix instance families (T4 → A10G compute capability `sm_75` vs `sm_86`) will see expected cache misses when a task lands on a different GPU. EFS-mounted caches help across hosts only when the ASG is homogeneous.
- **TensorRT engine cache durability requires fsync.** ORT/TRT writes engine plan files via normal `write(2)`s, which sit in the kernel page cache until the writeback timer (default 30 s) fires. ECS `OutOfMemoryError` SIGKILL (`exitCode 137`) is immediate — no signal handlers, no flush — so a freshly-compiled engine can be lost even after the inode is listed on EFS. The server now fsyncs every regular file in `{BGE_M3_CACHE_DIR}/trt-engines/` plus the directory itself after each successful warmup compile. If you see "two consecutive cold starts produced identical 172 s recompile times" without the new `trt cache: found N cached engines` log line, suspect (a) the operator is on a pre-fix build, (b) the EFS mount disappeared between restarts, or (c) the ECS task definition is not actually mounting `/cache` with `persistent=true` semantics.
- **TensorRT timing cache** — separate from the engine cache; stored at `{BGE_M3_CACHE_DIR}/trt-timing`. Persists per-tactic kernel timings so the TRT builder can skip tactic-selection on subsequent engine builds. Enabled unconditionally alongside the engine cache.
- **Client disconnect / hedged-race cancellation.** ORT `session.run()` is a synchronous blocking C call wrapped in `spawn_blocking`; it cannot be interrupted mid-MatMul. The worker therefore performs two best-effort checks via `oneshot::Sender::is_closed()`: (1) pre-dispatch — if the request was abandoned while sitting in the worker queue, skip inference entirely and log `WARN request abandoned by client before dispatch`; (2) post-completion — if the client disconnected during inference, log `WARN request abandoned by client during inference` with `inference_ms_so_far` and `chunks` so operators can size the router's hedge budget against the actual GPU wall time. Mid-inference cancellation is a future enhancement that would require plumbing a cancellation token into `embed_dense`/`embed_sparse`/`embed_both` between bin-packed chunks.
- **ort 2.0.0-rc.12 TensorRT builder methods** — use `with_engine_cache(bool)`, `with_engine_cache_path(impl ToString)`, `with_timing_cache(bool)`, `with_timing_cache_path(impl ToString)`, `with_fp16(bool)`. The `*_enable` suffixed variants (`with_engine_cache_enable`, `with_fp16_enable`) do not exist in this version. CUDA uses `with_device_id(i32)`.
- **Init container and `healthCheckGracePeriodSeconds`:** ECS measures the service `healthCheckGracePeriodSeconds` from task start, not from when the main container starts. When using the `BGE_M3_WARMUP_ONLY=1` init container pattern, the warmup container can run for 90–180 min before the main container starts. Set `healthCheckGracePeriodSeconds` to at least `10800` (3 hours) if using the full 16-shape grid on an L4, so the ECS service does not fail the main container's health checks before it has a chance to start. Once the cache is warm on subsequent deploys, the grace period is not consumed — only the warmup container's compile window is affected on cold cache. The main container's Dockerfile `HEALTHCHECK startPeriod` can remain short (120 s) because its cache is already warm when it starts.
- **TRT plans are compute-capability-specific:** TRT engine plan files embed the GPU compute capability (`sm_XX`), CUDA version, TRT version, and ONNX model SHA. An L4 compiles to `sm_89`; an L40S also compiles to `sm_89`; an A10G compiles to `sm_86`; a T4 compiles to `sm_75`; an RTX PRO 6000 Blackwell (AWS g7e) compiles to `sm_120`. Plans compiled for one SM version **cannot run on a different SM version** — the TRT EP silently recompiles on a cache miss. Keep ASG instance families consistent when using a persistent engine cache. **Same SM ≠ same GPU model:** L4 (sm_89, 24 GB VRAM) and L40S (sm_89, 48 GB VRAM) share the same compute capability; their plan files are architecturally compatible, but TRT's tactic-selection workspace budget scales with available VRAM, so a plan compiled on an L40S may select a tactic that exceeds L4 VRAM headroom. For best reliability, warm engines on the same GPU model that will serve inference.
- **ORT already namespaces TRT plan files by SM version — no per-family subdirectory needed:** ORT embeds `_smXX` in every engine filename (example: `TensorrtExecutionProvider_TRTKernel_..._sm89.engine`, `..._sm120.engine`). Plans for different GPU architectures (e.g., L40S at `sm_89` and RTX PRO 6000 Blackwell at `sm_120`) coexist safely in the same `{BGE_M3_CACHE_DIR}/trt-engines/` directory — no `BGE_M3_TRT_CACHE_NAMESPACE` env var or per-family subdirectory is required. Each GPU architecture compiles its own plans on first task placement; subsequent placements on that architecture are served from the EFS cache. Plans are also portable within the same GPU family regardless of instance size (e.g., a g6e.4xlarge and g6e.8xlarge have identical GPUs; warming on either warms both) because TRT plan files key on compute capability + CUDA/TRT version + ONNX model SHA, not on VRAM size or instance dimensions.
- **Dockerfile.cuda downloads the Microsoft ORT GPU tarball — NOT pyke.io.** The previous pyke.io download required careful `nvrtx` vs `cu12` feature-flag selection (the root cause of the 2026-05 codekeeper silent-persistence outage where `nvrtx` omitted `libonnxruntime_providers_tensorrt.so`, causing a silent CPU fallback) and needed a Python lzma2 extraction step. `Dockerfile.cuda` now uses the upstream Microsoft tarball at `https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/onnxruntime-linux-x64-gpu-{VERSION}.tgz`, extracted with plain `tar xzf`. That archive ships everything in `lib/`: `libonnxruntime.so.1.24.2` + `libonnxruntime.so.1` symlink, `libonnxruntime_providers_shared.so`, `libonnxruntime_providers_tensorrt.so` (standard `TensorrtExecutionProvider`), and `libonnxruntime_providers_cuda.so`. ORT is linked **dynamically** via `ORT_PREFER_DYNAMIC_LINK=1` (set as `ENV` in the builder stage before `cargo build`), which switches `ort-sys` from `libonnxruntime.a` static link to `dylib=onnxruntime`. With dynamic linking, `dladdr(&Env::Default, &dl_info)` returns the path to `libonnxruntime.so.1.24.2` (not `argv[0]`), so `GetRuntimePath()` == `dirname(dli_fname)` == the directory where the `.so` was loaded from. In the runtime stage: `libonnxruntime.so.1.24.2` and all providers are copied to `/usr/local/bin/`; the `libonnxruntime.so.1 → libonnxruntime.so.1.24.2` soname symlink is created with `RUN ln -s` (Docker COPY dereferences symlinks, so it must be recreated); `/usr/local/bin/` is added to `ldconfig` via `/etc/ld.so.conf.d/onnxruntime.conf` so the dynamic linker resolves `libonnxruntime.so.1` at process startup. The runtime stage still installs `libnvinfer10`, `libnvinfer-plugin10`, `libnvonnxparsers10` from NVIDIA's CUDA APT repo because `libonnxruntime_providers_tensorrt.so` links against `libnvinfer.so.10`, `libnvinfer_plugin.so.10`, and `libnvonnxparser.so.10`. Do not add `download-ort` back — it is only for CI where `ORT_LIB_LOCATION` is unavailable. **Upgrading ORT:** bump `ARG ORT_VERSION=X.Y.Z` in `Dockerfile.cuda` (the URL is constructed from that ARG), re-pin `ARG ORT_MS_SHA256` (download the new tarball and run `sha256sum`), update the `ln -s libonnxruntime.so.X.Y.Z` version string in the runtime stage, and bump `Cargo.toml` `ort = "=X.Y.Z"` to match.
- **`ort` `tracing` Cargo feature is required for visibility.** The `ort` crate's internal `crate::info!`/`warn!`/`error!` calls — including the `apply_execution_providers` "Successfully registered `<EP>`" / "Couldn't register `<EP>`" / "An error occurred when attempting to register `<EP>`" lines, and the C-side `tracing_logger` callback that forwards ONNX Runtime's own log lines into `tracing` — are all gated on `#[cfg(feature = "tracing")]`. Without the feature, ORT registration failures are completely silent — which masked the `nvrtx`-prebuilt root cause for an entire deploy cycle. `Cargo.toml` enables `ort/tracing` explicitly. CloudWatch will now show `target=ort` events for registration outcomes; grep them whenever a TRT regression is suspected.
- **`error_on_failure()` on GPU EPs surfaces registration failures loudly.** `ort::ep::ExecutionProviderDispatch::new` defaults `error_on_failure: false`, which routes a failed EP registration into a `WARN`/`ERROR` log (gated on `ort/tracing`) and silently continues with no EPs registered, falling back to MLAS/CPU. For the `tensorrt` and `cuda` selections in `src/embedder/session.rs::execution_providers` we now call `.error_on_failure()` on the dispatch so the registration error propagates as an `ort::Error` from `Session::builder().with_execution_providers(...)`, which `load_session` already converts into a worker-load failure. Net effect: missing provider `.so` files, missing TensorRT runtime libs, or CUDA driver problems trip a clear startup failure — not a silent CPU fallback. The hard-coded `ep_active="TensorRT"` log field describes the requested EP, not what ORT actually built; treat it as intent, not ground truth.
- **TRT profile-based engine files: one `.engine` file covers all `(batch, seq)` shapes — use `engine_count_after > 0` as the postcondition, NOT `engine_count_delta > 0`.** ORT's TRT EP stores one profile-based `.engine` file per fused subgraph. The profile records `[min, max]` ranges for each input dimension; when a new shape falls inside the existing range the file is a cache hit (unchanged); when it falls outside, the EP rewrites the file in-place with an expanded range. Either way, after the first cold compile the on-disk count stays at 1 — `engine_count_delta == 0` and `engine_count_increased == false` are the normal steady-state for every subsequent shape. The old `engine_count_delta <= 0` ERROR rule produced false-positive ERRORs on every shape after the first one (confirmed in production: workers 0/1/3 all showed `before=1, after=1, increased=0` on every shape). The correct rule: `engine_count_after == 0` (no engine file at all) is the persistence failure. Both `prewarm_persistence_postcondition_failed` and the per-shape WARN in `runner.rs` now use this condition. The `engine_count_after` field (not `engine_count_delta`) is the authoritative postcondition input in all call sites.
- **Static-linked ORT requires providers next to the binary, not in `/usr/local/lib/onnxruntime/`. `LD_LIBRARY_PATH` and `ldconfig` cannot fix it.** *(Historical — `Dockerfile.cuda` now uses dynamic linking via `ORT_PREFER_DYNAMIC_LINK=1`. With dynamic linking `ldconfig` works correctly and `GetRuntimePath()` resolves via the `.so`'s own `dli_fname`. The analysis below explains why static linking required the `CMD` absolute-path trick, preserved as reference if static linking is ever revisited.)* ORT 1.24.2's `provider_bridge_ort.cc::ProviderLibrary::Load` computes the provider `dlopen` path as `Env::Default().GetRuntimePath() + filename` (e.g. `libonnxruntime_providers_shared.so`). `GetRuntimePath` (in `core/platform/posix/env.cc`) calls `dladdr(&Env::Default, &dl_info)`, then returns `std::filesystem::absolute(dl_info.dli_fname).remove_filename()`. When `libonnxruntime.a` is statically linked into our binary, `dladdr` resolves the address to the main executable, and on glibc 2.39 (Ubuntu 24.04) `dl_info.dli_fname` is `argv[0]` — NOT `AT_EXECFN`. Empirically verified with a tiny C program built into the same `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04` base: `CMD ["bge-m3-embedding-server"]` (relative basename) gives `dli_fname="bge-m3-embedding-server"`; `CMD ["/usr/local/bin/bge-m3-embedding-server"]` gives `dli_fname="/usr/local/bin/bge-m3-embedding-server"`. With the relative-basename form, `std::filesystem::absolute()` resolves against process cwd, which is `/` by default in Docker, so the runtime path becomes `/`, and ORT tries to `dlopen("/libonnxruntime_providers_shared.so", ...)`. Per `dlopen(3)`, **any `/` in the filename forces it to be interpreted as a literal pathname** — `dlopen` will not search `LD_LIBRARY_PATH` or `/etc/ld.so.cache`. So the previous fix attempt (copying providers to `/usr/local/lib/onnxruntime/` + `/etc/ld.so.conf.d/onnxruntime.conf` + `ldconfig`) was structurally incapable of working. The correct fix is to make the runtime path stable and known: place provider `.so` files in `/usr/local/bin/` next to the binary, and use `CMD ["/usr/local/bin/bge-m3-embedding-server"]` (absolute path) so `argv[0]` and therefore `dli_fname` are absolute regardless of cwd. ORT then computes the runtime path as `/usr/local/bin/` and finds the providers. There is no ORT env-var override for this path (no `ORT_LIB_PATH` / `ORT_DYLIB_DIR` / similar in 1.24.2). Verification: `docker run --rm --entrypoint /bin/sh bge-m3-embedding-server:cuda -c 'ls /usr/local/bin/libonnxruntime_providers_*.so'` should list all three providers. On a host without an NVIDIA driver, the binary now fails inside `tensorrt_execution_provider_info.cc:81` (CUDA failure 35) — proof that the providers loaded successfully and ORT got past the `dlopen`.
