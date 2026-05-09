# CLAUDE.md — bge-m3-embedding-server

Axum HTTP server serving BGE-M3 dense and sparse embeddings via direct ONNX Runtime integration.

## Use Cases

- Document indexing and hybrid search via the `/v1/embeddings` (dense) and `/v1/sparse-embeddings` (SPLADE-style) endpoints
- Semantic memory retrieval for agent / RAG workloads via the OpenAI-compatible `/v1/embeddings` endpoint
- Drop-in replacement for hosted embedding APIs when local inference, data residency, or BGE-M3-specific sparse vectors are required

## Features

| Feature | Description |
|---------|-------------|
| `coreml-profile` | Emit per-op CoreML hardware dispatch decisions to stderr at model load. Diagnostic only — use when profiling which ops land on GPU vs CPU. |

## Build & Test Commands

```bash
# Build
cargo build

# Run tests (requires cargo-nextest)
cargo nextest run --all-features --no-tests=warn

# Run equivalence tests (requires model download + BGE_M3_EQUIVALENCE_TEST=1)
BGE_M3_EQUIVALENCE_TEST=1 BGE_M3_CACHE_DIR=/tmp/bge-m3-cache \
  cargo test --test equivalence -- --ignored --nocapture

# Lint
cargo clippy --all-targets -- -D warnings

# Format check
cargo fmt --check

# Supply chain audit (requires cargo-deny)
cargo deny check

# Build with CoreML dispatch profiling (emits per-op GPU/CPU/ANE decisions to stderr at model load)
cargo build --features coreml-profile
```

## Run Locally

```bash
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```

On first run, the model files are downloaded to `BGE_M3_CACHE_DIR`. Subsequent runs reuse the cache.
The server starts accepting requests once model warm-up completes (watch logs for "ready").

During startup on Linux, the server:
1. Detects available memory (cgroup v2 → v1 → `/proc/meminfo`).
2. Runs a startup probe on the leader worker sweeping `(batch, seq)` shapes from 64 to `MAX_SEQ_LENGTH`.
3. Fits a quadratic cost model `a * N + b * N²` to measured RSS deltas.
4. Sets `max_workspace_bytes` for the bin-packer accordingly.

Set `BGE_M3_DISABLE_AUTO_BUDGET=1` to skip the probe (uses conservative defaults).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/embeddings` | Dense embeddings (OpenAI-compatible) |
| `POST` | `/v1/sparse-embeddings` | Sparse embeddings (BGE-M3 SPLADE-style) |
| `GET` | `/health` | Readiness probe — `200 OK` when ready, `503` while loading; see health states below |
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
| `BGE_M3_WORKERS` | `2` | Number of worker threads (each loads its own model instance; min 1) |
| `BGE_M3_MAX_BATCH` | `256` | Maximum number of texts accepted per request (min 1) |
| `BGE_M3_MAX_SEQ_LENGTH` | `8192` | Maximum tokenized sequence length. Range `[1, 8192]`. Lower values reduce memory use; `8192` is the BGE-M3 model's published maximum. |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Seconds of inactivity before models are unloaded; `0` disables idle unloading |
| `BGE_M3_MODEL` | `fp16` | Model variant: `fp32` = `BAAI/bge-m3` (~2.16 GB/session); `fp16` = `Xenova/bge-m3` (~1.08 GB/session); `int8` = `Xenova/bge-m3` quantized (~568 MB/session). See model variant notes below. |

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

The server uses a **worker pool** pattern to handle concurrent embedding requests:

- Workers are spawned via `spawn_blocking`, each loading a **single ORT session** that produces both dense and sparse outputs from one ONNX model.
- Work dispatched through bounded `mpsc` channel; workers share receiver via `Arc<Mutex<Receiver>>`.
- **Tokenize-once, bin-pack:** each request tokenizes all texts in one pass, then the bin-packer (`src/binpack.rs`) groups them into ONNX `session.run()` calls that fit within `max_workspace_bytes`. Chunks are padded only to the longest sequence within that chunk — not the global maximum. Results are scattered back to the original input order.
- **Quadratic cost model:** workspace per call is `a * (batch * seq) + b * (batch * seq²)`. At `MAX_SEQ_LENGTH=8192`, the quadratic term dominates by ~16× vs linear, so the bin-packer naturally puts fewer texts per chunk for long sequences and more texts per chunk for short sequences.
- **Startup probe (Linux):** after the leader worker loads its model, the server sweeps 18 `(batch, seq)` shapes, measures peak RSS deltas, and fits the cost model coefficients via least squares. Falls back to conservative defaults if RSS measurement fails.
- **Cold-start ordering**: worker 0 (the "leader") is spawned and awaited first to ensure the model cache is warm before followers start. This prevents `hf-hub` file-lock contention when `BGE_M3_WORKERS > 1` and the cache is empty.
- **Idle unloading**: after `BGE_M3_IDLE_TIMEOUT_SECS` of no requests, workers drop their model instances. On the next request, models are reloaded transparently (~10–30 s from cache). Workers never exit during idle.
- **Sparse projection**: computed by projecting token hidden states through a bundled `sparse_linear.safetensors` weight vector (4 KB), then applying ReLU and max-pooling.
- HTTP observability via `tower-http::TraceLayer`.

## Long-Context Support

BGE-M3's BAAI/bge-m3 FP32 export ships positional embeddings to 8192 tokens. The server's
previous 512-token cap was self-imposed; it is removed in this release.

**To use long-context embeddings:** the default is now `BGE_M3_MAX_SEQ_LENGTH=8192`. No
operator action required.

**Memory implications:** attention is `O(seq²)`, so going from 512 to 8192 multiplies
per-chunk workspace by ~256× for the same batch size. The bin-packer and auto-budget
probe handle this automatically — long-seq requests get fewer texts per ONNX call.

**Equivalence validation:** run `scripts/generate_equivalence_fixtures.py` then
`BGE_M3_EQUIVALENCE_TEST=1 cargo test --test equivalence -- --ignored --nocapture`
to validate that embeddings at long context lengths are equivalent to the FP32 reference.

**macOS scope limitation (v1):** the startup probe and auto-budget derive memory from
`/proc` and cgroups, which are Linux-only. The native macOS LaunchAgent build (the
`scripts/install-bge-m3-apple.sh` path) uses host RAM from `sysctl hw.memsize` for
memory detection but cannot read process RSS, so conservative defaults apply. Apple
Silicon CoreML behavior is unchanged. **Note:** the Linux Docker image runs `/proc`
+ cgroup paths normally even on Apple Silicon hosts (the probe sees the Docker VM,
not the macOS host), so the scope limitation only affects the native macOS binary.

## Verifying auto-budget tuning after deploy

After the server starts and `/health` returns `ok`, curl the health endpoint to confirm the
probe ran and the expected knobs are in effect:

```bash
curl http://localhost:8081/health | jq '{status,max_seq_length,tuning}'
```

Expected fields in the response:

| Field | Meaning |
|---|---|
| `max_seq_length` | Should match the configured (or default) `BGE_M3_MAX_SEQ_LENGTH` — e.g. `8192` |
| `tuning.memory_source` | How memory was detected — `cgroup_v2`, `cgroup_v1`, or `proc_meminfo` |
| `tuning.available_bytes` | Container memory visible to the server |
| `tuning.max_workspace_bytes` | Derived workspace budget after applying `BGE_M3_MEMORY_SAFETY_FACTOR` |
| `tuning.a_bytes_per_token` | Probe-fitted linear coefficient; should be ~18000–20000 for fp16 on amd64 |
| `tuning.b_bytes_per_token_sq` | Probe-fitted quadratic coefficient; should be ~5–8 for fp16 |
| `tuning.model_rss_bytes_per_worker` | Per-worker model RSS — ~1.1 GB for fp16 |

If `tuning` is absent or `max_seq_length` does not match the expected value, check whether
`BGE_M3_DISABLE_AUTO_BUDGET=1` was accidentally set, or whether the model variant lacks
positional embeddings at the configured length (the startup probe will error in that case).

## Docker

```bash
# Build
docker build -t bge-m3-embedding-server .

# Run (mount a host directory to persist the model cache across restarts)
docker run --rm \
  -p 8081:8081 \
  -v /path/to/model-cache:/cache \
  bge-m3-embedding-server
```

The container exposes port `8081`. The built-in `HEALTHCHECK` polls `/health` every 10 seconds
with a 120-second start period to allow time for model download, ONNX initialization, and
the startup probe.

## Releasing

The Release workflow creates git tags automatically. **Do not create tags locally.**
To release: bump version in `Cargo.toml`, commit, push to `main`. The workflow handles tag creation, multi-arch Docker builds, and GitHub Release.

## Security Considerations

- **DoS / workspace bounds:** See `docs/decisions/long-context-security.md` for the full security analysis of the long-context unlock. Summary: bin-packer + cost model is the admission control; per-chunk workspace is bounded; no single OOM path exists.
- **Rate limiting** (SEC-3): No application-level rate limiting. This is an internal LAN service; concurrency is bounded by the worker pool (`BGE_M3_WORKERS`). Network-level controls (firewall rules, reverse proxy throttling) are the intended mitigation.
- **TLS CA bundles** (SEC-4): `hf-hub` uses `native-tls`, delegating certificate validation to the system keychain. Docker containers based on minimal base images may lack full CA bundles. Production Docker images should include `ca-certificates` or use `ORT_LIB_LOCATION` with pre-downloaded models to avoid runtime TLS calls.
- **Cache directory path** (SEC-5): `BGE_M3_CACHE_DIR` is used without path normalization. This is an accepted risk because the variable is operator-controlled. Symlink traversal or path injection requires host-level compromise.

## Gotchas

- Stale model cache causes silent worker load failures ("Worker exited before signaling readiness") — fix by clearing `BGE_M3_CACHE_DIR`
- Config tests use `from_lookup()` closure pattern instead of `env::set_var` to avoid process-global state mutation under parallel test execution
- Always run `cargo fmt --all` before pushing — CI fails `cargo fmt --all --check` even when all tests pass
- `gh pr merge` requires `--admin` to bypass branch protection, or `--auto` to queue for merge after CI passes
- After a squash-merged PR, reset local main with `git reset --hard origin/main` to avoid divergent merge commits
- `sudo cmd > /tmp/file` silently fails on macOS (sudo password prompt is swallowed by redirect) — use `sudo sh -c 'cmd > /tmp/file'` instead
- CoreML compiled MIL lives at `{BGE_M3_CACHE_DIR}/coreml/{hash}/N_dynamic_mlprogram/model/compiled_model.mlmodelc/model.mil` — useful for dispatch analysis and op tallying
- `BGE_M3_MODEL=fp16` loads a smaller ONNX file into ORT, but CoreML compiles it to a FP32 compute graph internally; the memory saving (~1.08 GB vs ~2.16 GB) reflects ORT session weight loading only, not CoreML runtime precision
- **`BGE_M3_ONNX_BATCH_SIZE` is deprecated.** If set, a `WARN` is logged and the value is translated to a workspace ceiling via `BGE_M3_TOKEN_BUDGET`. Remove from all deployments and use `BGE_M3_DISABLE_AUTO_BUDGET=1` + `BGE_M3_TOKEN_BUDGET` if you need to pin a specific budget.
- **Xenova FP16/INT8 long-context:** the Xenova/bge-m3 FP16 and INT8 ONNX exports at the pinned revision may have been exported with `max_position_embeddings=512`. The startup probe tests `(1, MAX_SEQ_LENGTH)` inference and fails fast with a clear error if the model cannot support the configured length. Use `BGE_M3_MODEL=fp32` or lower `BGE_M3_MAX_SEQ_LENGTH` in that case.
- **Local Docker testing on Apple Silicon:** the published image is multi-arch, so `docker build` / `docker run` on macOS picks the native `linux/arm64` variant by default. Inside the Linux container there is no CoreML EP available — only ORT's MLAS CPU path — so probe times at `MAX_SEQ_LENGTH=8192` are several minutes (vs. ~60 s on amd64 Fargate). Forcing `--platform linux/amd64` runs under Rosetta 2 and pushes probe time to 15–20 minutes; only do that to validate the amd64 build path. For fast dev-loop iteration use `BGE_M3_DISABLE_AUTO_BUDGET=1` to skip the probe entirely; for production-realistic workspace tuning, use the native LaunchAgent install instead.
- **`Dockerfile` builder image:** the builder stage is `ubuntu:24.04` (not `rust:slim-bookworm`) because the prebuilt ORT binary downloaded by `ort-sys` requires glibc ≥ 2.38 (`__isoc23_strtoul` and friends). Debian Bookworm ships glibc 2.36 and fails to link with `undefined symbol: __isoc23_strtoul`. Ubuntu 24.04 has glibc 2.39 which satisfies this. Rust is installed via `rustup-init` downloaded to a file with SHA-256 verification — never `curl | sh` (supply-chain rule).
