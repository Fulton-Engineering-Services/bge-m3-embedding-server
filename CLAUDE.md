# CLAUDE.md — bge-m3-axum-fastembed-rs

Axum HTTP server serving BGE-M3 dense and sparse embeddings via direct ONNX Runtime integration.

## Consumers

- **mcp-local-knowledge-base** — calls `/v1/sparse-embeddings` and `/v1/embeddings` to index and search documents
- **dpos-coordinator** — calls `/v1/embeddings` for semantic memory retrieval

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

# Lint
cargo clippy --all-targets -- -D warnings

# Format check
cargo fmt --check

# Supply chain audit (requires cargo-deny)
cargo deny check
```

## Run Locally

```bash
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```

On first run, the model files are downloaded to `BGE_M3_CACHE_DIR`. Subsequent runs reuse the cache.
The server starts accepting requests once model warm-up completes (watch logs for "ready").

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | Path where ONNX model files are cached |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Number of worker threads (each loads its own model instance; min 1) |
| `BGE_M3_MAX_BATCH` | `256` | Maximum number of texts accepted per request (min 1) |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Seconds of inactivity before models are unloaded; `0` disables idle unloading |
| `BGE_M3_ONNX_BATCH_SIZE` | `8` (macOS) / `256` (other) | Max texts per ONNX `session.run()` call. On macOS, CoreML `MLProgram` with `FastPrediction` pre-allocates the full intermediate-tensor workspace; BGE-M3 at batch=50×seq=512 can peak at ~35 GB, triggering Jetsam OOM kills. Chunking to 8 keeps peak under 6 GB. |

## Architecture

The server uses a **worker pool** pattern to handle concurrent embedding requests:

- Workers are spawned via `spawn_blocking`, each loading a **single ORT session** that produces both dense and sparse outputs from one ONNX model.
- Work dispatched through bounded `mpsc` channel; workers share receiver via `Arc<Mutex<Receiver>>`.
- Readiness: each worker signals via a separate `mpsc` readiness channel after model load. The init task collects all signals, then a warm-up probe runs both dense and sparse inference before setting the `AtomicBool` ready flag.
- **Cold-start ordering**: worker 0 (the "leader") is spawned and awaited first to ensure the model cache is warm before followers start. This prevents `hf-hub` file-lock contention when `BGE_M3_WORKERS > 1` and the cache is empty.
- **Idle unloading**: after `BGE_M3_IDLE_TIMEOUT_SECS` of no requests, workers drop their `Option<(Session, Tokenizer)>`. On the next request, models are reloaded transparently (~10–30 s from cache). The `loaded_workers` counter drives the `"idle"` health state. Workers themselves never exit during idle — only their model instances are dropped.
- **Sparse projection**: sparse embeddings are computed by projecting token hidden states through a bundled `sparse_linear.safetensors` weight vector (4 KB), then applying ReLU and max-pooling. This replaces fastembed's separate sparse model with a single-session approach.
- HTTP observability is provided by `tower-http::TraceLayer`.

## Docker

```bash
# Build
docker build -t bge-m3-axum-fastembed-rs .

# Run (mount a host directory to persist the model cache across restarts)
docker run --rm \
  -p 8081:8081 \
  -v /path/to/model-cache:/cache \
  bge-m3-axum-fastembed-rs
```

The container exposes port `8081`. The built-in `HEALTHCHECK` polls `/health` every 10 seconds
with a 120-second start period to allow time for model download and ONNX initialization.

## Releasing

The Release workflow creates git tags automatically. **Do not create tags locally.**
To release: bump version in `Cargo.toml`, commit, push to `main`. The workflow handles tag creation, multi-arch Docker builds, and GitHub Release.

## Security Considerations

- **Rate limiting** (SEC-3): No application-level rate limiting on embedding endpoints. This is an internal LAN service; concurrency is bounded by the worker pool (`BGE_M3_WORKERS`). Network-level controls (firewall rules, reverse proxy throttling) are the intended mitigation for production deployments.
- **TLS CA bundles** (SEC-4): `hf-hub` uses `native-tls`, delegating certificate validation to the system keychain. Docker containers based on minimal base images may lack full CA bundles. Production Docker images should include `ca-certificates` or use `ORT_LIB_LOCATION` with pre-downloaded models to avoid runtime TLS calls.
- **Cache directory path** (SEC-5): `BGE_M3_CACHE_DIR` is used without path normalization. This is an accepted risk because the variable is operator-controlled (set via environment or Docker compose). Symlink traversal or path injection requires host-level compromise.

## Gotchas

- Stale model cache causes silent worker load failures ("Worker exited before signaling readiness") — fix by clearing `BGE_M3_CACHE_DIR`
- Config tests use `from_lookup()` closure pattern instead of `env::set_var` to avoid process-global state mutation under parallel test execution
- Always run `cargo fmt --all` before pushing — CI fails `cargo fmt --all --check` even when all tests pass
- `gh pr merge` requires `--admin` to bypass branch protection, or `--auto` to queue for merge after CI passes
- After a squash-merged PR, reset local main with `git reset --hard origin/main` to avoid divergent merge commits
