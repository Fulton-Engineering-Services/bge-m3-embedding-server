# CLAUDE.md — bge-m3-axum-fastembed-rs

Axum HTTP server wrapping fastembed-rs to serve BGE-M3 dense and sparse embeddings.

## Consumers

- **mcp-local-knowledge-base** — calls `/v1/sparse-embeddings` and `/v1/embeddings` to index and search documents
- **dpos-coordinator** — calls `/v1/embeddings` for semantic memory retrieval

## Build & Test Commands

```bash
# Build
cargo build

# Run tests (requires cargo-nextest)
cargo nextest run --all-features --no-tests=pass

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
| `GET` | `/health` | Readiness probe — `200 OK` when ready, `503` while loading |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | Path where ONNX model files are cached |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Number of worker threads (each loads its own model instance; min 1) |
| `BGE_M3_MAX_BATCH` | `256` | Maximum number of texts accepted per request (min 1) |

## Architecture

The server uses a **worker pool** pattern to handle concurrent embedding requests:

- Workers are spawned via `spawn_blocking`, each loading dense + sparse model instances.
- Work dispatched through bounded `mpsc` channel; workers share receiver via `Arc<Mutex<Receiver>>`.
- Readiness: each worker signals via a separate `mpsc` readiness channel after model load. The init task collects all signals, then a warm-up probe runs both dense and sparse inference before setting the `AtomicBool` ready flag.
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

## Gotchas

- `fastembed::SparseEmbedding` does not implement `Debug` — use `.err().expect()` instead of `.unwrap_err()` on `Result<Vec<SparseEmbedding>>`
- Stale model cache causes silent worker load failures ("Worker exited before signaling readiness") — fix by clearing `BGE_M3_CACHE_DIR`
- Config tests use `from_lookup()` closure pattern instead of `env::set_var` to avoid process-global state mutation under parallel test execution
