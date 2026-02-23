[![CI](https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/actions/workflows/ci.yml) [![Release](https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/actions/workflows/release.yml/badge.svg)](https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/actions/workflows/release.yml) [![codecov](https://codecov.io/gh/Fulton-Engineering-Services/bge-m3-axum-fastembed-rs/graph/badge.svg?token=CODECOV_TOKEN)](https://codecov.io/gh/Fulton-Engineering-Services/bge-m3-axum-fastembed-rs) [![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT) [![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/fultonengineeringservices/bge-m3-axum-fastembed-rs/pkgs/container/bge-m3-axum-fastembed-rs)

# bge-m3-axum-fastembed-rs

An Axum HTTP server that wraps [fastembed-rs](https://github.com/Anush008/fastembed-rs) to serve
BGE-M3 dense and sparse embeddings. It exposes an OpenAI-compatible `/v1/embeddings` endpoint for
dense vectors and a `/v1/sparse-embeddings` endpoint for SPLADE-style sparse vectors.

## Quick Start

### Build

```bash
cargo build --release
```

### Run

```bash
# Model files are downloaded to /tmp/bge-m3-cache on first run
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run --release
```

Wait for the log line indicating the server is ready before sending requests. On first run this
takes a minute or two while the ONNX model files download.

### Verify

```bash
# Readiness check
curl http://localhost:8081/health

# Dense embedding
curl -s http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "query: what is Rust?", "model": "bge-m3"}' | jq .

# Sparse embedding
curl -s http://localhost:8081/v1/sparse-embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["what is Rust?"]}' | jq .
```

## API Reference

Full OpenAPI 3.1 specification: [`openapi.yaml`](./openapi.yaml)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | `POST` | Dense embeddings — OpenAI-compatible |
| `/v1/sparse-embeddings` | `POST` | Sparse embeddings — BGE-M3 SPLADE-style |
| `/health` | `GET` | Readiness probe with worker pool status |

### `GET /health`

Returns `200 OK` once the worker pool is fully initialized. Returns `503 Service Unavailable`
while the models are loading.

```bash
curl http://localhost:8081/health
```

**Response (ready)**

```
200 OK
```

**Response (loading)**

```
503 Service Unavailable
```

---

### `POST /v1/embeddings`

Dense embeddings. OpenAI-compatible request and response format.

**Request**

```bash
curl -s http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "query: what is Rust?",
    "model": "bge-m3"
  }'
```

`input` accepts a single string or an array of strings. BGE-M3 query inputs should be prefixed
with `"query: "` and passage inputs with `"passage: "` for best retrieval quality.

**Response**

```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789]
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Error Response**

```json
{
  "error": {
    "message": "batch size 512 exceeds maximum 256",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

---

### `POST /v1/sparse-embeddings`

Sparse embeddings using BGE-M3's SPLADE-style sparse model.

**Request**

```bash
curl -s http://localhost:8081/v1/sparse-embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["what is Rust?"]
  }'
```

`input` accepts a single string or an array of strings.

**Response**

```json
{
  "data": [
    {
      "index": 0,
      "sparse_values": {
        "indices": [42, 100, 3527],
        "values": [0.5, 0.8, 0.3]
      }
    }
  ]
}
```

Each entry in `data` corresponds to one input string. `indices` are vocabulary token IDs and
`values` are the associated relevance weights.

**Error Response**

```json
{
  "error": {
    "message": "service unavailable: models still loading",
    "type": "service_unavailable",
    "code": 503
  }
}
```

---

## Configuration

All configuration is via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | Directory where ONNX model files are cached |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Worker thread count (each loads its own model; min 1) |
| `BGE_M3_MAX_BATCH` | `256` | Maximum texts per request (min 1) |

## Docker

### Build

```bash
docker build -t bge-m3-axum-fastembed-rs .
```

### Run

Mount a host directory as `/cache` so the model files persist across container restarts.

```bash
docker run --rm \
  -p 8081:8081 \
  -v /path/to/model-cache:/cache \
  bge-m3-axum-fastembed-rs
```

Override workers and batch size at runtime:

```bash
docker run --rm \
  -p 8081:8081 \
  -v /path/to/model-cache:/cache \
  -e BGE_M3_WORKERS=4 \
  -e BGE_M3_MAX_BATCH=128 \
  bge-m3-axum-fastembed-rs
```

The container includes a built-in `HEALTHCHECK` that polls `GET /health` every 10 seconds.
The start period is 120 seconds to allow time for model download and ONNX initialization on
first run.

## Architecture

```
HTTP request
     │
     ▼
  Axum router
     │
     ├─ POST /v1/embeddings        ─┐
     ├─ POST /v1/sparse-embeddings ─┤── handler sends EmbedRequest via mpsc channel
     └─ GET  /health                │
                                    ▼
                       bounded mpsc channel (Arc<Mutex<Receiver>>)
                          │            │
                          ▼            ▼
                      Worker 0      Worker 1  ...  Worker N
                   (spawn_blocking) (spawn_blocking)
                   TextEmbedding    TextEmbedding
                   SparseTextEmb.   SparseTextEmb.
                          │
                          ▼
                    Result sent back via oneshot channel
                          │
                          ▼
                    JSON response to client
```

**Key design decisions:**

- Each worker loads its own model instance inside `spawn_blocking` to avoid blocking the async
  runtime during ONNX inference.
- The shared `Arc<tokio::sync::Mutex<Receiver>>` lets all workers compete for work from a single
  channel, providing natural load balancing without a separate dispatcher.
- An `AtomicBool` readiness flag is set after every worker finishes loading. The `/health`
  endpoint returns `503` until all workers are ready.
- `tower-http::TraceLayer` provides per-request tracing spans at the HTTP layer.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.
