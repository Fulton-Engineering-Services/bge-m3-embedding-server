# Design: Robust BGE-M3 Embedding Service

**Date**: 2026-02-23
**Status**: Approved

## Problem

The current service is a minimal Axum wrapper around fastembed-rs that serves only sparse embeddings behind a single `Mutex`, serializing all requests. Two consumers need this service:

- **mcp-local-knowledge-base** — needs both dense (`POST /v1/embeddings`, OpenAI-compatible) and sparse (`POST /v1/sparse-embeddings`) embeddings for hybrid RRF search. Currently uses a *separate* service for dense.
- **dpos-coordinator** — needs dense embeddings via Spring AI's `OpenAiEmbeddingModel`. Currently uses Ollama.

By adding an OpenAI-compatible dense endpoint, this single service replaces two deployment dependencies.

## API

### `POST /v1/embeddings` (dense, OpenAI-compatible)

Serves Spring AI's `OpenAiEmbeddingModel` in both consumers.

**Request:**
```json
{
  "input": ["text1", "text2"],
  "model": "bge-m3"
}
```

The `model` field is accepted but ignored (only BGE-M3 is loaded). `input` may also be a single string.

**Response:**
```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.123, -0.456, ...] }
  ],
  "usage": { "prompt_tokens": 42, "total_tokens": 42 }
}
```

Dense vectors are 1024-dimensional `f32`.

### `POST /v1/sparse-embeddings` (sparse, existing format)

Serves `SparseEmbeddingClient` in mcp-local-knowledge-base. Format unchanged.

**Request:**
```json
{ "input": ["text1", "text2"] }
```

**Response:**
```json
{
  "data": [
    {
      "index": 0,
      "sparse_values": {
        "indices": [101, 2023],
        "values": [0.45, 0.33]
      }
    }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}` (200) when models are loaded, `{"status": "loading"}` (503) during startup.

### Error Format

All errors return structured JSON:
```json
{
  "error": {
    "message": "input must not be empty",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

| Condition | Status | Type |
|-----------|--------|------|
| Empty or missing input | 400 | `invalid_request_error` |
| Batch too large | 400 | `invalid_request_error` |
| Malformed JSON | 422 | `invalid_request_error` |
| Model not ready | 503 | `service_unavailable` |
| Inference failure | 500 | `internal_error` |

## Concurrency: Worker Pool

### Constraint

fastembed's `embed()` takes `&mut self`. A single Mutex serializes all requests.

### Design

A bounded worker pool where each worker owns exclusive model instances:

```
                         ┌─────────────────────┐
  HTTP request ──►  mpsc │  Worker 0            │
  HTTP request ──►  chan │    TextEmbedding      │──► spawn_blocking ──► response
  HTTP request ──►  ───► │    SparseTextEmbedding│
                         ├─────────────────────┤
                         │  Worker 1            │
                         │    TextEmbedding      │──► spawn_blocking ──► response
                         │    SparseTextEmbedding│
                         └─────────────────────┘
```

- **Pool size**: configurable via `BGE_M3_WORKERS` (default 2)
- **Channel**: bounded `tokio::sync::mpsc` — provides backpressure when all workers are busy
- **Work items**: enum `EmbedRequest { Dense { texts, reply_tx }, Sparse { texts, reply_tx } }`
- **Inference**: each `embed()` call runs inside `spawn_blocking` to avoid blocking the async runtime
- **Startup**: workers load models in parallel via `spawn_blocking`, server accepts connections only after all workers report ready

### Why 2 workers?

ONNX Runtime uses all available CPU cores per inference call. Two workers let one saturate the CPU while the other tokenizes/pre-processes the next batch. More workers would contend for CPU without throughput gain.

## Configuration

All via environment variables, consistent with 12-factor.

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | Model download/cache directory |
| `BGE_M3_BIND` | `0.0.0.0:8081` | Listen address |
| `BGE_M3_WORKERS` | `2` | Number of worker pool instances |
| `BGE_M3_MAX_BATCH` | `256` | Maximum texts per request |
| `RUST_LOG` | `info` | tracing-subscriber filter |

## Project Structure

Single crate (no workspace — scope doesn't warrant it).

```
src/
  main.rs          — startup, config, server binding
  config.rs        — Config struct, env parsing, validation
  models.rs        — request/response types (serde), OpenAI compat types
  error.rs         — AppError enum, IntoResponse impl for structured JSON errors
  embedder.rs      — EmbedWorker, EmbedPool, work item types, pool management
  handler.rs       — Axum handlers: dense_embeddings, sparse_embeddings, health
  state.rs         — AppState (holds EmbedPool handle)
```

## Observability

- **Structured logging**: `tracing` with `tracing-subscriber` env-filter
- **Request IDs**: `tower-http` `RequestId` layer, propagated in tracing spans
- **Startup timing**: log model load duration
- **Request logging**: method, path, status, latency per request via `tower-http::trace`

## Testing

- **Unit tests**: config parsing, request/response serialization, error formatting, input validation
- **Integration tests**: full handler round-trips using `axum::test` (requires model — gated behind feature flag or ignored in CI without model)

## Docker

- **Builder**: Ubuntu 24.04, rustup stable (GCC 14 for ort-sys ABI compat)
- **Runtime**: Ubuntu 24.04 slim
- **Healthcheck**: `HEALTHCHECK CMD curl -sf http://localhost:8081/health || exit 1`
- **Cache volume**: mount at `/cache` for model persistence across restarts

## Non-Goals

- Token counting accuracy (we report approximate token counts in the OpenAI response — fastembed doesn't expose exact tokenizer counts easily)
- Model selection at runtime (only BGE-M3 is loaded)
- GPU/CUDA support (CPU inference only for now)
- ColBERT/multi-vector embeddings (BGE-M3 supports these but neither consumer uses them)
