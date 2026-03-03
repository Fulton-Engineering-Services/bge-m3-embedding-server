# bge-m3-embedding-server — Project Overview

## Purpose
Axum HTTP server wrapping `fastembed-rs` to serve BGE-M3 dense and sparse (SPLADE-style) embeddings over HTTP. Provides an OpenAI-compatible `/v1/embeddings` endpoint plus a custom `/v1/sparse-embeddings` endpoint.

## Consumers
- **mcp-local-knowledge-base** — calls `/v1/sparse-embeddings` and `/v1/embeddings`
- **dpos-coordinator** — calls `/v1/embeddings` for semantic memory retrieval

## Tech Stack
- **Language**: Rust (MSRV 1.88), edition 2021
- **HTTP framework**: Axum 0.8 + Tower
- **Async runtime**: Tokio (multi-thread)
- **Embedding engine**: fastembed 5
- **Serialization**: serde / serde_json
- **Observability**: tracing + tracing-subscriber (JSON structured logs), tower-http TraceLayer
- **Testing**: cargo-nextest + proptest + criterion (benches)
- **Supply chain**: cargo-deny

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/embeddings | Dense embeddings (OpenAI-compatible) |
| POST | /v1/sparse-embeddings | Sparse embeddings |
| GET | /health | Readiness probe (200/503) |
| GET | /v1/models | Fleet discovery |

## Version
Current: 0.8.0 (Cargo.toml)

## Repository
GitHub: Fulton-Engineering-Services/bge-m3-embedding-server
