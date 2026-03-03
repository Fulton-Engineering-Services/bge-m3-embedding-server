# Architecture Overview

`bge-m3-embedding-server` is an Axum HTTP server that serves **BGE-M3
dense and sparse embeddings** over an OpenAI-compatible REST API via direct
[ONNX Runtime](https://onnxruntime.ai/) integration. It is designed for
low-latency, concurrent inference in Docker-based and native deployments,
with optional CoreML EP acceleration on Apple Silicon.

## High-Level Component Diagram

```mermaid
graph TB
    subgraph Clients
        KnowledgeBase["knowledge-base-server"]
        Coordinator["distributed-fleet-coordinator"]
    end

    subgraph "bge-m3-embedding-server"
        Router["Axum Router<br/>(tower middleware)"]
        Handlers["Request Handlers<br/>(dense / sparse / health / models)"]
        AppState["AppState<br/>(pool, ready flag, config)"]
        Pool["EmbedPool<br/>(mpsc channel + atomics)"]

        subgraph "Worker Pool (spawn_blocking)"
            W0["Worker 0<br/>ORT Session + Tokenizer"]
            W1["Worker 1<br/>ORT Session + Tokenizer"]
            Wn["Worker N<br/>ORT Session + Tokenizer"]
        end
    end

    Cache[("ONNX Model Cache<br/>(BGE_M3_CACHE_DIR)")]

    KnowledgeBase -->|"POST /v1/embeddings<br/>POST /v1/sparse-embeddings"| Router
    Coordinator -->|"POST /v1/embeddings"| Router
    Router --> Handlers
    Handlers --> AppState
    AppState --> Pool
    Pool -->|"EmbedRequest via mpsc"| W0
    Pool -->|"EmbedRequest via mpsc"| W1
    Pool -->|"EmbedRequest via mpsc"| Wn
    W0 --- Cache
    W1 --- Cache
    Wn --- Cache
```

## Module Layout

| Module | File | Responsibility |
|--------|------|----------------|
| `main` | `src/main.rs` | Bootstrap, router construction, readiness probe |
| `config` | `src/config.rs` | Env-var configuration (`Config::from_env`) |
| `state` | `src/state.rs` | Shared `AppState` struct |
| `handler` | `src/handler.rs` | HTTP handlers and input validation |
| `embedder` | `src/embedder.rs` | Worker pool, ORT session + tokenizer loading, inference, channel dispatch |
| `weights` | `src/weights/mod.rs` | Bundled sparse-linear projection weights (`sparse_linear.safetensors`) |
| `models` | `src/models.rs` | Request/response serde types |
| `error` | `src/error.rs` | `AppError` → HTTP status code mapping |

## Worker Pool Design

The core concurrency primitive is a **shared-receiver worker pool**.
Workers run on Tokio `spawn_blocking` threads (each loads its own model
instances), and all share a single bounded `mpsc` channel through an
`Arc<Mutex<Receiver>>`.

```mermaid
graph LR
    Handler["Handler<br/>(async)"]
    Channel["Bounded mpsc<br/>(capacity = N × 4)"]
    Mutex["Arc&lt;Mutex&lt;Receiver&gt;&gt;"]
    W0["Worker 0"]
    W1["Worker 1"]

    Handler -->|"send(EmbedRequest)"| Channel
    Channel --> Mutex
    Mutex -->|"lock → recv()"| W0
    Mutex -->|"lock → recv()"| W1

    W0 -->|"oneshot reply"| Handler
    W1 -->|"oneshot reply"| Handler
```

### Why shared-receiver?

The `Mutex` serializes **which worker waits for the next message**, not the
inference work itself. As soon as `recv()` returns a message, the worker
releases the lock and begins ONNX inference (~10–100 ms). Meanwhile the next
idle worker immediately acquires the lock and begins its own `recv()`. Under
normal load, at most one request is queued behind the lock at any time.

Each `EmbedRequest` carries a `oneshot::Sender` so the handler can `await`
its specific result without blocking other in-flight requests.

### Key Data Structures

```mermaid
classDiagram
    class AppState {
        +EmbedPool pool
        +AtomicBool ready
        +usize max_batch
        +usize total_workers
    }

    class EmbedPool {
        -Sender~EmbedRequest~ tx
        -Arc~AtomicUsize~ live_workers
        -Arc~AtomicUsize~ loaded_workers
        +dense(texts) Result~Vec~Vec~f32~~~
        +sparse(texts) Result~Vec~SparseEmbedding~~
        +live_worker_count() usize
        +loaded_worker_count() usize
    }

    class EmbedRequest {
        <<enum>>
        Dense: texts, reply
        Sparse: texts, reply
    }

    class WorkerGuard {
        -Arc~AtomicUsize~
        +drop() decrements live_workers
    }

    AppState --> EmbedPool
    EmbedPool ..> EmbedRequest : sends via channel
    EmbedRequest ..> WorkerGuard : runs inside worker
```

## Middleware Stack

The Axum router applies four `tower` layers in order (outermost first):

| Layer | Purpose |
|-------|---------|
| `SetRequestIdLayer` | Generates a UUID `x-request-id` for each request |
| `TraceLayer` | Logs request/response spans with timing via `tracing` |
| `PropagateRequestIdLayer` | Copies `x-request-id` from the request into the response |
| `DefaultBodyLimit` | Caps request bodies at 2 MiB |

```mermaid
graph TB
    Req["Incoming Request"] --> SetId["SetRequestIdLayer<br/>(generate UUID)"]
    SetId --> Trace["TraceLayer<br/>(start span)"]
    Trace --> Propagate["PropagateRequestIdLayer<br/>(copy to response)"]
    Propagate --> BodyLimit["DefaultBodyLimit<br/>(2 MiB cap)"]
    BodyLimit --> Route["Route Handler"]
    Route --> Propagate
    Propagate --> Trace
    Trace --> SetId
    SetId --> Resp["Response with x-request-id"]
```

## Model Lifecycle

Each worker independently manages a single `Option<(ort::session::Session, tokenizers::Tokenizer)>`.
A single ORT session produces both `sentence_embedding` (dense) and `token_embeddings`
(sparse) outputs in one forward pass. The sparse-linear projection (weight `[1024]` +
bias scalar from bundled `sparse_linear.safetensors`) is applied as a post-processing
step on the `token_embeddings` output.

Models transition through three states during a worker's lifetime:

1. **Loading** — ORT session and tokenizer initialized from ONNX files at startup (or reload).
2. **Loaded** — Session and tokenizer in memory, processing requests.
3. **Unloaded** — Session and tokenizer dropped after `BGE_M3_IDLE_TIMEOUT_SECS` of
   inactivity; reloaded from cache on next request.

Workers themselves never exit during idle — only their session/tokenizer pair is
dropped. The `live_workers` atomic tracks thread liveness while
`loaded_workers` tracks model presence. This distinction drives the health
endpoint's ability to differentiate `"ok"` from `"idle"` states.

## Configuration

All configuration is read once at startup from environment variables via
`Config::from_env`. The server does not support runtime reconfiguration.

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | ONNX model cache directory |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Worker thread count (min 1) |
| `BGE_M3_MAX_BATCH` | `256` | Max texts per request (min 1) |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Idle unload timeout; `0` disables |
| `BGE_M3_ONNX_BATCH_SIZE` | `8` (macOS) / `256` (other) | Max texts per `session.run()` call; chunked to avoid CoreML OOM |
| `BGE_M3_MODEL` | `fp32` | Model variant: `fp32` = BAAI/bge-m3 (~2.16 GB/session); `fp16` = Xenova/bge-m3 (~1.08 GB/session). FP16 recommended for Apple Silicon. See [docs/model-variants.md](model-variants.md). |

> **Apple Silicon:** A `.cargo/config.toml` with `rustflags = ["-C", "target-cpu=native"]`
> for `aarch64-apple-darwin` is committed in the repo. This enables M2+ instruction set
> extensions (i8mm, bf16) beyond the M1 baseline. CI and Docker builds are unaffected
> (they target `x86_64-unknown-linux-gnu` or `aarch64-unknown-linux-gnu`).
