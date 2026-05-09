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
        AppState["AppState<br/>(pool, ready flag, tuning, config)"]
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
| `main` | `src/main.rs` | Bootstrap, router construction, memory detection, probe, readiness |
| `config` | `src/config.rs` | Env-var configuration (`Config::from_env`), cost-model override resolution |
| `state` | `src/state.rs` | Shared `AppState` struct, `TuningInfo` for `/health` |
| `handler` | `src/handler.rs` | HTTP handlers and input validation |
| `embedder` | `src/embedder.rs` | Worker pool, ORT session + tokenizer loading, tokenize-once + bin-pack inference |
| `binpack` | `src/binpack.rs` | `CostModel` + quadratic-aware bin-packer |
| `sysinfo` | `src/sysinfo.rs` | Memory detection (cgroup v2/v1 → host RAM) and RSS reads |
| `probe` | `src/probe.rs` | Startup workspace probe, least-squares coefficient fitting |
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
        +usize max_seq_length
        +OnceLock~TuningInfo~ tuning
    }

    class TuningInfo {
        +f64 a_bytes_per_token
        +f64 b_bytes_per_token_sq
        +usize max_workspace_bytes
        +String memory_source
        +usize available_bytes
        +usize model_rss_bytes_per_worker
    }

    class EmbedPool {
        -Sender~EmbedRequest~ tx
        -Arc~AtomicUsize~ live_workers
        -Arc~AtomicUsize~ loaded_workers
        +dense(texts) Result~Vec~Vec~f32~~~
        +sparse(texts) Result~Vec~SparseEmbedding~~
        +probe(texts) Result~ProbeResult~
        +live_worker_count() usize
        +loaded_worker_count() usize
    }

    class EmbedRequest {
        <<enum>>
        Dense: texts, reply
        Sparse: texts, reply
        Probe: texts, reply
    }

    class WorkerConfig {
        +CostModel cost_model
        +Option~Duration~ idle_timeout
        +ModelVariant model_variant
        +usize max_seq_length
    }

    class CostModel {
        +f64 a
        +f64 b
        +usize max_workspace_bytes
        +chunk_cost(count, max_seq) u128
        +fits(count, max_seq) bool
    }

    AppState --> EmbedPool
    AppState --> TuningInfo
    EmbedPool ..> EmbedRequest : sends via channel
    EmbedPool ..> WorkerConfig : spawns workers with
    WorkerConfig --> CostModel
```

## Tokenize-Once + Bin-Pack Inference Pipeline

Instead of splitting texts into static-size chunks before tokenization, the server:

1. **Tokenizes all texts in one pass** (`tokenize_no_pad`) — no padding applied yet.
2. **Bin-packs by length** — `bin_pack()` sorts texts by sequence length and greedily groups them so `a × (count × max_seq) + b × (count × max_seq²) ≤ max_workspace_bytes`.
3. **Pads per chunk** — each chunk is padded only to its own longest sequence, not the global maximum.
4. **Runs inference per chunk** and **scatters results** back to the original input indices.

This eliminates the "one long text forces the entire batch to pay `N × max_seq²` attention cost" problem. Short-text batches pack densely; long-text chunks shrink automatically.

```mermaid
graph TD
    Input["texts: Vec~String~"]
    Tok["tokenize_no_pad() — one pass, no padding"]
    Lens["seq_lens: Vec~usize~"]
    Pack["bin_pack(seq_lens, cost_model)"]
    Chunks["chunks: Vec~Vec~usize~~ (original indices)"]
    Loop["For each chunk"]
    Pad["build_chunk_arrays — pad to chunk-local max"]
    ORT["session.run()"]
    Scatter["scatter results → original slots"]
    Out["embeddings[orig_idx]"]

    Input --> Tok --> Lens --> Pack --> Chunks --> Loop
    Loop --> Pad --> ORT --> Scatter --> Out
    Scatter -->|"next chunk"| Loop
```

## Middleware Stack

The Axum router applies four `tower` layers in order (outermost first):

| Layer | Purpose |
|-------|---------|
| `SetRequestIdLayer` | Generates a UUID `x-request-id` for each request |
| `TraceLayer` | Logs request/response spans with timing via `tracing` |
| `PropagateRequestIdLayer` | Copies `x-request-id` from the request into the response |
| `DefaultBodyLimit` | Caps request bodies at 2 MiB |

## Model Lifecycle

Each worker independently manages a single `Option<(ort::session::Session, tokenizers::Tokenizer)>`.
A single ORT session produces both `sentence_embedding` (dense) and `token_embeddings`
(sparse) outputs in one forward pass. The sparse-linear projection (weight `[1024]` +
bias scalar from bundled `sparse_linear.safetensors`) is applied as a post-processing
step on the per-token hidden states.

Models transition through three states during a worker's lifetime:

1. **Loading** — ORT session and tokenizer initialized from ONNX files at startup (or reload).
2. **Loaded** — Session and tokenizer in memory, processing requests.
3. **Unloaded** — Session and tokenizer dropped after `BGE_M3_IDLE_TIMEOUT_SECS` of
   inactivity; reloaded from cache on next request.

Workers themselves never exit during idle — only their session/tokenizer pair is
dropped. The `live_workers` atomic tracks thread liveness while
`loaded_workers` tracks model presence. This distinction drives the health
endpoint's ability to differentiate `"ok"` from `"idle"` states.

## Startup Sequence (Linux)

```mermaid
sequenceDiagram
    participant Main
    participant Sysinfo as sysinfo
    participant Leader as Worker 0 (leader)
    participant Probe as probe
    participant Followers as Workers 1..N

    Main->>Sysinfo: detect_available_memory()
    Sysinfo-->>Main: MemoryReading { bytes, source }
    Main->>Leader: spawn_blocking(run_worker)
    Leader->>Leader: load_models()
    Leader-->>Main: ready signal
    Main->>Sysinfo: read_process_rss_bytes() (post-load)
    Main->>Probe: run_probe(pool, max_seq, rss_ceiling)
    Probe->>Leader: EmbedRequest::Probe (18 shapes)
    Leader-->>Probe: ProbeResult { rss_before, rss_after }
    Probe-->>Main: (a, b) coefficients
    Main->>Main: compute CostModel { a, b, max_workspace }
    Main->>Followers: spawn_blocking(run_worker) with CostModel
    Followers-->>Main: ready signals
    Main->>Main: dense + sparse readiness probes
    Main->>Main: state.ready = true
```

## Configuration

All configuration is read once at startup from environment variables via
`Config::from_env`. The server does not support runtime reconfiguration.

### Core variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_CACHE_DIR` | `/cache` | ONNX model cache directory |
| `BGE_M3_BIND` | `0.0.0.0:8081` | TCP bind address |
| `BGE_M3_WORKERS` | `2` | Worker thread count (min 1) |
| `BGE_M3_MAX_BATCH` | `256` | Max texts per request (min 1) |
| `BGE_M3_MAX_SEQ_LENGTH` | `8192` | Max tokenized sequence length `[1, 8192]` |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `300` | Idle unload timeout; `0` disables |
| `BGE_M3_MODEL` | `fp16` | Model variant: `fp32` = BAAI/bge-m3; `fp16` = Xenova/bge-m3; `int8` = Xenova/bge-m3 quantized |

### Auto-budget variables (Linux)

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_DISABLE_AUTO_BUDGET` | unset | Skip probe, use conservative defaults |
| `BGE_M3_AVAILABLE_MEMORY_BYTES` | detected | Override sysinfo memory detection |
| `BGE_M3_MEMORY_SAFETY_FACTOR` | `0.7` | Workspace headroom fraction `[0.1, 1.0]` |
| `BGE_M3_TOKEN_BUDGET` | unset | Legacy workspace ceiling (replaces `BGE_M3_ONNX_BATCH_SIZE`) |
| `BGE_M3_COST_MODEL_A` / `BGE_M3_COST_MODEL_B` | probe-derived | Override cost coefficients |

> **Apple Silicon:** A `.cargo/config.toml` with `rustflags = ["-C", "target-cpu=native"]`
> for `aarch64-apple-darwin` is committed in the repo. This enables M2+ instruction set
> extensions (i8mm, bf16) beyond the M1 baseline. CI and Docker builds are unaffected.
