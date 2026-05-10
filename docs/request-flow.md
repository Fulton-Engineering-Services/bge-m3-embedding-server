# Request Flow

This document traces the lifecycle of embedding requests from client to
response. Both the **dense** (`/v1/embeddings`) and **sparse**
(`/v1/sparse-embeddings`) endpoints follow the same pipeline; only the
model invocation and response shape differ.

## Dense Embedding Request

```mermaid
sequenceDiagram
    participant Client
    participant Router as Axum Router
    participant Handler as dense_embeddings()
    participant State as AppState
    participant Pool as EmbedPool
    participant Channel as mpsc channel
    participant Worker as Worker Thread

    Client->>Router: POST /v1/embeddings
    Router->>Router: SetRequestIdLayer → generate UUID
    Router->>Router: TraceLayer → start span
    Router->>Router: DefaultBodyLimit → check ≤ 2 MiB
    Router->>Handler: Deserialize JSON → DenseRequest

    Handler->>State: check_ready()
    Note right of State: AtomicBool::load(ready) + live_worker_count > 0

    alt Not ready or no live workers
        State-->>Handler: Err(ServiceUnavailable)
        Handler-->>Client: 503 Service Unavailable
    end

    Handler->>Handler: validate_input(texts, max_batch)
    Note right of Handler: Check non-empty, max_batch texts, max 32768 chars

    alt Validation fails
        Handler-->>Client: 400 Bad Request
    end

    Handler->>State: acquire request_permits (Semaphore)
    Note right of State: N-1 permits during probe, raised to N on completion. Queues if all slots busy.

    Handler->>Pool: pool.dense(texts)
    Pool->>Channel: send(EmbedRequest::Dense { texts, reply_tx })
    Channel->>Worker: recv() — next idle worker picks up

    alt Models unloaded (idle timeout)
        Worker->>Worker: load_models() from cache
        Note right of Worker: ~10–30 s reload
    end

    Worker->>Worker: tokenize_no_pad(all texts)
    Worker->>Worker: bin_pack(seq_lens, cost_model)
    loop For each chunk
        Worker->>Worker: build_chunk_arrays — pad to chunk-local max
        Worker->>Worker: session.run(input_ids, attention_mask)
        Worker->>Worker: scatter results → original slots
    end
    Worker-->>Pool: reply_tx.send(Ok(embeddings))
    Pool-->>Handler: await reply_rx

    Handler->>Handler: Build DenseResponse (OpenAI-compatible format)
    Handler-->>Client: 200 OK + JSON response
```

## Sparse Embedding Request

The sparse path is structurally identical. The differences are:

- Sends `EmbedRequest::Sparse` through the channel.
- Worker invokes `embed_sparse()` instead of `embed_dense()` — both use
  the same single ORT session; only the output tensor and post-processing differ.
- Sparse post-processing: per-token hidden states → sparse-linear projection → ReLU → max-pool by token ID.
- Response body uses `SparseResponse` with `indices` + `values` per
  embedding instead of a flat `f32` vector.

```mermaid
sequenceDiagram
    participant Client
    participant Handler as sparse_embeddings()
    participant Pool as EmbedPool
    participant Worker as Worker Thread

    Client->>Handler: POST /v1/sparse-embeddings

    Handler->>Handler: check_ready() + validate_input()

    Handler->>Pool: pool.sparse(texts)
    Pool->>Worker: EmbedRequest::Sparse { texts, reply_tx }

    Worker->>Worker: tokenize_no_pad(all texts)
    Worker->>Worker: bin_pack(seq_lens, cost_model)
    loop For each chunk
        Worker->>Worker: session.run()
        Worker->>Worker: sparse_project → ReLU → max-pool per token ID
        Worker->>Worker: scatter → original slots
    end
    Worker-->>Handler: Vec~SparseEmbedding~

    Handler->>Handler: Map indices (usize to u32) / Build SparseResponse
    Handler-->>Client: 200 OK + JSON
```

## Request Validation Pipeline

Every embedding request passes through a multi-layer validation pipeline
before reaching the worker pool. Each layer catches a distinct class of
invalid input at the earliest possible point.

```mermaid
graph TD
    Req["Incoming Request"]
    Body["Body Limit<br/>≤ 2 MiB"]
    Deser["JSON Deserialization<br/>(Axum extractor)"]
    Ready["check_ready()<br/>ready flag + live workers"]
    Validate["validate_input()<br/>batch size + char limits"]
    Permit["acquire_owned()<br/>request_permits Semaphore"]
    Pool["EmbedPool dispatch"]

    Req --> Body
    Body -->|"Exceeds"| R413["413 Payload Too Large"]
    Body -->|"OK"| Deser
    Deser -->|"Parse error"| R400a["400/422 Bad Request"]
    Deser -->|"OK"| Ready
    Ready -->|"Not ready"| R503["503 Service Unavailable"]
    Ready -->|"OK"| Validate
    Validate -->|"Invalid"| R400b["400 Bad Request"]
    Validate -->|"OK"| Permit
    Permit -->|"Queues if N-1 slots busy during probe"| Pool
    Pool --> Inference["Worker Inference<br/>(tokenize-once + bin-pack)"]
```

### Validation constraints

| Check | Scope | Limit | Error |
|-------|-------|-------|-------|
| Body size | HTTP layer | 2 MiB | 413 Payload Too Large |
| JSON parse | Axum extractor | Valid JSON + expected shape | 400/422 |
| Readiness | `check_ready()` | `ready == true` AND `live_workers > 0` | 503 |
| Batch size | `validate_input()` | `1..=BGE_M3_MAX_BATCH` | 400 |
| String length | `validate_input()` | `≤ 32768` chars per text | 400 |
| Concurrency | `request_permits` Semaphore | `max(N-1, 1)` in-flight during probe; `N` after probe completes | queued (not rejected) |

Note: `BGE_M3_MAX_SEQ_LENGTH` (default 8192 tokens) caps how many tokens
are actually embedded — the tokenizer silently truncates any text that
tokenizes to more than that many tokens.

## Bin-Pack Chunk Dispatch

Within a worker, texts are not chunked into a static `batch_size`. Instead,
the `bin_pack()` function partitions them into workspace-safe groups using
the quadratic cost model:

```
workspace_per_chunk ≈ a × (count × max_seq) + b × (count × max_seq²)
```

A chunk closes when adding the next text would exceed `max_workspace_bytes`.
Each chunk is then padded only to its own longest sequence and passed to
`session.run()`. Results are scattered back to the original input positions.

This means:
- A batch of 256 short texts (e.g., 60 tokens each) may fit in a single `session.run()`.
- A batch of 256 long texts (e.g., 8192 tokens each) becomes 256 single-text calls.
- Mixed batches pack short texts together while long texts get their own chunks.

## Channel Dispatch Detail

The dispatch between the handler and worker pool uses a bounded `mpsc`
channel paired with `oneshot` reply channels. This design decouples the
async HTTP handler from the blocking ONNX inference.

```mermaid
graph LR
    subgraph "Async (Tokio runtime)"
        H["Handler"]
        TX["mpsc::Sender"]
        RX_one["oneshot::Receiver"]
    end

    subgraph "Blocking (spawn_blocking)"
        MutexRX["Arc&lt;Mutex&lt;mpsc::Receiver&gt;&gt;"]
        W["Worker"]
        TX_one["oneshot::Sender"]
    end

    H -->|"1. Create oneshot pair"| TX_one
    H -->|"2. Send EmbedRequest<br/>(texts + reply_tx)"| TX
    TX --> MutexRX
    MutexRX -->|"3. lock() → recv()"| W
    W -->|"4. tokenize-once + bin-pack + ONNX"| W
    W -->|"5. reply_tx.send(result)"| TX_one
    TX_one --> RX_one
    RX_one -->|"6. await result"| H
```

The bounded channel (capacity `N × 4` where `N` is the worker count) provides
natural backpressure. If all workers are busy and the channel fills, callers
block on `send()` until a worker drains a message — preventing unbounded
memory growth under load.

## Error Response Format

All error responses follow a consistent JSON structure:

```json
{
  "error": {
    "message": "descriptive error message",
    "type": "invalid_request_error"
  }
}
```

| HTTP Status | `type` | Triggered by |
|-------------|--------|-------------|
| 400 | `invalid_request_error` | Empty input, oversized batch, string too long |
| 500 | `internal_error` | Worker crash, ONNX inference failure |
| 503 | `service_unavailable` | Models loading or all workers dead |
