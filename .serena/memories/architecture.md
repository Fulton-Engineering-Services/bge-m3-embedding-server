# Architecture

## Source Layout
```
src/
  main.rs       — Bootstrap, router, memory detection, probe, readiness probe, main()
  config.rs     — Config struct + CostModel override resolution; from_env()/from_lookup()
  state.rs      — AppState { pool, ready, max_batch, total_workers, max_seq_length, tuning: OnceLock<TuningInfo> }
  embedder.rs   — EmbedPool, WorkerConfig { cost_model, idle_timeout, model_variant, max_seq_length },
                  tokenize_no_pad(), build_chunk_arrays(), embed_dense(), embed_sparse(),
                  EmbedRequest { Dense, Sparse, Probe }, run_worker(), load_models()
  binpack.rs    — CostModel { a, b, max_workspace_bytes } + bin_pack() quadratic-aware greedy packer
  sysinfo.rs    — detect_available_memory() (cgroup v2/v1 → /proc/meminfo → sysctl), read_process_rss_bytes()
  probe.rs      — run_probe() startup workspace sweep, least-squares cost-model fit
  handler.rs    — HTTP handlers: dense_embeddings, sparse_embeddings, health (+ tuning), models
  models.rs     — Request/response types: DenseRequest/Response, SparseRequest/Response, TextInput
  error.rs      — AppError enum (InvalidRequest 400, ServiceUnavailable 503, InternalError 500)
  weights/      — Bundled sparse_linear.safetensors (4 KB) for sparse projection
```

## Worker Pool Pattern
- `EmbedPool` owns a bounded `mpsc::Sender<EmbedRequest>` and atomic counters for `live_workers` / `loaded_workers`
- Workers are spawned via `spawn_blocking`; each loads its own ORT session + tokenizer
- Workers share the receiver via `Arc<Mutex<Receiver>>`; natural load balancing
- `WorkerGuard` decrements `live_workers` on Drop
- Each worker uses a `WorkerConfig { cost_model: CostModel, idle_timeout, model_variant, max_seq_length }`

## Tokenize-Once + Bin-Pack Inference
- `tokenize_no_pad(tokenizer, texts)` — single pass, no padding applied
- `bin_pack(&seq_lens, &cost_model)` — groups original indices into workspace-safe chunks
  using `a × (count × seq) + b × (count × seq²) ≤ max_workspace_bytes`
- `build_chunk_arrays(encodings, indices, pad_to)` — pads each chunk to its own max seq
- Results scattered back to original indices: `embeddings[orig_idx] = result`

## Startup Sequence (Linux)
1. `sysinfo::detect_available_memory()` — cgroup v2 → v1 → `/proc/meminfo`
2. Spawn leader worker, await ready signal
3. `sysinfo::read_process_rss_bytes()` — measure model RSS delta
4. `probe::run_probe(pool, max_seq, rss_ceiling)` — sweep 18 shapes, fit (a, b) coefficients
5. Compute `CostModel { a, b, max_workspace_bytes }` and store in AppState.tuning via OnceLock
6. Spawn follower workers with the derived config
7. Dense + sparse warm-up probes, set `ready = true`

## Idle Unloading
- After `BGE_M3_IDLE_TIMEOUT_SECS` of no requests, workers drop their `Option<(Session, Tokenizer)>`
- On next request, models reload transparently (~10–30 s from cache)
- `loaded_workers` counter drives the "idle" health state
- Workers themselves never exit during idle — only model instances are dropped

## Health States
| Status | HTTP | Meaning |
|--------|------|---------|
| loading | 503 | Models still initializing |
| fail | 503 | All workers exited (fatal) |
| idle | 200 | Workers alive, models unloaded |
| warn | 200 | Some workers exited |
| ok | 200 | All workers healthy, models loaded; includes `tuning` object |

## Configuration (key env vars)
- `BGE_M3_MAX_SEQ_LENGTH` — default 8192, range [1, 8192]
- `BGE_M3_MODEL` — `fp16` (default), `fp32`, `int8`
- `BGE_M3_ONNX_BATCH_SIZE` — **deprecated**; use `BGE_M3_TOKEN_BUDGET` or auto-budget
- `BGE_M3_DISABLE_AUTO_BUDGET` — skip probe, use conservative defaults
- `BGE_M3_MEMORY_SAFETY_FACTOR` — default 0.7

## Test Helpers (embedder.rs)
- `EmbedPool::closed_for_test()` — closed channel, no workers (tests "pool dead" paths)
- `EmbedPool::with_fixed_responses()` — returns preset dense/sparse vectors; handles Probe variant
- `EmbedPool::idle_for_test()` — live_workers=1, loaded_workers=0 (tests "idle" health state)

## Key Gotchas
- `tokenize_no_pad` calls `encode_batch_fast` (not `encode_batch`) — no padding in the tokenizer config
- `build_chunk_arrays` pads with `pad_id=1` (XLM-RoBERTa `<pad>`) to chunk-local max seq, not global max
- `AppState.tuning` is `OnceLock<TuningInfo>` — written exactly once before `ready=true`, safe for concurrent reads
- Config tests use `from_lookup()` closure instead of `env::set_var` to avoid process-global mutation
- `build_router()` is `pub(crate)` — use `tower::ServiceExt::oneshot()` for router-level tests
- `EmbedRequest::Probe` is internal-only; workers handle it before `ready` is set
