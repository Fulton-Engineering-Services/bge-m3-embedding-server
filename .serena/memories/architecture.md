# Architecture

## Source Layout
```
src/
  main.rs       — Entry point: build_router(), run_readiness_probe(), UuidRequestId, main()
  config.rs     — Config struct (cache_dir, bind_addr, workers, max_batch, idle_timeout); from_env()/from_lookup()
  state.rs      — AppState { pool: EmbedPool, ready: AtomicBool, max_batch, total_workers }
  embedder.rs   — EmbedPool, WorkerGuard, EmbedRequest enum, run_worker(), load_models()
  handler.rs    — HTTP handlers: dense_embeddings, sparse_embeddings, health, models; validate_input, check_ready
  models.rs     — Request/response types: DenseRequest/Response, SparseRequest/Response, ModelsResponse, TextInput
  error.rs      — AppError enum (InvalidRequest 400, ServiceUnavailable 503, InternalError 500)
```

## Worker Pool Pattern
- `EmbedPool` owns a bounded `mpsc::Sender<EmbedRequest>` and atomic counters for `live_workers` / `loaded_workers`
- Workers are spawned via `spawn_blocking`; each loads its own `TextEmbedding` + `SparseTextEmbedding` instances
- Workers share the receiver via `Arc<Mutex<Receiver>>`
- `WorkerGuard` decrements `live_workers` on Drop
- Readiness: each worker signals via a separate readiness channel; init task collects all N signals, then runs a warm-up probe before setting the `AtomicBool` ready flag

## Idle Unloading
- After `BGE_M3_IDLE_TIMEOUT_SECS` of no requests, workers drop their `Option<TextEmbedding>` / `Option<SparseTextEmbedding>`
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
| ok | 200 | All workers healthy, models loaded |

## Test Helpers (embedder.rs)
- `EmbedPool::closed_for_test()` — closed channel, no workers (tests "pool dead" paths)
- `EmbedPool::with_fixed_responses()` — returns preset dense/sparse vectors
- `EmbedPool::idle_for_test()` — live_workers=1, loaded_workers=0 (tests "idle" health state)

## Key Gotchas
- `fastembed::SparseEmbedding` does not implement `Debug` — use `.err().expect()` not `.unwrap_err()` on Results
- Config tests use `from_lookup()` closure instead of `env::set_var` to avoid process-global mutation under parallel tests
- `build_router()` is `pub(crate)` — use `tower::ServiceExt::oneshot()` for router-level tests
