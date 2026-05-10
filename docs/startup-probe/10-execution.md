# 10. Execution — Background Tasks and Lock-Free Handoff

The probe takes ${\sim}120\,\text{s}$ on a cache miss. Blocking startup on it would stall liveness probes and delay rolling-update completion, so the probe must run in the background. But running the probe in the background while real `/v1/embeddings` traffic flows would contaminate the per-shape RSS measurements: every concurrent allocation perturbs the process-wide RSS that the probe reads. The probe's solution is a *traffic gate*: spawn the probe in the background, but block real traffic at a semaphore until the probe finishes. Liveness probes pass throughout (the `/health` endpoint returns 200), the load balancer treats the task as healthy, and incoming `/v1/embeddings` requests queue at the semaphore for the probe window before flooding through.

Implementing this correctly is subtle. The semaphore permit acquired in the parent function must be moved into the spawned task; otherwise it is released when the parent returns and traffic enters immediately. Tokio provides the `OwnedSemaphorePermit` + `forget()` idiom for exactly this pattern. A complementary problem is the post-probe handoff: how do the fitted coefficients reach the workers without a restart or a lock? The answer is `ArcSwap`, a wait-free pointer swap that lets readers and writers operate without coordination. This page covers both: the traffic gate during the probe, and the coefficient handoff after.

## The concurrency gate during the probe window

Because the probe measures process-wide RSS deltas via `/proc/self/statm`, any concurrent `session.run()` call from real traffic pollutes the per-shape measurement. To prevent contamination, the probe holds all `cfg_workers` permits for the duration of the probe + readiness window, blocking incoming `/v1/embeddings*` requests at the semaphore. Requests queue rather than being rejected; `/health` still returns 200 during this window. The dense and sparse readiness checks and the `state.ready = true` flip both happen inside the probe task after the probe sweep completes.

`AppState` holds a `tokio::sync::Semaphore` (`request_permits`) that gates the three embedding handlers:

```55:58:src/handler/dense.rs
    let _permit = Arc::clone(&state.request_permits)
        .acquire_owned()
        .await
        .expect("request semaphore is never closed");
    let embeddings = state.pool.dense(texts).await?;
```

The semaphore is initialised to $\max(\texttt{cfg\_workers} - 1, 1)$ permits at startup, with one slot already reserved for the probe worker. On a cache miss, `spawn_probe_task` acquires the remaining $\texttt{cfg\_workers} - 1$ permits via `Arc<Semaphore>::acquire_many_owned`, moves the resulting `OwnedSemaphorePermit` into the spawned task closure, and releases everything via `add_permits(cfg_workers)` once the probe + readiness work completes.

### The OwnedSemaphorePermit subtlety

`tokio::spawn` returns synchronously, before the spawned task starts executing. A permit bound to a local variable in the parent function is dropped immediately at the end of that function, well before the probe begins:

```rust
// broken: permit dropped immediately after tokio::spawn returns
let _probe_lock = state.request_permits.acquire_many(...).await.ok();
tokio::spawn(async move { /* probe runs here */ });
return Ok(());  // _probe_lock drops here, permits released immediately
```

Real `/v1/embeddings*` traffic could enter the worker pool during the probe window, contaminating per-shape RSS deltas.

`acquire_many_owned` returns a `tokio::sync::OwnedSemaphorePermit` independent of the source `Semaphore` lifetime, so it can be moved into the closure and held for the full duration of the probe:

```rust
// correct: permit moved into the closure, lives until add_permits
let probe_permit = Arc::clone(&state.request_permits)
    .acquire_many_owned(...).await.ok();
tokio::spawn(async move {
    if let Some(p) = probe_permit { p.forget(); }  // we'll add_permits manually
    /* probe runs here */
    state.request_permits.add_permits(cfg_workers);  // open traffic
});
```

The closure calls `forget()` on the permit (preventing its drop handler from returning the permits) and then explicitly calls `add_permits(cfg_workers)` at the end of the task. The semaphore count goes from 0 (drained) directly to `cfg_workers` (full concurrency) in one operation, releasing both the explicitly acquired permits and the originally reserved probe slot.

Override and cache-hit paths do not acquire the extra permits — they run readiness checks inline, and the initial $\texttt{cfg\_workers} - 1 + 1$ permits never get drained.

### Sequence diagram of the probe window

```mermaid
sequenceDiagram
    participant Main as main()
    participant Pool as EmbedPool
    participant Probe as Probe task
    participant Sem as request_permits<br/>(Semaphore)
    participant H as /v1/embeddings handler

    Main->>Pool: spawn N workers
    Pool->>Pool: each worker primes ORT arena
    Pool-->>Main: model_rss_per_worker
    Main->>Sem: init(N − 1)
    Main->>Sem: acquire_many_owned(N − 1)
    Sem-->>Main: OwnedSemaphorePermit (drains permits to 0)
    Main->>Probe: tokio::spawn(closure { permit moved })
    Main-->>Main: return; / health = 200

    rect rgb(255, 240, 220)
        Note over Probe,Sem: Probe window — all permits held
        Probe->>Probe: arena warm-up
        Probe->>Probe: sweep 7 shapes
        Probe->>Probe: fit OLS, save cache
        Probe->>Probe: dense + sparse readiness
        H-)Sem: acquire (queues — no permits)
    end

    Probe->>Sem: forget(permit) + add_permits(N)
    Sem-->>H: permit granted
    H->>Pool: dense(texts)
    Pool-->>H: embeddings
```

The probe holds all permits across the entire window, so any real traffic that arrives during the sweep simply queues at the semaphore. When the probe finishes, the semaphore is fully replenished in one `add_permits(N)` call and queued requests flood through.

## Two synchronisation primitives

Two synchronisation primitives make the post-probe coefficient handoff safe.

### `Arc<ArcSwap<CostModel>>` for the coefficients

`ArcSwap` is a wait-free pointer swap. Every worker holds a clone of the same `Arc<ArcSwap<CostModel>>`, calls `.load()` to get a snapshot pointer when it needs to bin-pack, and the probe task calls `.store(Arc::new(cm))` once when it is done. Readers never block the writer; writers never block readers. The new cost model becomes visible on each worker's next `load()` — no restart, no message, no lock.

This produces three properties during a cold start:

1. **Workers start with conservative defaults.** The bin-packer over-budgets, packing slightly fewer texts per chunk than necessary. This is safe and slightly slower than optimal.
2. **The server flips `ready = true` *before* the probe finishes.** Liveness checks pass; production traffic can flow immediately at conservative pack ratios (once the semaphore is released).
3. **The probe task finishes asynchronously.** The next `chunk_cost()` call sees the fitted coefficients via the next `ArcSwap` load.

The transition is lock-free and observation-consistent: any single worker either uses old-everywhere or new-everywhere within one bin-pack call.

### Why `ArcSwap` and not `Mutex<Arc<CostModel>>`

A `Mutex<Arc<CostModel>>` would also work — readers acquire the lock, clone the inner `Arc`, release. But every read takes a mutex acquire-release roundtrip, which is roughly an order of magnitude more expensive than an `ArcSwap::load()`. With every bin-pack call doing a read, the difference adds up — and there is no need for it, because the workload has no contention scenario beyond the single probe writer.

`ArcSwap` is a Rust-idiomatic implementation of the standard "read-mostly, write-rarely" wait-free pattern: readers do an atomic pointer load (essentially free); writers do an atomic pointer store. The only cost readers pay is occasionally serving an old pointer for a few nanoseconds before the new one becomes visible — exactly the relaxation that fits the workload.

### `AtomicU8` for `probe_status`

The companion field tracks the probe lifecycle:

```27:41:src/state.rs
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    /// Probe not run — a cost-model override (`BGE_M3_DISABLE_AUTO_BUDGET`,
    /// `BGE_M3_TOKEN_BUDGET`, or explicit A/B env vars) was in effect.
    Disabled = 0,
    /// Probe is running in the background; workers are using conservative defaults.
    Running = 1,
    /// Probe completed successfully; fitted `(a, b)` are now active.
    Complete = 2,
    /// Probe failed or produced invalid coefficients; conservative defaults remain.
    Failed = 3,
    /// Probe was skipped — valid coefficients loaded from the EFS cache file.
    CacheHit = 4,
}
```

`probe_status` is exposed in `/health` so operators can distinguish "we just started up" from "the probe failed and we're stuck on conservative defaults":

| Status | Meaning |
|--------|---------|
| `disabled` | A cost-model override env var bypasses the probe entirely. |
| `running` | Probe in flight; workers using conservative defaults. |
| `complete` | Fitted $(a, b)$ are active and have been written to the cache. |
| `failed` | Probe ran but the fit was invalid (singular system, capability-check failure, or all-conservative coefficients). Conservative defaults remain in effect. |
| `cache_hit` | Probe skipped because a fingerprint-matching cache file existed. |

The `AtomicU8` ensures the status update is single-write, single-read consistent — no torn reads, no lock, no `Ordering::SeqCst` in the hot path. The `/health` reader uses `Ordering::Relaxed` and the probe task's final write uses `Ordering::Release`; `/health` is allowed to lag briefly behind the actual state, which is acceptable for a status field read by humans and load balancers.

## What if the probe never finishes

A pathological case: the probe is running, holds all permits, and then hangs forever. Real traffic queues at the semaphore and never gets through. Eventually the load balancer marks the task unhealthy and ECS replaces it.

The probe should not hang in practice — it has timeouts on individual `session.run()` calls and the OOM-protection layers of §6 skip pathologically large shapes before they are attempted. If a shape errors out, the probe logs and skips, never blocks. The whole sweep is wrapped in a Tokio timeout (configurable, defaults to 5 minutes) so even unforeseen hangs eventually trigger a fall-back-to-conservative path.

If a `running` status persists past 5 minutes:

- Check `/health` for `live_workers` — any dead workers indicate a degraded pool.
- Check container logs for `Probe: skipping shape` messages — a shape may be repeatedly hitting OOM guards.
- Check `BGE_M3_MAX_SEQ_LENGTH` — if it is set higher than the model variant supports, the dynamic $(1, \texttt{max\_seq})$ shape will hang on positional-embedding errors before timing out.

In all cases, the worst outcome is "task gets replaced after liveness timeout, restart picks up the cache or defaults" — never silent corruption.

## A successful cold start, step by step

1. **`main()` launches.** Reads config, creates the worker pool, kicks off the workers' `spawn_blocking` threads.
2. **Workers load and prime.** Each worker measures pre-load RSS, downloads the model (cache hit on warm starts), creates the ORT session, runs a tiny `session.run((1, 8))` to prime the arena, measures post-load RSS, and sends its delta to `main()`.
3. **`main()` aggregates deltas.** The median across workers becomes `model_rss_per_worker`. The workspace-budget formula computes `max_workspace_bytes` from container memory, the worker count, and the safety factor.
4. **Probe-cache check.** If the fingerprint matches, jump to step 7.
5. **`main()` drains the semaphore.** Acquires all $\texttt{cfg\_workers} - 1$ remaining permits via `acquire_many_owned`, spawning the probe task with the permit `move`d in.
6. **`main()` returns; `/health` flips to 200.** The load balancer is satisfied. Any incoming `/v1/embeddings` requests queue at the drained semaphore.
7. **Inside the probe task:** arena warm-up; sweep 7 shapes (each protected by 3 OOM-guard layers); fit OLS in normalised space; unscale; clamp; save cache; run dense + sparse readiness probes.
8. **Probe task `forget()`s the permit and `add_permits(cfg_workers)`.** Semaphore goes from 0 to N in one operation. Queued requests flood through.
9. **Probe task `cost_model.store(Arc::new(fitted))`.** All future bin-pack calls see the fitted coefficients on their next `load()`.
10. **Probe task sets `probe_status = Complete`.** `/health` now reports the full tuning info.

Cache-hit paths skip steps 5, 6, 7, 8 (no traffic gate, just an inline readiness check; cost model loaded from cache).

## The four properties this design provides

| Property | Mechanism |
|----------|-----------|
| Fast startup for warm starts | Cache hit short-circuits the probe |
| No traffic contamination during a cold-start probe | Semaphore drain via `OwnedSemaphorePermit::forget()` |
| No worker restarts when coefficients land | `ArcSwap` pointer swap |
| Operator-visible state during the lifecycle | `AtomicU8` `probe_status` exposed in `/health` |

A serial design fails on the first property. A naïve background design fails on the second. A "restart workers when probe finishes" design fails on the third. A "no status field" design fails on the fourth. Each piece of machinery is in service of one of these properties.

---

← [Previous: Cache](09-cache.md) | [↑ Series overview](../startup-probe.md) | [Next: End-to-end →](11-end-to-end.md)
