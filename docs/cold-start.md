# Cold-Start Sequence

When the server starts with an empty or stale model cache, it must download
~2 GB of BGE-M3 ONNX model files from Hugging Face Hub before it can serve
requests. This document explains the leader–follower startup pattern that
ensures reliable initialization even with multiple workers.

## The Problem

`hf-hub` (the Hugging Face file downloader) acquires **per-blob exclusive
file locks** (`flock(LOCK_EX)`) during download with a hardcoded 5-second
retry window. BGE-M3 models are large enough that a fresh download takes
minutes, far exceeding that window.

If all `N` workers start concurrently on an empty cache:

```mermaid
sequenceDiagram
    participant W0 as Worker 0
    participant W1 as Worker 1
    participant Cache as Model Cache
    participant Hub as HF Hub

    par All workers start simultaneously
        W0->>Hub: Download model blob A
        W1->>Hub: Download model blob A
    end

    Hub-->>W0: Downloading... (holds flock)
    Note right of W1: flock(LOCK_EX) fails<br/>after 5s retry window

    W1--xW1: ApiError::LockAcquisition
    Note right of W1: Worker exits before<br/>signaling readiness
```

The result: followers fail with `ApiError::LockAcquisition` and the init
task reports "Worker exited before signaling readiness."

## The Solution: Leader–Follower Ordering

Worker 0 (the "leader") is spawned and fully awaited before any followers
start. The leader's readiness signal acts as a **barrier**: once it fires,
the model cache is guaranteed warm and followers load from local disk.

```mermaid
sequenceDiagram
    participant Init as Init Task
    participant W0 as Worker 0 (Leader)
    participant W1 as Worker 1 (Follower)
    participant Wn as Worker N (Follower)
    participant Cache as Model Cache
    participant Hub as HF Hub

    rect rgb(240, 248, 255)
        Note over Init,Hub: Phase 1 — Leader populates cache
        Init->>W0: spawn_blocking(run_worker(id=0))
        W0->>Hub: Download dense model (~1 GB)
        Hub-->>W0: Complete
        W0->>Hub: Download sparse model (~1 GB)
        Hub-->>W0: Complete
        W0->>Cache: Models cached to disk
        W0->>Init: ready_tx.send(Ok(()))
        Note over Init: loaded_workers: 0 → 1
        Init->>Init: "Leader worker ready, model cache warm (1/N)"
    end

    rect rgb(240, 255, 240)
        Note over Init,Hub: Phase 2 — Followers load from warm cache
        par Spawn all followers
            Init->>W1: spawn_blocking(run_worker(id=1))
            Init->>Wn: spawn_blocking(run_worker(id=N))
        end

        W1->>Cache: Load dense model (local)
        W1->>Cache: Load sparse model (local)
        W1->>Init: ready_tx.send(Ok(()))
        Note over Init: loaded_workers: 1 → 2

        Wn->>Cache: Load dense model (local)
        Wn->>Cache: Load sparse model (local)
        Wn->>Init: ready_tx.send(Ok(()))
        Note over Init: loaded_workers: 2 → 3
    end

    Note over Init: "All N workers ready"
    Init-->>Init: Return Ok(())
```

## Full Startup Sequence

The complete startup includes configuration, pool spawn, HTTP listener
bind, and the readiness probe — all coordinated by `main()`.

```mermaid
sequenceDiagram
    participant Main as main()
    participant Config as Config::from_env()
    participant Pool as EmbedPool::spawn()
    participant Listener as TcpListener::bind()
    participant Probe as run_readiness_probe()
    participant InitTask as Init Task (spawned)
    participant Workers as Worker Threads

    Main->>Config: Read environment variables
    Config-->>Main: Config { cache_dir, bind, workers, ... }

    Main->>Pool: spawn(workers, cache_dir, idle_timeout)
    Pool-->>Main: (EmbedPool, init_handle)
    Note right of Pool: Workers begin loading in background

    Main->>Main: Build AppState { pool, ready: false, ... }
    Main->>Main: build_router(state)

    Main->>Listener: TcpListener::bind(bind_addr)
    Note right of Listener: Server is listening but ready=false

    Main->>Probe: tokio::spawn(run_readiness_probe)

    par Server accepts connections (returns 503)
        Note over Listener: GET /health → 503 "loading"
    and Workers initialize
        InitTask->>Workers: Phase 1: Leader loads models
        Workers-->>InitTask: Leader ready
        InitTask->>Workers: Phase 2: Followers load from cache
        Workers-->>InitTask: All followers ready
        InitTask-->>Probe: init_handle resolves Ok(())
    end

    Probe->>Pool: dense(["ready"]) — warm-up probe
    Pool-->>Probe: Ok(embeddings)
    Probe->>Pool: sparse(["ready"]) — warm-up probe
    Pool-->>Probe: Ok(embeddings)

    Probe->>Main: ready.store(true)
    Note over Main: "Models ready — accepting requests"
    Note over Listener: GET /health → 200 "ok"
```

## Failure Modes

### Leader fails to load

If the leader worker fails (e.g., corrupt cache, disk full), the init
task returns an error immediately and the process exits. No followers are
spawned.

```mermaid
graph TD
    Start["EmbedPool::spawn()"]
    SpawnLeader["Spawn Worker 0 (leader)"]
    LeaderResult{"Leader readiness?"}
    LeaderOk["Phase 2: spawn followers"]
    LeaderFail["Return Err(leader failed)"]
    ProcessExit["process::exit(1)"]

    Start --> SpawnLeader
    SpawnLeader --> LeaderResult
    LeaderResult -->|"Ok(())"| LeaderOk
    LeaderResult -->|"Err(e)"| LeaderFail
    LeaderFail --> ProcessExit
```

### Follower fails to load

If any follower fails, the init task returns an error and the process
exits. Partial readiness is not supported — all configured workers must
load successfully.

### Readiness probe fails

Even if all workers signal readiness, the readiness probe runs a trial
dense and sparse embedding. If either fails, the process exits. This
catches edge cases like a corrupted ONNX model that loads without error
but produces invalid output.

## Idle-Timeout Reloads vs Cold Start

The leader–follower ordering is **only relevant at initial startup** when
the cache may be empty. Idle-timeout reloads follow a different path:

| Scenario | Cache state | Download required | Lock contention risk | Ordering |
|----------|-------------|-------------------|----------------------|----------|
| Cold start | Empty | Yes (~2 GB) | High | Leader first, then followers |
| Idle reload | Warm (files on disk) | No | None | Each worker reloads independently |

After an idle timeout, workers reload from disk in ~10–30 seconds. Since
model files are already cached, `hf-hub` skips the download path entirely
and no file locks are contended. Workers reload independently as requests
arrive — there is no barrier.

## Single-Worker Mode

When `BGE_M3_WORKERS=1`, the leader–follower distinction is a no-op:

- Phase 1 spawns Worker 0 (the only worker)
- Phase 2's follower loop (`1..1`) is empty
- The readiness signal collection loop (`1..1`) is also empty
- The init task returns immediately after the leader is ready

This means single-worker mode has zero overhead from the cold-start
ordering logic.
