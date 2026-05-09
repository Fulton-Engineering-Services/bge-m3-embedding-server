# Performance

This document covers MLAS vs CoreML benchmark results, memory footprint analysis, and embedding quality for the BGE-M3 embedding server on Apple Silicon.

> **BGE_M3_ONNX_BATCH_SIZE note:** The static batch-size knob described in sections below is now deprecated. The server automatically derives a safe workspace budget via the startup probe on Linux (see [architecture.md](architecture.md)). The historical analysis below remains accurate for understanding why the static default was `8` on macOS and informs the conservative fallback constants in the quadratic cost model.

## Overview

**Machine:** MacBook Pro M3 Max (16P+4E cores, 128 GB unified memory), macOS Tahoe

**ORT:** Custom build from Fulton Engineering Services fork commit `1e37c3583`, compiled with `-mcpu=native` (via `.cargo/config.toml`). See [coreml-ep.md](coreml-ep.md) for the full build procedure.

**Benchmark tool:** Criterion, 20 samples per benchmark, median with 95% CI.

**Key finding:** CoreML delivers **20–61% lower single-text latency** vs the MLAS NEON baseline. Batch operations at `onnx_batch_size=8` are **76–319% slower** than MLAS due to serial chunked `session.run()` calls, each incurring CoreML scheduler → Metal submit → completion fence overhead.

---

## Benchmark Corpus

Curated from three production databases via a `db-backup` sidecar. Stored at `benches/fixtures/corpus.json`.

| Scenario | Source | Count | Char range | Description |
|----------|--------|-------|------------|-------------|
| `document_chunks` | `knowledgebase.chunks` | 50 | 337–1,599 | Stratified sample: 10 short, 20 medium, 20 long. Hamlet PDF + Spring AI docs. |
| `tool_descriptions` | `coordinator.vector_store` | 75 | 33–283 | Complete set. Tool/capability descriptions for semantic memory retrieval. |
| `code_symbols` | `codekeeper.symbols` | 50 | 22–120 | Random sample from 185K symbols. Class/method/field name_paths. |

**Database inventory:**

| Database | Host container | Relevant tables | Row count | Notes |
|----------|---------------|-----------------|-----------|-------|
| `knowledgebase` | `coordinator-db` | `chunks`, `documents` | 386 chunks / 5 docs | `halfvec(1024)` dense + `sparsevec(250002)` sparse stored alongside content |
| `coordinator` | `coordinator-db` | `vector_store`, `captures` | 75 vectors / 0 captures | Tool descriptions with `vector(1024)` embeddings |
| `codekeeper` | `codekeeper-db` | `symbols`, `symbol_embeddings` | 185K symbols / 0 embeddings | Embeddings not yet generated; symbols have name_path + signature |
| `langfuse` | `langfuse-db` | `observations`, `traces` | 0 / 0 | Not yet wired for tracing |

**Extraction queries (for reproducibility):** the corpus is built by sampling rows from
three Postgres databases used by downstream consumers. The exact `psql` commands are not
portable — adapt the queries below to your own environment by pointing them at any
Postgres instance with similar tables.

```sql
-- Knowledgebase chunks (stratified by length)
SELECT json_agg(row_to_json(t)) FROM (
  (SELECT content, length(content) AS char_count, 'short' AS bucket
   FROM chunks WHERE length(content) < 1000 ORDER BY random() LIMIT 10)
  UNION ALL
  (SELECT content, length(content), 'medium'
   FROM chunks WHERE length(content) BETWEEN 1000 AND 1500 ORDER BY random() LIMIT 20)
  UNION ALL
  (SELECT content, length(content), 'long'
   FROM chunks WHERE length(content) > 1500 ORDER BY random() LIMIT 20)
) t;

-- Tool descriptions (complete set)
SELECT json_agg(row_to_json(t)) FROM (
  SELECT content, length(content) AS char_count
  FROM vector_store WHERE content IS NOT NULL ORDER BY length(content)
) t;

-- Code symbols (random sample)
SELECT json_agg(row_to_json(t)) FROM (
  SELECT s.name_path AS content, length(s.name_path) AS char_count, s.kind
  FROM symbols s ORDER BY random() LIMIT 50
) t;
```

---

## Harness Design

The benchmark calls `embed_dense` and `embed_sparse` at the **direct ORT level**, bypassing the HTTP server, worker pool, and Axum routing. This isolates ONNX inference timing from JSON serialization and channel dispatch overhead.

**EP configuration via environment variable** (`BGE_M3_BENCH_EP`):

| Value | Execution providers | What it measures |
|-------|-------------------|-----------------|
| `mlas_only` | Empty vec (CPU EP, MLAS NEON) | Baseline — current production without CoreML |
| `coreml_all` | `CoreML::default()` with `ComputeUnits::All` | CoreML decides GPU vs CPU per-subgraph |
| `coreml_cpu_only` | `CoreML` with `ComputeUnits::CPUOnly` | Accelerate → AMX path (no GPU) |
| `coreml_cpu_and_gpu` | `CoreML` with `ComputeUnits::CPUAndGPU` | GPU (Metal) + CPU mix |

**Constraints:**

- Requires `ORT_LIB_LOCATION` pointing to the custom ORT build with CoreML EP
- Requires model cache at `BGE_M3_CACHE_DIR` (pre-populated; first run per EP config pays CoreML compilation cost)
- Local-only — CI runners lack the custom ORT build

---

## MLAS Baseline

`BGE_M3_BENCH_EP=mlas_only` — re-run on post-fix codebase (`--save-baseline mlas_only`).

| Scenario | Dense Single | Dense Batch | Sparse Single | Sparse Batch |
|----------|-------------|-------------|---------------|--------------|
| code_symbols (50×, 22–120 chars) | 34.8 ms | 1.31 s | 33.2 ms | 1.30 s |
| document_chunks (50×, 337–1,599 chars) | 152.6 ms | 11.9 s | 134.4 ms | 11.97 s |
| tool_descriptions (75×, 33–283 chars) | 30.5 ms | 3.27 s | 30.8 ms | 3.46 s |

Batch columns reflect all N texts submitted in a single HTTP-level request; MLAS processes them in one monolithic `session.run()` call.

---

## CoreML All Results

`BGE_M3_BENCH_EP=coreml_all`, `onnx_batch_size=8`. Percentages are vs the MLAS baseline above.

| Scenario | Dense Single | Dense Batch | Sparse Single | Sparse Batch |
|----------|-------------|-------------|---------------|--------------|
| code_symbols (50×, 22–120 chars) | 25.8 ms (**-26%**) | 5.31 s (+305%) | 26.8 ms (**-21%**) | 5.43 s (+319%) |
| document_chunks (50×, 337–1,599 chars) | 60.2 ms (**-61%**) | 20.9 s (+76%) ✓ | 65.2 ms (**-51%**) | 22.4 s (+87%) ✓ |
| tool_descriptions (75×, 33–283 chars) | 21.9 ms (**-28%**) | 7.30 s (+124%) | 28.8 ms (**-10%**) | 7.69 s (+122%) |

✓ = previously SIGKILL before the `BGE_M3_ONNX_BATCH_SIZE` fix; now completes.

**Key findings:**

| Workload type | CoreML vs MLAS | Explanation |
|---------------|----------------|-------------|
| Single-text latency | **20–61% faster** | GPU MatMul/attention dominates; CoreML dispatch overhead amortized over 192 ops per forward pass |
| Full-batch throughput (N > 8) | **76–319% slower** | 50 texts → 7 serial `session.run()` calls at `onnx_batch_size=8`; each incurs CoreML scheduler → Metal submit → completion fence overhead |

MLAS processes all N texts in one monolithic ONNX call. CoreML with `onnx_batch_size=8` uses `ceil(N/8)` serial calls. The per-call overhead (CoreML scheduler → Metal submit → fence) multiplies 7× for a 50-text batch.

---

## CoreML CPU-Only: Not Recommended

`BGE_M3_BENCH_EP=coreml_cpu_only` — routes ONNX ops through CoreML → Accelerate/AMX rather than directly to MLAS NEON kernels.

| Scenario | Dense Single | Dense Batch |
|----------|-------------|-------------|
| code_symbols (50×, 22–120 chars) | 64.6 ms (+71%) | 3.67 s (+175%) |
| document_chunks (50×, 337–1,599 chars) | 250.3 ms (+60%) | SIGKILL |

**Verdict: categorically slower.** The CoreML → Accelerate indirection adds 60–175% overhead vs MLAS's direct NEON SIMD path. CoreML's GCD-based scheduling does not saturate all cores the way MLAS's work-stealing thread pool does. The run was abandoned after the pattern was clear.

`coreml_cpu_and_gpu` produces results virtually identical to `coreml_all` — CoreML's default dispatch already excludes the ANE (dynamic shapes prevent ANE eligibility), so explicitly setting `CPUAndGPU` changes nothing.

---

## The BGE_M3_ONNX_BATCH_SIZE Fix

### Root Cause

`MLProgram` + `FastPrediction` pre-allocates the full inference workspace at the first `session.run()` call for each unique `(batch_size, seq_len)` input shape. For BGE-M3 (24 transformer layers, 16 attention heads, 1,024 hidden dim, 4,096 FFN intermediate dim) at `batch=50, seq=512`:

| Tensor | Shape | Size |
|--------|-------|------|
| Attention scores per layer | `[50, 16, 512, 512]` × float32 | 800 MB |
| FFN intermediate per layer | `[50, 512, 4,096]` × float32 | 400 MB |
| Q+K+V projections per layer | `[50, 512, 1,024]` × 3 × float32 | 300 MB |
| **Worst-case total (24 layers)** | all simultaneously pre-allocated | **~35 GB** |

With 96 GB RAM, macOS Jetsam kills the process at ~70–80% memory pressure. This is unified-memory (RAM) exhaustion, not a Metal buffer limit — `coreml_cpu_only` crashed identically.

### Fix

The `BGE_M3_ONNX_BATCH_SIZE` environment variable controls the maximum texts per `session.run()` call, independent of the HTTP-level `BGE_M3_MAX_BATCH`. It defaults to `8` on macOS and `256` elsewhere.

| `onnx_batch_size` | Attn scores per layer | Worst-case workspace | Status |
|-------------------|-----------------------|----------------------|--------|
| 50 | 800 MB | ~35 GB | SIGKILL |
| 16 | 256 MB | ~11 GB | risky |
| 8 (macOS default) | 128 MB | ~5.6 GB | **safe** |
| 4 | 64 MB | ~2.8 GB | very safe |
| 1 | 16 MB | ~0.7 GB | minimal |

With `onnx_batch_size=8`, a 50-text batch becomes 7 sequential ONNX calls (6 × 8 + 1 × 2), eliminating the workspace spike while preserving full throughput through the worker pool's request-level parallelism.

---

## onnx_batch_size=32 Probe

### Short texts

For `code_symbols` (22–120 chars, ~8–30 tokens), increasing `onnx_batch_size` from 8 to 32 significantly reduces batch latency:

| Benchmark | MLAS | batch=8 | batch=32 | Improvement over batch=8 |
|-----------|------|---------|----------|--------------------------|
| `dense/batch/code_symbols` | 1.31 s | 5.31 s | **2.17 s** | **-59%** |

The 50-text batch is handled in 2 ONNX calls (32 + 18) instead of 7. At batch=32, the result is only +66% above MLAS vs +305% at batch=8.

### Long texts

The probe on `document_chunks` (337–1,599 chars) was abandoned due to macOS memory pressure. Workspace scales quadratically with sequence length:

| Config | Attention scores per layer | Workspace |
|--------|---------------------------|-----------|
| batch=8, seq=512 | `[8, 16, 512, 512]` × f32 = 128 MB | ~5.6 GB |
| batch=32, seq=400 | `[32, 16, 400, 400]` × f32 = 3.3 GB | **~78 GB** |
| batch=32, seq=512 | `[32, 16, 512, 512]` × f32 = 512 MB | ~22 GB |

The `document_chunks` corpus texts tokenize to ~350–400 tokens, so batches of 32 are padded to that length — pushing the workspace to ~78 GB and causing severe memory pressure on 128 GB hardware. No SIGKILL, but effectively unusable (~200+ seconds per call).

### Constraint table

The safety envelope is `batch × seq_len²`, not batch alone:

| `onnx_batch_size` | seq_len=64 (code) | seq_len=256 (mixed) | seq_len=512 (long docs) |
|-------------------|-------------------|---------------------|------------------------|
| 8 | ~0.1 GB | ~1.5 GB | ~5.6 GB ✓ |
| 16 | ~0.2 GB | ~3.0 GB | ~11 GB ✓ |
| 32 | ~0.4 GB | ~6.0 GB | ~22 GB ✓* |
| 64 | ~0.8 GB | ~12 GB | ~44 GB ⚠ |

\* Safe at seq_len=512 (22 GB), but real `document_chunks` texts tokenize to ~350–400 tokens, producing ~78 GB workspace. The seq_len=512 column understates risk for this corpus.

**Production recommendation:** `onnx_batch_size=8` is the correct macOS default — safe for all text lengths. For workloads exclusively comprising short texts (code symbols, short queries), `BGE_M3_ONNX_BATCH_SIZE=32` can recover batch throughput, but it must not be the default because document-length text will hit memory pressure.

---

## Production Relevance

The batch regression (76–319% slower at `onnx_batch_size=8`) dominates the benchmark numbers but is largely irrelevant to production experience. The two consumers have distinct access patterns:

| Consumer | Operation | Texts/request | Latency-sensitive? |
|----------|-----------|---------------|-------------------|
| `dpos-coordinator` | Semantic memory lookup | 1 | **Yes** — user/agent waiting |
| `mcp-local-knowledge-base` | Search query embedding | 1 | **Yes** — user waiting |
| `mcp-local-knowledge-base` | Document chunk indexing | 10–50 | No — background task |

The interactive/online path (queries, semantic lookups) submits a single text per request. `onnx_batch_size` is irrelevant here — 1 text = 1 ONNX call regardless of the batch limit. This is where CoreML delivers **20–61% lower latency** and directly affects user-perceived performance.

The batch/indexing path (embedding document chunks during ingestion) submits 10–50 texts per request. This is where `onnx_batch_size=8` sub-batching cost manifests. However:

1. **Background operation** — no user is blocked waiting for indexing to complete.
2. **Infrequent** — occurs when new documents are added, not on every query.
3. **Worker-pool isolated** — with `BGE_M3_WORKERS=2`, one worker processes the indexing batch while the other remains available for interactive queries.

**Bottom line:** CoreML's value for this service is single-text latency (20–61% faster). The batch regression is on a non-latency-sensitive background path and is an acceptable trade-off.

---

## Memory Footprint

### MLAS Baseline — Measured

Production service on MacBook Pro M3 Max (128 GB), `BGE_M3_WORKERS=2`, `BGE_M3_IDLE_TIMEOUT_SECS=0`, MLAS-only (no CoreML EP registered). Measured with `footprint(1)` — the canonical macOS physical memory accounting tool.

| Category | Size | Contents |
|----------|------|----------|
| `MALLOC_LARGE` | 13 GB | ORT model weights + session state (4 sessions) |
| `MALLOC_SMALL` | 1.2 GB | Tokenizer data, ORT buffers, small allocations |
| `IOKit` | 12 MB | Device I/O framework overhead |
| `graphics` | 2 MB | Metal framework init (linked, idle) |
| `neural` | 6.4 MB (peak 14 MB) | CoreML framework init (linked, idle) |
| **Total footprint** | **14 GB** | |

**Weight accounting:** The BGE-M3 ONNX model is 2.16 GB on disk. ORT loads weights independently per session:

```
2 workers × 2 sessions/worker (dense + sparse) × 2.16 GB = 8.6 GB model weights
+ ORT overhead (graph, execution plan, intermediate buffers) ≈ 5.4 GB
= ~14 GB total
```

Note: `ps aux` RSS shows only ~40 MB for this process. macOS aggressively pages inactive memory on Apple Silicon. `footprint(1)` reflects actual physical memory the OS accounts against the process, including compressed and wired pages.

### CoreML Projected Memory Impact

With CoreML EP enabled, each ORT `InferenceSession` additionally:

1. **Compiles ONNX → CoreML `.mlmodelc` format** — creates a second copy of model weights in CoreML's internal representation (~2 GB per model). ORT keeps its ONNX-format copy for CPU-fallback ops; CoreML keeps its own for dispatched subgraphs. These are separate allocations — no shared pages despite unified memory.

2. **Pre-allocates `FastPrediction` workspace** — the full intermediate-tensor graph for each unique `(batch_size, seq_len)` input shape. At `onnx_batch_size=8`, `seq_len=512`: ~5.6 GB per model per shape.

Projected total for `BGE_M3_WORKERS=2`:

| Component | MLAS (measured) | + CoreML (projected) | Notes |
|-----------|-----------------|----------------------|-------|
| ORT model weights | 8.4 GB | 8.4 GB | unchanged — ORT still loads ONNX weights |
| ORT session overhead | 5.6 GB | 5.6 GB | graph, execution plan, buffers |
| CoreML compiled weights | — | ~8 GB | 4 sessions × ~2 GB compiled model |
| `FastPrediction` workspace | — | 3–22 GB | depends on shapes seen; peak at `[8, 512]` |
| GPU/Metal allocations | — | unknown | Metal command buffers, shader cache |
| **Total** | **14 GB** | **25–44 GB** | 2–3× increase |

On 128 GB hardware this is workable (20–34% of RAM). On a 96 GB host the upper range could cause memory pressure when running alongside other services (LLM gateway, observability, vector DBs, etc.).

### Worker Count Trade-Off

| Config | Workers | Memory | Single-text P50 | 2-concurrent P99 (est.) |
|--------|---------|--------|-----------------|-------------------------|
| MLAS | 2 | 14 GB | 30–153 ms | ~153 ms (parallel) |
| CoreML | 2 | 25–44 GB | 22–60 ms | ~60 ms (parallel) |
| CoreML | 1 | 12–22 GB | 22–60 ms | ~120 ms (queued) |

A single CoreML worker at P99 ~120 ms (two requests queued) is still faster than the current MLAS 2-worker deployment at P50 for document-length text (153 ms), while saving 12–22 GB of memory.

---

## RAM Reduction Options

Estimated savings are relative to the CoreML 2-worker projection of 25–44 GB. Options are additive and can be combined.

### Tier 1 — Configuration only (no code changes)

| # | Option | Est. Savings | Trade-off |
|---|--------|-------------|-----------|
| 1 | **`BGE_M3_WORKERS=1`** | ~12–22 GB (CoreML) | Requests queue behind a single worker. P99 ~120 ms queued still beats MLAS P50 for long texts. |
| 2 | **Shorter idle timeout** | Full model memory when idle | `BGE_M3_IDLE_TIMEOUT_SECS` already implemented. With CoreML model cache, reload ~5–10 s from compiled cache vs ~15–30 s cold. |
| 3 | **Lower `BGE_M3_MAX_SEQ_LENGTH`** | Reduces auto-budget workspace ceiling | Setting `BGE_M3_MAX_SEQ_LENGTH=512` restores historical behavior; setting `=2048` matches codekeeper `max-tokens`. The bin-packer will pack more texts per chunk at shorter lengths. |

### Tier 2 — Moderate code changes

| # | Option | Est. Savings | Trade-off |
|---|--------|-------------|-----------|
| 4 | **Drop `FastPrediction` → use `Default` specialization** | Eliminates 3–22 GB pre-allocated workspace per session | Higher per-request latency (est. 10–30% regression). Eliminates the single biggest CoreML memory consumer. |
| 5 | **Shared ORT session (dense + sparse)** | **Already resolved** — the server uses a single ORT session for both dense and sparse outputs. No action required. | — |
| 6 | **`PrepackedWeights` across workers** | Modest — CPU EP ops only | CoreML EP bypasses prepacking entirely. Savings primarily on CPU-fallback ops. Estimated 100–500 MB. |
| 7 | **Disable CPU memory arena** | Small | `CPU::with_arena_allocator(false)` trades RSS for fragmentation. Marginal benefit. |

### Tier 3 — Model variant (significant effort)

| # | Option | Est. Savings | Trade-off |
|---|--------|-------------|-----------|
| 8 | **FP16 model** | ~1.08 GB vs 2.16 GB per session (50%) | **Already the Apple Silicon production default** (set via LaunchAgent plist, `BGE_M3_MODEL=fp16`). No action required. |
| 9 | **INT8 quantized model** | ~568 MB vs 2.16 GB per session (74%) | Largest per-session savings. CoreML may not dispatch INT8 ops to ANE. Embedding quality validated — see [Embedding Quality](#embedding-quality) section below. Available from `Xenova/bge-m3`. |

### Memory projection by configuration

All estimates assume CoreML EP.

| Configuration | Sessions | Per-session weights | Workspace (FastPred) | Total (est.) |
|---------------|----------|--------------------|-----------------------|-------------|
| FP32 × 2 workers | 4 | 2.16 GB × 4 = 8.6 GB | 3–22 GB × 4 | 25–44 GB |
| FP32 × 1 worker | 2 | 2.16 GB × 2 = 4.3 GB | 3–22 GB × 2 | 12–22 GB |
| FP32 × 1 worker, no FastPrediction | 2 | 2.16 GB × 2 = 4.3 GB | ~0 | 8–10 GB |
| FP16 × 1 worker | 2 | 1.08 GB × 2 = 2.2 GB | 3–22 GB × 2 | 10–18 GB |
| FP16 × 1 worker, no FastPrediction | 2 | 1.08 GB × 2 = 2.2 GB | ~0 | 6–8 GB |
| INT8 × 1 worker, no FastPrediction | 2 | 0.54 GB × 2 = 1.1 GB | ~0 | 5–6 GB |
| Shared session + FP16 × 1 worker | 1 | 1.08 GB × 1 = 1.1 GB | 3–22 GB × 1 | 6–10 GB |

The most practical path is options 1 + 4 (1 worker, drop FastPrediction): **8–10 GB total**, beating the MLAS 2-worker baseline of 14 GB while preserving CoreML's 20–61% single-text latency advantage.

## Embedding Quality

Cosine similarity of each model variant vs the FP32 reference (`BAAI/bge-m3`), measured
on the 184-text bench corpus using MLAS (CPU only, no CoreML EP) on Apple M-series.
Run via `BGE_M3_MODEL=<variant> cargo bench --bench coreml -- quality`.

### FP16 (`Xenova/bge-m3`, `onnx/model_fp16.onnx`)

| Embedding type | n | mean | p5 | min |
|----------------|---|------|----|-----|
| Dense          | 184 | 0.999999 | 0.999999 | 0.999914 |
| Sparse         | 184 | 1.000000 | 0.999999 | 0.999975 |

FP16 is numerically indistinguishable from FP32 — all 184 texts score above 0.9999.
ORT internally dequantizes FP16 weights to FP32 for compute, so there is no meaningful
precision loss in practice.

### INT8 (`Xenova/bge-m3`, `onnx/model_int8.onnx`)

| Embedding type | n | mean | p5 | min |
|----------------|---|------|----|-----|
| Dense          | 184 | 0.976085 | 0.968588 | 0.962981 |
| Sparse         | 184 | 0.994023 | 0.983805 | 0.974505 |

INT8 uses weights-only quantization (Optimum). Dense embeddings drift more than sparse
(min ~0.963 vs ~0.975), because rounding error accumulates across 24 transformer layers
before CLS pooling. Sparse embeddings pass through one additional linear projection that
partially averages the per-token error.

**Verdict:** INT8 is acceptable for approximate nearest-neighbour search and semantic
ranking tasks where cosine similarity > 0.96 is sufficient. Avoid INT8 for applications
that require ranking precision within very small similarity margins (< 0.05 apart).

---

## FP16 Inference Performance

**Model:** `Xenova/bge-m3` (`onnx/model_fp16.onnx`), commit `4de13258303883538bd53b696b452bf8099f0858`

**Session memory:** ~1.08 GB per session (vs ~2.16 GB for FP32; 50% reduction).

### FP16 MLAS Baseline

`BGE_M3_MODEL=fp16 BGE_M3_BENCH_EP=mlas_only` — single-text only. FP16 batch via MLAS is
prohibitively slow (Cast overhead scales super-linearly with batch×seq\_len; measured at
~173 s/iter for `document_chunks` with `onnx_batch_size=8`).

| Scenario | Dense Single | Sparse Single |
|----------|-------------|---------------|
| code\_symbols (50×, 22–120 chars) | 283.75 ms (**+715%**) | 284.14 ms (**+756%**) |
| document\_chunks (50×, 337–1,599 chars) | 947.48 ms (**+521%**) | 946.42 ms (**+604%**) |
| tool\_descriptions (75×, 33–283 chars) | 233.03 ms (**+664%**) | 235.73 ms (**+665%**) |

Percentages are vs FP32 MLAS baseline. **FP16 MLAS is 6–9× slower than FP32 MLAS.**

ORT's MLAS executor does not support native FP16 GEMM. For every layer, the ORT graph engine
inserts a `Cast(float16 → float32)` node before each MatMul and a `Cast(float32 → float16)`
node after. These Cast operations dominate inference time and scale with the full compute
budget of the transformer (batch × sequence\_length × hidden\_dim), not just the weight size.

### FP16 CoreML Results

`BGE_M3_MODEL=fp16 BGE_M3_BENCH_EP=coreml_all`, single-text only. Batch not collected:
FP16 CoreML batch matches FP16 MLAS batch in latency (CoreML provides no acceleration; see below).

| Scenario | Dense Single | Sparse Single |
|----------|-------------|---------------|
| code\_symbols (50×, 22–120 chars) | 291.26 ms (**+1,028%**) | ~284 ms† |
| document\_chunks (50×, 337–1,599 chars) | 966.42 ms (**+1,506%**) | ~947 ms† |
| tool\_descriptions (75×, 33–283 chars) | 243.43 ms (**+1,011%**) | 238.47 ms (**+728%**) |

Percentages are vs FP32 CoreML baseline. †Sparse code\_symbols/document\_chunks not
independently benchmarked: tool\_descriptions sparse (238 ms) matches FP16 MLAS (236 ms),
consistent with the dense pattern.

**Key finding: CoreML provides no acceleration for FP16 ONNX graphs.**

FP16 CoreML single-text latency (~291 ms for `code_symbols`) matches FP16 MLAS (~284 ms),
not FP32 CoreML (25.8 ms). The root cause is structural: the Xenova FP16 ONNX model contains
FP16↔FP32 Cast nodes at every layer boundary. ORT's CoreML EP builds a CoreML execution plan
around these Cast nodes, but CoreML cannot fuse them into the larger MatMul/attention
subgraphs. Each Cast node executes on CPU; the transformer block never forms a single
contiguous subgraph eligible for GPU dispatch.

In contrast, the FP32 ONNX model has no Cast nodes. ORT can dispatch the entire
multi-head attention + FFN block as one CoreML subgraph, achieving 20–61% faster inference
than MLAS via M-series GPU utilization.

**Deployment guidance:**
- FP16 + CoreML EP offers no latency advantage over FP16 + MLAS (both ~8–10× slower than FP32 + CoreML)
- The sole benefit of FP16 is session memory (~1.08 GB vs ~2.16 GB per session, 50% reduction)
- For latency-critical paths, FP32 + CoreML EP is the correct choice
- FP16 is appropriate only when memory is the primary constraint and the ~8× latency regression is acceptable

---

## INT8 Inference Performance

**Model:** `Xenova/bge-m3` (`onnx/model_int8.onnx`), commit `4de13258303883538bd53b696b452bf8099f0858`

**Session memory:** ~568 MB per session (vs ~2.16 GB for FP32; 74% reduction).

### INT8 MLAS Baseline

`BGE_M3_MODEL=int8 BGE_M3_BENCH_EP=mlas_only` — single-text only.

| Scenario | Dense Single | Sparse Single |
|----------|-------------|---------------|
| code\_symbols (50×, 22–120 chars) | 42.48 ms (**+22%**) | 42.44 ms (**+28%**) |
| document\_chunks (50×, 337–1,599 chars) | 249.68 ms (**+64%**) | 251.85 ms (**+87%**) |
| tool\_descriptions (75×, 33–283 chars) | 27.88 ms (**−9%**) | 27.84 ms (**−10%**) |

Percentages are vs FP32 MLAS baseline. **INT8 MLAS is within 22–87% of FP32 MLAS** (and
faster for short texts). This is dramatically better than FP16 MLAS (+715%) because
INT8 dequantization uses simple integer arithmetic (zero\_point subtraction, scale multiply
via `DequantizeLinear`) rather than the full floating-point format conversion required by
FP16 `Cast`. Additionally, INT8 weights are 4× smaller than FP32, reducing memory bandwidth
pressure — which explains the −9% improvement on `tool_descriptions` (where texts are short
enough that weight-loading latency dominates compute).

### INT8 CoreML Results

`BGE_M3_MODEL=int8 BGE_M3_BENCH_EP=coreml_all`, single-text only. Batch not collected:
INT8 CoreML singles are slower than INT8 MLAS singles (see below); batch performance follows
the same fragmentation pattern as FP16 CoreML.

| Scenario | Dense Single | Sparse Single |
|----------|-------------|---------------|
| code\_symbols (50×, 22–120 chars) | 69.29 ms (**+169%**) | 70.86 ms (**+165%**) |
| document\_chunks (50×, 337–1,599 chars) | 355.63 ms (**+491%**) | 360.19 ms (**+453%**) |
| tool\_descriptions (75×, 33–283 chars) | 49.95 ms (**+128%**) | 50.03 ms (**+74%**) |

Percentages are vs FP32 CoreML baseline. The "Context leak detected, CoreAnalytics returned
false" messages logged during the INT8 CoreML run are benign CoreML framework diagnostics
(analytics context lifecycle), not errors.

**Key finding: INT8 CoreML is slower than INT8 MLAS.**

| Scenario | INT8 MLAS | INT8 CoreML | CoreML overhead |
|----------|-----------|-------------|----------------|
| code\_symbols dense single | 42.48 ms | 69.29 ms | +63% |
| document\_chunks dense single | 249.68 ms | 355.63 ms | +42% |
| tool\_descriptions dense single | 27.88 ms | 49.95 ms | +79% |

The Xenova INT8 model uses ONNX `DequantizeLinear` nodes to convert INT8 weights to FP32 at
runtime. ORT's CoreML EP fragments the execution graph around these nodes (same mechanism as
FP16 Cast nodes): CoreML dispatches small FP32 subgraphs between each `DequantizeLinear` op,
but the per-subgraph dispatch overhead (CoreML scheduler → Metal submit → completion fence)
exceeds the compute saved by GPU acceleration. The net result is slower than single-threaded
MLAS, which processes the `DequantizeLinear` + MatMul fused sequence on-die without dispatch
overhead.

**Deployment guidance for INT8:**
- INT8 + MLAS EP is the correct deployment choice: 74% memory savings with only 22–87%
  latency overhead (and −9% faster than FP32 for short texts)
- INT8 + CoreML EP adds 42–79% overhead over INT8 MLAS with no GPU benefit
- For latency-critical paths, FP32 + CoreML EP remains the best option

---

## Model Variant Comparison Summary

All single-text medians (lower is better). Machine: MacBook Pro M3 Max, macOS Tahoe.

| Config | Session memory | code\_symbols | document\_chunks | tool\_descriptions |
|--------|----------------|---------------|------------------|--------------------|
| FP32 + MLAS | 2.16 GB | 34.8 ms | 152.6 ms | 30.5 ms |
| **FP32 + CoreML** | **2.16 GB** | **25.8 ms** | **60.2 ms** | **21.9 ms** |
| FP16 + MLAS | 1.08 GB | 283.8 ms | 947.5 ms | 233.0 ms |
| FP16 + CoreML | 1.08 GB | 291.3 ms | 966.4 ms | 243.4 ms |
| **INT8 + MLAS** | **0.54 GB** | **42.5 ms** | **249.7 ms** | **27.9 ms** |
| INT8 + CoreML | 0.54 GB | 69.3 ms | 355.6 ms | 50.0 ms |

Dense embeddings shown. Sparse values are within 5% of dense for all configurations.

**Recommendations:**

| Constraint | Recommended config | Rationale |
|------------|-------------------|-----------|
| Best latency | FP32 + CoreML EP | 20–61% faster than MLAS; full GPU dispatch |
| Memory-constrained (≤ 2 GB/session) | INT8 + MLAS EP | 74% memory savings; near-FP32 latency |
| Memory-critical (≤ 1 GB/session) | INT8 + MLAS EP with `BGE_M3_WORKERS=1` | One session (~0.54 GB) |
| Avoid | FP16 (any EP) | 6–10× slower than FP32; no CoreML benefit |
| Avoid | INT8 + CoreML EP | 42–79% slower than INT8 + MLAS |
