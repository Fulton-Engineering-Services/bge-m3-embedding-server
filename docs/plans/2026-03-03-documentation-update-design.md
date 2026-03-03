# Documentation Update Design

**Date:** 2026-03-03
**Branch:** `feat/documentation-update`

## Problem

`docs/apple-silicon.md` is a ~1,700-line chronological research journal that accumulated
through several phases of project evolution. It contains outdated sections (fastembed
dependency chain, "Future: Enabling CoreML EP", "Removing fastembed analysis"), accurate
but scattered current-state content, and completed research data that belongs in dedicated
reference documents.

The existing docs also have stale artifacts from the fastembed removal: the dependency
chain diagram in `docs/architecture.md`, and the `docs/README.md` index entry describing
`apple-silicon.md` in terms of its original (not current) scope.

## Goal

Replace `docs/apple-silicon.md` with four focused, polished, current-state reference
documents. Update the index and architecture doc to reflect the current codebase.

## File Changes

### Created

| File | Purpose |
|------|---------|
| `docs/coreml-ep.md` | CoreML Execution Provider: hardware, ORT build, ENOTDIR fix, op coverage, dispatch |
| `docs/performance.md` | Benchmarks, memory footprint, SIGKILL fix, onnx_batch_size analysis |
| `docs/model-variants.md` | FP32 vs FP16 variants, Phase A precision results, production recommendation |
| `docs/deployment.md` | macOS LaunchAgent: install script, plist config, service management |

### Updated

| File | Changes |
|------|---------|
| `docs/README.md` | Remove apple-silicon row; add 4 new rows for new docs |
| `docs/architecture.md` | Update dependency chain diagram (remove fastembed); add `BGE_M3_MODEL` to config table |
| `README.md` | Fix install script path reference; tighten CoreML/ANE prose |

### Deleted

| File | Reason |
|------|--------|
| `docs/apple-silicon.md` | All content migrated to the four new docs above |

## New Doc Structures

### `docs/coreml-ep.md`

1. Overview (current production status: CoreML EP active)
2. Apple Silicon Compute Units (NEON active, GPU active via CoreML, ANE blocked by dynamic shapes)
3. Dependency Chain (current: `App → ort[coreml] → ort-sys → libonnxruntime.a`)
4. The ORT ENOTDIR Bug and FES Fork (root cause, fix, fork coordinates)
5. Building the Custom ORT (prerequisites, build command, CMake workarounds, output layout)
6. Rust Build Against Custom ORT (`ORT_LIB_LOCATION`, cargo cache gotcha)
7. BGE-M3 Op Coverage (op census table, 99.5% CoreML-dispatchable)
8. Dynamic Shape Impact (ANE eligibility, GPU eligible, ComputeUnits strategy table)
9. CoreML EP Configuration (`execution_providers()`, MLProgram + FastPrediction + model cache)
10. References

### `docs/performance.md`

1. Overview (machine, config)
2. Benchmark Setup (corpus, harness, EP configurations)
3. MLAS Baseline results
4. CoreML All results (post-fix, onnx_batch_size=8): single-text 20–61% faster; batch 76–319% slower
5. CoreML CPU-only verdict: never use
6. SIGKILL Root Cause and Fix (FastPrediction workspace pre-allocation, batch × seq_len² scaling)
7. onnx_batch_size=32 Probe (short texts: better; long texts: quadratic blowup)
8. Production Relevance (consumer access patterns, why single-text latency is the metric)
9. Memory Footprint (MLAS 14 GB measured; CoreML 25–44 GB projected; breakdown)
10. RAM Reduction Options (Tier 1/2/3 table; memory projection by configuration)

### `docs/model-variants.md`

1. Overview (FP32 default; FP16 recommended for Apple Silicon)
2. Available Variants (FP32, FP16, INT4, Q4F16, INT8, UINT8 — file, size, source)
3. Selecting a Variant (`BGE_M3_MODEL` env var)
4. FP16 Precision Evaluation
   - The "already halfvec" insight
   - Rank-based retrieval context
   - Phase A results (cosine similarity, max abs diff, Jaccard, weight correlation)
   - Per-scenario breakdown
   - Verdict: FP16 suitable for production
5. ANE Dispatch Implications of FP16
6. Sparse Embedding Stability
7. Migration Notes (clean re-index is correct approach)

### `docs/deployment.md`

1. Overview (launchd UserAgent, port 8089)
2. Prerequisites
3. Installation (`install-bge-m3-apple.sh`: ORT build → compile → install → LaunchAgent)
4. LaunchAgent Configuration (full settings table with rationale)
5. Service Management (status, stop, restart, logs)
6. Log Locations
7. Upgrading (idempotent re-run)

## Source Material Disposition

| `apple-silicon.md` section | Destination |
|---------------------------|-------------|
| Compute units (NEON, AMX, GPU, ANE) | `coreml-ep.md` §2 |
| Old fastembed dependency chain diagram | **Discarded** (outdated) |
| "CoreML EP — Present but Not Registered" | **Discarded** (outdated) |
| "Future: Enabling CoreML EP" | **Discarded** (completed) |
| Op census + coverage tables | `coreml-ep.md` §7 |
| Dynamic shape impact table | `coreml-ep.md` §8 |
| ComputeUnits strategy analysis | `coreml-ep.md` §8 |
| Phase 1 & 2 (CoreML EP code, `.cargo/config.toml`) | Become declarative current-state in `coreml-ep.md` §9 |
| Custom ORT build (ENOTDIR fix, build steps) | `coreml-ep.md` §4–6 |
| Phase 3 benchmark results (MLAS, CoreML variants) | `performance.md` §3–5 |
| SIGKILL root cause + fix | `performance.md` §6 |
| onnx_batch_size=32 probe | `performance.md` §7 |
| Production relevance analysis | `performance.md` §8 |
| Memory footprint analysis | `performance.md` §9 |
| RAM reduction options inventory | `performance.md` §10 |
| FP16 evaluation framework + Phase A results | `model-variants.md` §4 |
| Quantized model availability table | `model-variants.md` §2 |
| Memory projections by configuration | `model-variants.md` §2 + `performance.md` §10 |
| macOS deployment / launchd / install script | `deployment.md` §3–4 |
| Service management commands | `deployment.md` §5 |
| "Removing fastembed" analysis | **Discarded** (completed work) |

## Content Treatment

Each new doc is written as a **current-state reference**, not a journal. The Phase
Progress Log framing ("Phase 1 completed", "next steps" checklists) is dropped. Content
is rewritten declaratively: findings become facts, "Phase A verdict" becomes a section
conclusion, benchmark tables are presented as results without the pre-fix/post-fix
historical framing beyond what's needed for context.
