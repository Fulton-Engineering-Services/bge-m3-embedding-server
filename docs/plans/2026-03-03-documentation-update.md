# Documentation Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `docs/apple-silicon.md` with four focused current-state reference documents, update the index and architecture doc to reflect the current codebase (fastembed removed, CoreML EP active, FP16 default).

**Architecture:** Extract content from the ~1,700-line journal, rewrite each section declaratively (no "Phase X" framing), and split into four topic docs. Existing docs receive targeted updates only. The journal file is deleted once all content is migrated.

**Tech Stack:** Markdown, Mermaid diagrams (GitHub-compatible), existing doc style in `docs/`.

**Design doc:** `docs/plans/2026-03-03-documentation-update-design.md`

**Key source file:** `docs/apple-silicon.md` (1,706 lines — primary source for all new content)

**Critical current-state facts (verify these are reflected correctly in every doc):**
- `fastembed` is **gone** — `Cargo.toml` has `ort = { version = "=2.0.0-rc.11", features = ["coreml", "ndarray", "std", "download-binaries", "tls-rustls"] }`, no fastembed dependency
- `src/embedder.rs` exports `download_model_files`, `load_tokenizer`, `load_session` — all direct ORT, no fastembed wrappers
- `.cargo/config.toml` exists with `target-cpu=native` for `aarch64-apple-darwin`
- CoreML EP is active by default on macOS via the `coreml` ort feature
- Production LaunchAgent (`scripts/ai.bge-m3.server.plist`) uses `BGE_M3_MODEL=fp16`, `BGE_M3_ONNX_BATCH_SIZE=8`, `BGE_M3_IDLE_TIMEOUT_SECS=0`
- FP16 Phase A evaluation: **complete and approved for production**

---

## Task 1: Create `docs/coreml-ep.md`

**Files:**
- Create: `docs/coreml-ep.md`
- Source: `docs/apple-silicon.md` lines 1–219, 368–461, 463–541, 543–659, 678–702

**Step 1: Write the document**

Create `docs/coreml-ep.md` with the following structure. Write each section completely before moving on.

```markdown
# CoreML Execution Provider

Brief intro: CoreML EP is active in production on Apple Silicon. Custom ORT build
(FES fork) required for BGE-M3 due to ENOTDIR bug in upstream ORT ≤ v1.23.2.

## Apple Silicon Compute Units

Source: apple-silicon.md lines 9–80.
Include the mermaid diagram (lines 15–33) — keep it, it's accurate.
Update the status labels: CPU/NEON = active. GPU = active via CoreML.
ANE = blocked by dynamic shapes (explain why briefly).
AMX/SME = not used (MLAS explicitly disables it on macOS).

## Dependency Chain

**DO NOT copy the old fastembed diagram.** Write a new one:

```mermaid
graph TD
    App["bge-m3-embedding-server<br/><code>Cargo.toml</code>"]
    ORT["ort = &quot;=2.0.0-rc.11&quot;<br/>features: coreml, ndarray, std"]
    SYS["ort-sys = &quot;2.0.0-rc.9&quot;<br/>build.rs downloads binary OR uses ORT_LIB_LOCATION"]
    LIB_PYKE["libonnxruntime.a<br/>ORT 1.23.2 (pyke.io prebuilt)<br/>Standard builds"]
    LIB_CUSTOM["libonnxruntime_*.a<br/>ORT 1.23.2 (FES fork)<br/>Apple Silicon CoreML builds"]

    App -->|"depends on"| ORT
    ORT -->|"depends on"| SYS
    SYS -->|"default path"| LIB_PYKE
    SYS -->|"ORT_LIB_LOCATION set"| LIB_CUSTOM
```

Note: `download-binaries` feature covers standard builds (CI, Docker, Linux).
`ORT_LIB_LOCATION` overrides to the custom build for Apple Silicon CoreML deployment.

## The ORT ENOTDIR Bug and FES Fork

Source: apple-silicon.md lines 375–412.
Include: root cause (file path vs dir path), the one-line C++ fix, fork coordinates table.
Upstream PR status note (pending as of writing).

## Building the Custom ORT

Source: apple-silicon.md lines 802–895.

Subsections:
- Prerequisites (Xcode CLT, CMake 3.31.x note, Python 3, submodules)
- Fork Setup (clone, checkout fix branch, submodule init)
- Build Command (the full python3 tools/ci_build/build.py invocation)
- CMake Workarounds (table: cmake_minimum_required issue, coreml_proto export set issue)
- Build Output (the directory tree of .a files)
- Verified Properties (ORT version, git commit, arch, CoreML EP ON, KleidiAI ON)

## Rust Build Against Custom ORT

Source: apple-silicon.md lines 435–461.

Include:
- `ORT_LIB_LOCATION` export + `cargo build --release`
- The cargo cache gotcha (stale rlib with LTO thin — must `rm target/release/deps/libort_sys-*.rlib`)

## BGE-M3 Op Coverage

Source: apple-silicon.md lines 548–623.

Include:
- Model inputs/outputs table (batch_size×seq_length, both dynamic)
- Op census table (28 unique types, 2,495 total — include the full table from lines 567–597)
- Coverage summary table (99.5% of compute ops CoreML-dispatchable, 9 CPU-fallback ops)
- The supported ops list from `op_builder_factory.cc`

## Dynamic Shape Impact and Compute Unit Strategy

Source: apple-silicon.md lines 624–658.

Include:
- Dynamic shape impact table (ANE: static required → ineligible; GPU: eligible; CPU: eligible)
- ComputeUnits strategy table (All / CPUOnly / CPUAndGPU / CPUAndNeuralEngine)
- Key insight: `CPUOnly` via CoreML is slower than MLAS (benchmarks in performance.md)
- Recommendation: use `All` (default)

## CoreML EP Configuration

Source: apple-silicon.md lines 647–702.

Include:
- The `ort::ep::CoreML` builder API table (with_compute_units, with_model_format, etc.)
- The `execution_providers()` Rust function (lines 682–700)
- Note that `with_profile_compute_plan()` is gated behind `coreml-profile` Cargo feature

## References

Port the references list from apple-silicon.md lines 533–541.
Add: FES fork URL.
```

**Step 2: Accuracy check**

Before committing, verify:
- [ ] No mention of `fastembed` anywhere in the file
- [ ] Dependency chain diagram shows `ort[coreml]` not `fastembed → ort`
- [ ] Fork commit hash is `1e37c3583d05992bc1419269f87d941e8642248c`
- [ ] Op coverage: 99.5% dispatchable, 9 CPU-fallback ops named correctly
- [ ] CoreML EP configuration section has the actual Rust code block

**Step 3: Commit**

```bash
git add docs/coreml-ep.md
git commit -m "docs: add coreml-ep.md — hardware, custom ORT build, op coverage, EP config"
```

---

## Task 2: Create `docs/performance.md`

**Files:**
- Create: `docs/performance.md`
- Source: `docs/apple-silicon.md` lines 703–1248

**Step 1: Write the document**

```markdown
# Performance

## Overview

State upfront: measurements taken on MacBook Pro M3 Max (16P+4E, 128 GB), macOS Tahoe.
Custom ORT 1.23.2 from FES fork commit `1e37c3583`, `-mcpu=native`.
Criterion, 20 samples per benchmark, median 95% CI.

## Benchmark Corpus

Source: apple-silicon.md lines 707–762 (Corpus + Extraction commands).

Include:
- Corpus table (3 scenarios: document_chunks, tool_descriptions, code_symbols)
- Database inventory table (knowledgebase, coordinator, codekeeper, langfuse)
- Extraction commands (SSH psql queries) — these are useful for reproducibility

## Harness Design

Source: apple-silicon.md lines 764–788.

Include:
- Description: tests at fastembed API level → wait, fastembed is gone.
  **Rewrite**: tests at direct ORT level, calling embed_dense/embed_sparse directly,
  bypassing HTTP server and worker pool.
- BGE_M3_BENCH_EP env var table (mlas_only, coreml_all, coreml_cpu_only, coreml_cpu_and_gpu)
- Constraints (ORT_LIB_LOCATION required, model cache required, local-only)

## MLAS Baseline

Source: apple-silicon.md lines 903–910 (updated MLAS baseline, lines 1022–1028).

Use the post-fix updated baseline (lines 1022–1028) as the canonical numbers — these
are the most recent measurements on the same codebase as everything else.

Include the results table (code_symbols, document_chunks, tool_descriptions ×
Dense Single / Dense Batch / Sparse Single / Sparse Batch).

## CoreML All Results

Source: apple-silicon.md lines 1013–1055.

Include:
- Post-fix CoreML All table (lines 1030–1037, with bold % deltas)
- The two-row key findings summary (single-text: 20–61% faster; batch: 76–319% slower)
- The explanation: MLAS processes all N texts in one call; CoreML with onnx_batch_size=8
  uses ceil(N/8) serial calls, each incurring CoreML scheduler → Metal submit → fence overhead

Do NOT include the pre-fix historical tables (lines 928–954) — they describe a fixed bug.

## CoreML CPU-only: Not Recommended

Source: apple-silicon.md lines 913–926.

Include the partial results table and the verdict paragraph.
Key point: CoreML → Accelerate indirection adds 60–175% overhead vs MLAS's direct NEON path.
GCD-based scheduling doesn't saturate cores the way MLAS's work-stealing thread pool does.

## The BGE_M3_ONNX_BATCH_SIZE Fix

Source: apple-silicon.md lines 971–1011.

Include:
- Root cause: MLProgram + FastPrediction pre-allocates full workspace at first session.run()
  call for each unique (batch_size, seq_len) shape
- Tensor size table (attention scores, FFN intermediate, Q+K+V at batch=50, seq=512 → ~35 GB)
- The fix: BGE_M3_ONNX_BATCH_SIZE env var, macOS default 8
- Safety table (onnx_batch_size 50 → SIGKILL; 8 → safe ~5.6 GB)

## onnx_batch_size=32 Probe

Source: apple-silicon.md lines 1066–1118.

Include:
- Short texts finding: batch=32 cuts 5.31 s → 2.17 s for code_symbols batch
- Long texts finding: quadratic blowup at ~400 tokens — workspace hits ~78 GB on 128 GB machine
- The constraint surface table (onnx_batch_size × seq_len → workspace estimate)
- Production recommendation: 8 is the correct macOS default; 32 viable for short-text-only workloads

## Production Relevance

Source: apple-silicon.md lines 1120–1152.

Include:
- Consumer access patterns table (coordinator: 1 text, latency-sensitive; kb-server: 1 for queries,
  10–50 for indexing)
- Explanation: batch regression is on a non-latency-sensitive background path
- Bottom line: CoreML's value is single-text latency (20–61% faster); batch regression is acceptable

## Memory Footprint

Source: apple-silicon.md lines 1154–1248.

Include:
- MLAS baseline measured table (MALLOC_LARGE, MALLOC_SMALL, IOKit, graphics, neural, total 14 GB)
- The weight accounting math (2 workers × 2 sessions × 2.1 GB = 8.4 GB weights + overhead)
- Note on ps RSS vs footprint(1) discrepancy (macOS pages inactive memory aggressively)
- CoreML projected table (ORT weights, session overhead, CoreML compiled weights, FastPrediction
  workspace, total 25–44 GB)
- Worker count trade-off table (MLAS 2w 14 GB vs CoreML 2w 25–44 GB vs CoreML 1w 12–22 GB)

## RAM Reduction Options

Source: apple-silicon.md lines 1252–1327.

Include:
- Key architectural insight (both TextEmbedding and SparseTextEmbedding load same 2.1 GB ONNX
  file → 4.2 GB pure duplication with 2 workers — now resolved since fastembed removed,
  but still present per-worker if 2 workers are used)

  **Note:** The direct ORT implementation already eliminates the dense+sparse session
  duplication that fastembed caused. Update Tier 2 option 5 accordingly.

- Tier 1 table (config-only: BGE_M3_WORKERS=1, idle timeout, onnx_batch_size)
- Tier 2 table (Drop FastPrediction, PrepackedWeights, disable CPU arena)
  - Remove "Shared ORT session (dense+sparse)" from Tier 2 — already done (fastembed removed)
- Tier 3 table (FP16 model, INT8 model — note FP16 is already production default on Apple Silicon)
- Memory projection table by configuration
```

**Step 2: Accuracy check**

Before committing, verify:
- [ ] MLAS baseline uses the post-fix numbers (lines 1022–1028), not the pre-fix numbers
- [ ] Pre-fix benchmark tables (lines 928–950) are NOT included — they describe a fixed bug
- [ ] The "Removing fastembed" content is NOT included
- [ ] RAM reduction option 5 (shared ORT session) is noted as already resolved
- [ ] Tier 3 notes that FP16 is already the Apple Silicon production default

**Step 3: Commit**

```bash
git add docs/performance.md
git commit -m "docs: add performance.md — MLAS vs CoreML benchmarks, SIGKILL fix, memory footprint"
```

---

## Task 3: Create `docs/model-variants.md`

**Files:**
- Create: `docs/model-variants.md`
- Source: `docs/apple-silicon.md` lines 1295–1327, 1331–1556

**Step 1: Write the document**

```markdown
# Model Variants

## Overview

State: FP32 is the compiled-in default; FP16 (Xenova/bge-m3) is recommended for
Apple Silicon. FP16 halves per-session memory (~1.08 GB vs ~2.16 GB) and has been
evaluated as suitable for production.

## Available Variants

Source: apple-silicon.md lines 1295–1305 (Xenova model table).

Include the full table:
| Variant | File | Size | Notes |
|---------|------|------|-------|
| FP32 | model.onnx + model.onnx_data | 2,162 MB | Default (BAAI/bge-m3) |
| FP16 | model_fp16.onnx | 1,082 MB | Recommended for Apple Silicon |
| INT4 (Q4) | model_q4.onnx | 1,190 MB | Block-quantized 4-bit |
| Q4F16 | model_q4f16.onnx | 668 MB | INT4 weights + FP16 activations |
| INT8 | model_quantized.onnx | 543 MB | Dynamic INT8 |
| UINT8 | model_uint8.onnx | 542 MB | Static unsigned INT8 |

All quantized variants from Xenova/bge-m3 on Hugging Face Hub.

## Selecting a Variant

BGE_M3_MODEL env var: `fp32` (default) or `fp16`.
Note: INT8/INT4/Q4F16 variants require direct ORT session usage;
not selectable via BGE_M3_MODEL env var in current implementation.

## FP16 Precision Evaluation

Source: apple-silicon.md lines 1331–1556.

### The "Already halfvec" Insight

Source: lines 1359–1374.
Key point: mcp-local-knowledge-base stores dense embeddings as PostgreSQL halfvec (FP16).
The current pipeline already quantizes to FP16 at storage time. FP16 model just moves
the quantization boundary inside the model rather than at the database boundary.

### Retrieval Context

Source: lines 1376–1390.
Include the consumer retrieval method table (kb-server: RRF → discards raw scores;
coordinator: hybrid merge 50/50 → score magnitude matters less than rank).
Note: threshold sensitivity at 0.5/0.2 is low risk since both query and corpus
vectors will be FP16 after a clean re-index.

### Phase A Results

Source: lines 1478–1556.

Include:
- Evaluation setup (175 texts, M3 Max, FP32 = BAAI/bge-m3, FP16 = Xenova/bge-m3 model_fp16.onnx)
- Note on FP16 ONNX output name difference (BAAI exports sentence_embedding; Xenova exports
  last_hidden_state — both contain same hidden states)
- Per-scenario results table (lines 1502–1506)
- Overall results table (lines 1510–1515) with PASS/FAIL
- Analysis paragraphs — rewrite as declarative conclusions:
  - Dense: effectively lossless (worst-case cosine 0.999914)
  - Sparse: mean Jaccard 0.997 and mean weight correlation 0.999; outliers at ReLU boundary
    with near-zero weights, negligible for rank-based retrieval
- **Verdict:** FP16 is suitable for production use

### ANE Dispatch Implications

Source: lines 1457–1468.
FP16 is ANE's native format; broader op eligibility possible; CoreML model cache smaller.

### Sparse Embedding Stability

Source: lines 1394–1406.
What to measure: activation agreement (Jaccard), weight magnitude correlation.
Note: sparse_linear.safetensors stays at its loaded precision regardless of model format.

## Memory Projections by Configuration

Source: apple-silicon.md lines 1306–1327 + performance.md content.

Include the configuration projection table (FP32×2w, FP32×1w, FP16×1w, FP16×1w no FastPred).
Cross-reference performance.md for full RAM reduction options.

## Migration Notes

The system is in early development. All persisted embeddings in both PostgreSQL databases
(knowledgebase.chunks and coordinator.vector_store) can be discarded and re-indexed.
Switching to FP16: set BGE_M3_MODEL=fp16, clear model cache, restart, re-index.
No mixed-precision migration required.
```

**Step 2: Accuracy check**

Before committing, verify:
- [ ] Phase A overall results table matches source exactly (cosine min 0.999914, Jaccard min 0.909091, etc.)
- [ ] PASS/FAIL verdicts match source (dense PASS, sparse FAIL on min — but verdict is still "suitable")
- [ ] The "already halfvec" section correctly describes mcp-local-knowledge-base storage
- [ ] No mention of the fastembed limitation for FP16 sparse (that constraint is gone)

**Step 3: Commit**

```bash
git add docs/model-variants.md
git commit -m "docs: add model-variants.md — FP16 evaluation results, quantized variants, production recommendation"
```

---

## Task 4: Create `docs/deployment.md`

**Files:**
- Create: `docs/deployment.md`
- Source: `docs/apple-silicon.md` lines 301–366, `scripts/install-bge-m3-apple.sh`, `scripts/ai.bge-m3.server.plist`

**Step 1: Write the document**

```markdown
# macOS Deployment (Apple Silicon)

## Overview

The server deploys on Apple Silicon Macs as a launchd UserAgent on port 8089.
The install script handles the full lifecycle: custom ORT build, binary compilation,
installation, and LaunchAgent bootstrap.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Xcode Command Line Tools: `xcode-select --install`
- Rust toolchain: `rustup`
- CMake (see coreml-ep.md for 3.31.x requirement)
- Python 3

## Installation

Source: README.md Apple Silicon section + install script header comment.

```bash
# From the repo root:
./scripts/install-bge-m3-apple.sh

# Or with a pre-built binary:
./scripts/install-bge-m3-apple.sh /path/to/bge-m3-apple
```

What the script does (ordered list):
1. Verifies Apple Silicon (arm64)
2. Clones and builds ONNX Runtime from the FES fork (see coreml-ep.md) — takes 15–30 min
3. Compiles bge-m3-embedding-server with ORT_LIB_LOCATION + target-cpu=native
4. Installs binary to ~/.local/bin/bge-m3-apple
5. Creates model cache directory at ~/.cache/bge-m3
6. Creates log directory at ~/Library/Logs/bge-m3-apple/
7. Installs the LaunchAgent plist (substituting __HOME__)
8. Bootstraps via launchctl bootstrap
9. Runs a health check on port 8089

Note on idempotency: safe to re-run to update the binary or plist.

## LaunchAgent Configuration

Source: scripts/ai.bge-m3.server.plist.

Include a full settings table:

| Setting | Value | Rationale |
|---------|-------|-----------|
| BGE_M3_BIND | 0.0.0.0:8089 | Port 8089 avoids conflict with llama-server (8088) |
| BGE_M3_WORKERS | 2 | Two worker threads, each with its own ORT session |
| BGE_M3_MAX_BATCH | 256 | Maximum texts per request |
| BGE_M3_IDLE_TIMEOUT_SECS | 0 | Models stay resident permanently — dedicated server |
| BGE_M3_ONNX_BATCH_SIZE | 8 | Limits FastPrediction workspace; safe for all text lengths. See performance.md |
| BGE_M3_MODEL | fp16 | Xenova/bge-m3 FP16 (~1.08 GB/session). See model-variants.md |
| BGE_M3_CACHE_DIR | ~/.cache/bge-m3 | Model cache location |
| BGE_M3_LOG_FORMAT | json | Structured logging |
| RUST_LOG | info | Log level |
| KeepAlive | true | Restart on crash |
| RunAtLoad | true | Start at login |
| ThrottleInterval | 10 s | Minimum between restart attempts |

## Service Management

Source: apple-silicon.md lines 352–366.

```bash
# Status
launchctl list ai.bge-m3.server

# Stop
launchctl bootout gui/$(id -u)/ai.bge-m3.server

# Restart
launchctl kickstart -k gui/$(id -u)/ai.bge-m3.server

# Logs
tail -f ~/Library/Logs/bge-m3-apple/stderr.log
```

## Log Locations

| Stream | Path |
|--------|------|
| stdout | ~/Library/Logs/bge-m3-apple/stdout.log |
| stderr | ~/Library/Logs/bge-m3-apple/stderr.log |

## Upgrading

Re-run `./scripts/install-bge-m3-apple.sh`. The script is idempotent — it replaces
the binary and plist in-place, then restarts the agent.
```

**Step 2: Accuracy check**

Before committing, verify:
- [ ] Port is 8089 (not 8081 — that's the Docker/default port)
- [ ] BGE_M3_MODEL=fp16 in the table (not fp32)
- [ ] BGE_M3_ONNX_BATCH_SIZE=8 explicitly called out with "see performance.md"
- [ ] Install script path is `scripts/install-bge-m3-apple.sh` (not `dpos-ha-config/ops/`)
- [ ] ORT build time note (15–30 min) is present

**Step 3: Commit**

```bash
git add docs/deployment.md
git commit -m "docs: add deployment.md — macOS launchd install, plist config, service management"
```

---

## Task 5: Update `docs/README.md`

**Files:**
- Modify: `docs/README.md`

**Step 1: Update the index table**

Current content:
```markdown
| [Apple Silicon Build Target](apple-silicon.md) | Dependency chain, MLAS NEON kernels, CoreML EP status, release profile, launchd deployment |
```

Replace with four rows:
```markdown
| [CoreML Execution Provider](coreml-ep.md) | Apple Silicon compute units, custom ORT build, ENOTDIR fix, BGE-M3 op coverage, CoreML EP configuration |
| [Performance](performance.md) | MLAS vs CoreML benchmarks, BGE_M3_ONNX_BATCH_SIZE fix, memory footprint, RAM reduction options |
| [Model Variants](model-variants.md) | FP32 vs FP16 precision evaluation, quantized model table, production recommendation |
| [macOS Deployment](deployment.md) | install-bge-m3-apple.sh, LaunchAgent configuration, service management |
```

**Step 2: Verify**

- [ ] All four new docs are linked
- [ ] apple-silicon.md link is gone
- [ ] Existing rows (Architecture, Request Flow, etc.) unchanged

**Step 3: Commit**

```bash
git add docs/README.md
git commit -m "docs: update README index — replace apple-silicon with four new doc links"
```

---

## Task 6: Update `docs/architecture.md`

**Files:**
- Modify: `docs/architecture.md`

**Step 1: Update the dependency chain diagram**

The current architecture.md does not have a fastembed diagram — it only shows the
worker pool and component layout. Check line ~46–56 (Module Layout table).

The module layout table is accurate as-is (embedder.rs handles ORT directly now).
No changes needed there.

Check the Configuration table at the bottom of the file (~lines 181–188).
The table lists BGE_M3_CACHE_DIR through BGE_M3_ONNX_BATCH_SIZE but is missing
`BGE_M3_MODEL`. Add it:

```markdown
| `BGE_M3_MODEL` | `fp32` | Model variant: `fp32` = BAAI/bge-m3 (~2.16 GB/session); `fp16` = Xenova/bge-m3 (~1.08 GB/session). FP16 recommended for Apple Silicon. |
```

Add a note after the configuration table:

```markdown
> **Apple Silicon:** A `.cargo/config.toml` with `rustflags = ["-C", "target-cpu=native"]`
> for `aarch64-apple-darwin` is committed in the repo. This enables M2+ instruction set
> extensions (i8mm, bf16) beyond the M1 baseline. CI and Docker builds are unaffected
> (they target `x86_64-unknown-linux-gnu` or `aarch64-unknown-linux-gnu`).
```

**Step 2: Verify**

- [ ] BGE_M3_MODEL row added to config table
- [ ] .cargo/config.toml note added
- [ ] No mention of fastembed anywhere in the file

**Step 3: Commit**

```bash
git add docs/architecture.md
git commit -m "docs(architecture): add BGE_M3_MODEL to config table, note .cargo/config.toml"
```

---

## Task 7: Update `README.md`

**Files:**
- Modify: `README.md`

**Step 1: Two targeted fixes**

**Fix 1 — Install script path** (currently references old dpos-ha-config path):

The README.md Apple Silicon section currently says the install script lives in
`./scripts/install-bge-m3-apple.sh` — check this is correct (it is, as of v0.10.0).
But the "What the script does" description says it "Builds ONNX Runtime from the FES fork".
This is accurate. Keep the section largely as-is.

**Fix 2 — CoreML/ANE prose** (lines ~275–277):

Current:
```
CoreML `FastPrediction` is enabled by default for lowest latency on Apple Neural Engine.
```

The ANE is NOT used (blocked by dynamic shapes). GPU dispatch is the active path.
Replace with:
```
CoreML EP dispatches the vast majority of transformer ops to the GPU (Metal), delivering
20–61% lower single-text latency compared to the MLAS NEON baseline. The Neural Engine
is not used — BGE-M3's dynamic sequence length prevents ANE eligibility. See
[docs/coreml-ep.md](docs/coreml-ep.md) for details.
```

**Fix 3 — Configuration table** (line ~198):

Current BGE_M3_ONNX_BATCH_SIZE description says "Defaults to 8 on macOS to avoid CoreML OOM kills".
This is accurate. Keep it.

Current BGE_M3_MODEL description says `fp32` loads BAAI/bge-m3 and `fp16` loads Xenova/bge-m3.
Add "FP16 recommended for Apple Silicon — see [docs/model-variants.md](docs/model-variants.md)."

**Step 2: Verify**

- [ ] ANE claim removed / corrected to GPU dispatch
- [ ] Links to coreml-ep.md and model-variants.md added
- [ ] No reference to dpos-ha-config install script path

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs(README): correct CoreML GPU vs ANE, link new Apple Silicon docs"
```

---

## Task 8: Delete `docs/apple-silicon.md`

**Prerequisite:** Tasks 1–7 must be complete. All content has been migrated.

**Step 1: Confirm content migration is complete**

Before deleting, do a quick spot-check:
- Open `docs/apple-silicon.md` and verify the four "valuable" topic areas each have a home:
  - Compute units → coreml-ep.md ✓
  - Op census table → coreml-ep.md ✓
  - Benchmark results → performance.md ✓
  - FP16 Phase A results → model-variants.md ✓
  - LaunchAgent deployment → deployment.md ✓
- Sections that were discarded (not migrated):
  - "CoreML EP — Present but Not Registered" (outdated)
  - "Future: Enabling CoreML EP (Standard Build Path)" (completed)
  - "Removing fastembed — Cost/Benefit Analysis" (completed)
  - Pre-fix benchmark tables (historical, describes fixed bug)

**Step 2: Delete**

```bash
git rm docs/apple-silicon.md
git commit -m "docs: remove apple-silicon.md — content migrated to coreml-ep, performance, model-variants, deployment"
```

---

## Task 9: Final verification pass

**Step 1: Check all internal links resolve**

```bash
# Check for broken markdown links in docs/
grep -r '\](.*\.md' docs/ | grep -v 'plans/'
```

Manually verify each link target exists. Key ones to check:
- `docs/README.md` → all four new doc links
- `docs/performance.md` → references to `coreml-ep.md` and `model-variants.md`
- `docs/deployment.md` → references to `coreml-ep.md` and `performance.md`
- `README.md` → references to `docs/coreml-ep.md` and `docs/model-variants.md`

**Step 2: Spot-check for fastembed references**

```bash
grep -r 'fastembed' docs/ README.md CLAUDE.md
```

Expected: zero results in `docs/`. CLAUDE.md is not being updated, so ignore hits there.

**Step 3: Verify mermaid diagrams render**

Check that all mermaid code blocks use GitHub-compatible syntax (no unsupported features).
Reference the pattern from existing docs (e.g., `docs/architecture.md`):
- `graph TD` / `graph LR` / `graph TB`
- `<br/>` for node label line breaks (not `\n`)
- Node IDs must not start with a digit

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "docs: fix link and diagram issues from verification pass"
```

---

## Task 10: Push branch

```bash
TOKEN=$(gh auth token --user jpfulton-fultonengineeringservices) && \
  git push "https://jpfulton-fultonengineeringservices:${TOKEN}@github.com/Fulton-Engineering-Services/bge-m3-embedding-server.git" feat/documentation-update
```

> **Note:** Active gh account is `jpfulton-lockboxai`; FES repos need `jpfulton-fultonengineeringservices`.
> Do NOT switch active account — use token in push URL directly.
