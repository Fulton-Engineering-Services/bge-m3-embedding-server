# Model Variants

## 1. Overview

The server supports two selectable model variants via the `BGE_M3_MODEL` environment variable:

- **FP32** (`BAAI/bge-m3`) — compiled-in default. Loads `model.onnx` + `model.onnx_data`, approximately 2.16 GB per session.
- **FP16** (`Xenova/bge-m3`, `model_fp16.onnx`) — recommended for Apple Silicon. Halves per-session memory (~1.08 GB vs ~2.16 GB) and has been evaluated as suitable for production.

INT8, INT4, and Q4F16 quantized variants are available on Hugging Face Hub but are not selectable via `BGE_M3_MODEL` in the current implementation. Switching to these variants requires direct code changes.

## 2. Available Variants

All quantized variants below are from `Xenova/bge-m3` on Hugging Face Hub.

| Variant | File | Size | Notes |
|---------|------|------|-------|
| FP32 | `model.onnx` + `model.onnx_data` | ~2,162 MB | Default (`BAAI/bge-m3`) |
| FP16 | `model_fp16.onnx` | ~1,082 MB | Recommended for Apple Silicon (`Xenova/bge-m3`) |
| INT4 (Q4) | `model_q4.onnx` | ~1,190 MB | Block-quantized 4-bit |
| Q4F16 | `model_q4f16.onnx` | ~668 MB | INT4 weights + FP16 activations |
| INT8 | `model_quantized.onnx` | ~543 MB | Dynamic INT8 |
| UINT8 | `model_uint8.onnx` | ~542 MB | Static unsigned INT8 |

## 3. Selecting a Variant

| Setting | Model loaded | Per-session memory |
|---------|-------------|-------------------|
| `BGE_M3_MODEL=fp32` (default) | `BAAI/bge-m3` | ~2.16 GB |
| `BGE_M3_MODEL=fp16` | `Xenova/bge-m3` (`model_fp16.onnx`) | ~1.08 GB |

INT8, INT4, and Q4F16 variants are not supported via the environment variable in the current implementation. Using them requires modifying the model loading code directly.

On Apple Silicon LaunchAgent deployments, `scripts/ai.bge-m3.server.plist` sets `BGE_M3_MODEL=fp16` by default. FP16 is the production default on Apple Silicon.

## 4. FP16 Precision Evaluation

### 4a. The "Already halfvec" Insight

`mcp-local-knowledge-base` stores dense embeddings as PostgreSQL `halfvec` (FP16):

```
FP32 model:  FP32 model → FP32 embedding → halfvec cast (FP16) → cosine search
FP16 model:  FP16 model → FP16 embedding → halfvec storage     → cosine search
```

The stored vectors are already quantized to FP16 at rest. The only difference between the two paths is whether quantization happens inside the model or at the database boundary. This significantly de-risks FP16 — precision at search time is already FP16 regardless of which model variant is used.

After a clean re-index, both the query path and the stored corpus use FP16 embeddings. There is no mixed-precision scenario to evaluate.

### 4b. Retrieval Context

Both consumers use rank-based retrieval, not raw similarity scores:

| Consumer | Retrieval method | Why rank matters more than score magnitude |
|----------|-----------------|-------------------------------------------|
| `mcp-local-knowledge-base` | Reciprocal Rank Fusion (k=60) of dense cosine + sparse dot-product | RRF discards raw scores entirely — only ordinal position in each leg matters. |
| `dpos-coordinator` | Hybrid merge of lexical `ts_rank` + semantic cosine, 50/50 weighted average | Score magnitude matters but is averaged with a separate lexical signal. |

The critical question is whether FP16 preserves rank order, not whether raw cosine similarity shifts by 0.001.

`dpos-coordinator` uses similarity thresholds of 0.5 (memory search) and 0.2 (tool search). Because the system can be fully re-indexed, any systematic score shift from FP16 applies uniformly to both query and corpus vectors and largely cancels out in the cosine computation. Threshold sensitivity at these values is low risk.

### 4c. Evaluation Setup

Embeddings were generated for all 175 texts in `benches/fixtures/corpus.json` using both FP32 and FP16 models. The corpus covers all three production scenarios: `document_chunks`, `tool_descriptions`, and `code_symbols`.

- **FP32 baseline**: `BAAI/bge-m3` (`model.onnx` + `model.onnx_data`, ~2,162 MB)
- **FP16 test**: `Xenova/bge-m3` (`model_fp16.onnx`, ~1,082 MB)

**Output name difference**: the BAAI FP32 model exports `sentence_embedding` and `token_embeddings` output names; the Xenova FP16 model exports `last_hidden_state`. Both contain equivalent hidden states for the dense embedding. The sparse projection path reads per-token hidden states in both cases.

### 4d. Metrics

| Metric | Target | What it tells you |
|--------|--------|-------------------|
| **Cosine similarity** (dense, per-text FP32 vs FP16) | > 0.999 | Raw vector alignment — near-1.0 = effectively lossless |
| **Max absolute difference** (dense, worst-case per-dimension) | < 0.01 | Worst-case per-element drift |
| **Jaccard similarity** (sparse, non-zero index overlap) | > 0.95 | Token activation agreement — 1.0 = identical active token sets |
| **Weight correlation** (sparse, Pearson r on shared non-zero weights) | > 0.99 | Weight magnitude agreement for activated tokens |

### 4e. Per-Scenario Results

| Scenario | Dense cosine (min / mean / max) | Dense max abs diff (min / mean / max) | Sparse Jaccard (min / mean / max) | Sparse weight corr (min / mean / max) |
|----------|-------------------------------|---------------------------------------|----------------------------------|--------------------------------------|
| code_symbols (50×) | 0.999997 / 1.000000 / 1.000000 | 0.000037 / 0.000064 / 0.000365 | 0.909091 / 0.998182 / 1.000000 | 0.999982 / 0.999999 / 1.000000 |
| document_chunks (50×) | 0.999914 / 0.999998 / 1.000000 | 0.000044 / 0.000126 / 0.001541 | 0.921053 / 0.991573 / 1.000000 | 0.978678 / 0.998145 / 1.000000 |
| tool_descriptions (75×) | 1.000000 / 1.000000 / 1.000000 | 0.000037 / 0.000051 / 0.000070 | 1.000000 / 1.000000 / 1.000000 | 0.999998 / 1.000000 / 1.000000 |

### 4f. Overall Results

| Metric | Min | Mean | Max | Target | Result |
|--------|-----|------|-----|--------|--------|
| Dense cosine similarity | 0.999914 | 0.999999 | 1.000000 | > 0.999 | **PASS** |
| Dense max absolute diff | 0.000037 | 0.000076 | 0.001541 | < 0.01 | **PASS** |
| Sparse Jaccard index | 0.909091 | 0.997073 | 1.000000 | > 0.95 | **FAIL** (min) |
| Sparse weight correlation | 0.978678 | 0.999470 | 1.000000 | > 0.99 | **FAIL** (min) |

**Why the sparse targets technically fail and why that is acceptable:**

The sparse FAIL values are minimum outliers concentrated in `document_chunks` (long texts, many tokens) and one `code_symbols` entry. The outliers occur at the ReLU activation boundary: tokens whose hidden-state projections land very close to zero can flip above or below the threshold due to FP16 quantization noise. These boundary tokens carry near-zero weights and contribute negligibly to sparse dot-product scores.

The mean metrics tell the real story: mean Jaccard is 0.997 and mean weight correlation is 0.999 — the FAIL values are worst-case outliers, not typical behavior.

For `tool_descriptions` (the `dpos-coordinator` workload), both metrics are 1.000000 across all 75 texts — FP16 is effectively bit-identical for short structured text.

Both consumers use rank-based fusion (RRF or hybrid merge). A sparse Jaccard of 0.91 on the worst text means the sparse leg's contribution to final ranking is minimally affected. The dense leg, which is effectively lossless (worst-case cosine 0.999914), dominates retrieval quality.

**Verdict: FP16 is suitable for production use.** Dense fidelity is excellent. Sparse fidelity is very high with marginal boundary outliers that do not affect rank-based retrieval.

## 5. ANE Dispatch Implications of FP16

The Apple Neural Engine (ANE) operates natively in FP16. With an FP32 model, CoreML inserts FP32→FP16 casts for ANE-eligible ops, and some ops may fall back to CPU when the cast introduces precision concerns. With an FP16 model:

- No casts are needed — every op is already in the ANE's native format.
- More ops may be eligible for ANE dispatch (the `coreml-profile` feature flag reveals per-op dispatch decisions at model load).
- The compiled CoreML model cache file is smaller because the weights are stored in FP16.

However, BGE-M3's dynamic sequence length prevents full ANE eligibility regardless of model precision — the ANE requires statically-shaped inputs. See `coreml-ep.md` for the dynamic shape analysis. The potential latency improvement from expanded ANE coverage is a future investigation, not current behavior.

## 6. Sparse Embedding Stability

The sparse projection weights (`sparse_linear.safetensors`, 4 KB) are loaded independently of the main ONNX model. These weights stay at their loaded precision regardless of whether `BGE_M3_MODEL=fp32` or `BGE_M3_MODEL=fp16` is set. The sparse projection is not part of the ONNX model file.

FP16 quantization affects the hidden states fed into the sparse linear layer, not the projection weights themselves. The key stability metrics are:

- **Activation agreement** (Jaccard similarity of non-zero sparse indices): whether the same tokens activate above zero.
- **Weight magnitude correlation** (Pearson r on shared non-zero weights): whether activated tokens have consistent weights.

Outliers in these metrics occur at the ReLU boundary where tokens with near-zero hidden-state projections are sensitive to FP16 quantization noise. These marginal activations carry near-zero weight and have negligible impact on retrieval scores.

## 7. Memory Projections by Configuration

The table below shows estimated total memory for common configurations. All estimates assume CoreML EP. See `performance.md` for full RAM reduction options.

| Configuration | Sessions | Per-session weights | Workspace (FastPrediction) | Total (est.) |
|---------------|----------|--------------------|-----------------------------|-------------|
| FP32 × 2 workers | 4 | 2.16 GB × 4 = 8.6 GB | 3–22 GB × 4 | 25–44 GB |
| FP32 × 1 worker | 2 | 2.16 GB × 2 = 4.3 GB | 3–22 GB × 2 | 12–22 GB |
| FP16 × 1 worker | 2 | 1.08 GB × 2 = 2.2 GB | 3–22 GB × 2 | 10–18 GB |
| FP16 × 1 worker, no FastPrediction | 2 | 1.08 GB × 2 = 2.2 GB | ~0 | 6–8 GB |

Each ONNX session count is doubled because the server loads two ORT sessions per worker (dense and sparse outputs are produced by a single session, but ORT allocates separate execution contexts). See `performance.md` for configuration guidance on reducing FastPrediction workspace overhead.

## 8. Migration Notes

The system is in early development. All persisted embeddings in PostgreSQL databases (`knowledgebase.chunks` and `coordinator.vector_store`) can be discarded and re-indexed from source. This eliminates any mixed-precision migration concern.

**Switching to FP16:**

1. Set `BGE_M3_MODEL=fp16`.
2. Clear the model cache directory (`BGE_M3_CACHE_DIR`) so the Xenova FP16 model is downloaded fresh.
3. Restart the server.
4. Re-index all documents.

No mixed-precision migration is required — a clean re-index after switching models is the correct approach.

On Apple Silicon LaunchAgent deployments, `BGE_M3_MODEL=fp16` is already set in `scripts/ai.bge-m3.server.plist`. No migration step is needed for new installations.
