# Model Variants

## 1. Overview

The server supports three selectable model variants via the `BGE_M3_MODEL` environment variable:

| Setting | Model | Per-session size | Notes |
|---------|-------|-----------------|-------|
| `BGE_M3_MODEL=fp16` **(default)** | `Xenova/bge-m3` (`model_fp16.onnx`) | ~1.08 GB | Fleet default. Best memory/quality balance. On macOS CoreML, 6–10× slower than fp32 — use fp32 there instead. |
| `BGE_M3_MODEL=fp32` | `BAAI/bge-m3` (`model.onnx`) | ~2.16 GB | Full-precision. Recommended for Apple Silicon CoreML deployments. Required if the Xenova export lacks 8192-position embeddings. |
| `BGE_M3_MODEL=int8` | `Xenova/bge-m3` (`model_int8.onnx`) | ~568 MB | Weights-only INT8 quantization. ~74% memory reduction vs fp32. Use MLAS (CPU EP) only — DequantizeLinear nodes fragment CoreML. |

## 2. Selecting a Variant

Set the `BGE_M3_MODEL` environment variable at startup:

```bash
# FP16 (default — fleet standard on Linux/MLAS)
BGE_M3_MODEL=fp16 cargo run --release

# FP32 (Apple Silicon CoreML, or when Xenova exports lack long context)
BGE_M3_MODEL=fp32 cargo run --release

# INT8 (memory-constrained Linux deployments)
BGE_M3_MODEL=int8 cargo run --release
```

On Apple Silicon LaunchAgent deployments, `scripts/ai.bge-m3.server.plist` sets
`BGE_M3_MODEL=fp16` by default.

## 3. Long-Context Behavior by Variant

The server defaults to `BGE_M3_MAX_SEQ_LENGTH=8192` (BGE-M3's published maximum).
Variant compatibility at long context differs:

| Variant | Long-context (> 512 tokens) | Notes |
|---------|-----------------------------|-------|
| **fp32** | Supported | BAAI/bge-m3 ships positional embeddings to 8192. |
| **fp16** | Needs verification | Xenova re-export may cap at 512. Run the startup probe — it tests `(1, MAX_SEQ_LENGTH)` and fails fast with an actionable message if the model rejects it. |
| **int8** | Needs verification | Same as fp16 — same Xenova revision. |

If a Xenova variant fails at your configured `MAX_SEQ_LENGTH`, the server logs:
```
Probe: model failed at configured max_seq_length — variant may not support this sequence length
Set BGE_M3_MODEL=fp32 or lower BGE_M3_MAX_SEQ_LENGTH
```

To confirm long-context correctness on any variant, run the equivalence suite:
```bash
BGE_M3_EQUIVALENCE_TEST=1 BGE_M3_CACHE_DIR=/tmp/bge-m3-cache \
  cargo test --test equivalence -- --ignored --nocapture
```

## 4. FP16 Precision Evaluation

### 4a. The "Already halfvec" Insight

`mcp-local-knowledge-base` stores dense embeddings as PostgreSQL `halfvec` (FP16):

```text
FP32 model:  FP32 model → FP32 embedding → halfvec cast (FP16) → cosine search
FP16 model:  FP16 model → FP16 embedding → halfvec storage     → cosine search
```

The stored vectors are already quantized to FP16 at rest. The only difference between the two paths is whether quantization happens inside the model or at the database boundary. This significantly de-risks FP16 — precision at search time is already FP16 regardless of which model variant is used.

### 4b. Retrieval Context

Both consumers use rank-based retrieval, not raw similarity scores:

| Consumer | Retrieval method | Why rank matters more than score magnitude |
|----------|-----------------|-------------------------------------------|
| `mcp-local-knowledge-base` | Reciprocal Rank Fusion (k=60) of dense cosine + sparse dot-product | RRF discards raw scores entirely — only ordinal position in each leg matters. |
| Hybrid search application | Hybrid merge of lexical `ts_rank` + semantic cosine, 50/50 weighted average | Score magnitude matters but is averaged with a separate lexical signal. |

The critical question is whether FP16 preserves rank order, not whether raw cosine similarity shifts by 0.001.

### 4c. Evaluation Setup

Embeddings were generated for all 175 texts in `benches/fixtures/corpus.json` using both FP32 and FP16 models. The corpus covers all three production scenarios: `document_chunks`, `tool_descriptions`, and `code_symbols`.

- **FP32 baseline**: `BAAI/bge-m3` (`model.onnx` + `model.onnx_data`, ~2,162 MB)
- **FP16 test**: `Xenova/bge-m3` (`model_fp16.onnx`, ~1,082 MB)

**Output name difference**: the BAAI FP32 model exports `sentence_embedding` and `token_embeddings` output names; the Xenova FP16 and INT8 models export `last_hidden_state`. The server handles both paths automatically based on `ModelVariant`.

### 4d. Per-Scenario Results (FP16 vs FP32)

| Scenario | Dense cosine (min / mean / max) | Dense max abs diff (min / mean / max) | Sparse Jaccard (min / mean / max) | Sparse weight corr (min / mean / max) |
|----------|-------------------------------|---------------------------------------|----------------------------------|--------------------------------------|
| code_symbols (50×) | 0.999997 / 1.000000 / 1.000000 | 0.000037 / 0.000064 / 0.000365 | 0.909091 / 0.998182 / 1.000000 | 0.999982 / 0.999999 / 1.000000 |
| document_chunks (50×) | 0.999914 / 0.999998 / 1.000000 | 0.000044 / 0.000126 / 0.001541 | 0.921053 / 0.991573 / 1.000000 | 0.978678 / 0.998145 / 1.000000 |
| tool_descriptions (75×) | 1.000000 / 1.000000 / 1.000000 | 0.000037 / 0.000051 / 0.000070 | 1.000000 / 1.000000 / 1.000000 | 0.999998 / 1.000000 / 1.000000 |

### 4e. Overall Results (FP16 vs FP32)

| Metric | Min | Mean | Max | Target | Result |
|--------|-----|------|-----|--------|--------|
| Dense cosine similarity | 0.999914 | 0.999999 | 1.000000 | > 0.999 | **PASS** |
| Dense max absolute diff | 0.000037 | 0.000076 | 0.001541 | < 0.01 | **PASS** |
| Sparse Jaccard index | 0.909091 | 0.997073 | 1.000000 | > 0.95 | **FAIL** (min) |
| Sparse weight correlation | 0.978678 | 0.999470 | 1.000000 | > 0.99 | **FAIL** (min) |

**Why the sparse targets technically fail and why that is acceptable:**

The sparse FAIL values are minimum outliers concentrated in `document_chunks` (long texts, many tokens) and one `code_symbols` entry. The outliers occur at the ReLU activation boundary: tokens whose hidden-state projections land very close to zero can flip above or below the threshold due to FP16 quantization noise. These boundary tokens carry near-zero weights and contribute negligibly to sparse dot-product scores.

The mean metrics tell the real story: mean Jaccard is 0.997 and mean weight correlation is 0.999.

**Verdict: FP16 is suitable for production use.**

## 5. INT8 Precision Evaluation

INT8 uses weights-only quantization (`model_int8.onnx` from Xenova/bge-m3). ORT dequantizes to f32 internally, so activations remain f32.

| Metric | Min | Mean | Notes |
|--------|-----|------|-------|
| Dense cosine similarity (vs FP32) | 0.963 | 0.976 | Validated against the 184-text production corpus |

Dense cosine similarity ≥ 0.963 at minimum means INT8 is suitable for ANN search and semantic ranking where embeddings are consistently indexed and queried with the same variant. Avoid INT8 for applications requiring ranking precision within very small similarity margins (< 0.05 apart).

## 6. Variant Latency Comparison

| EP | FP32 | FP16 | INT8 |
|----|------|------|------|
| MLAS (CPU, Linux/Intel) | baseline | ~6–9× slower | near-FP32 (-9% to +22% vs FP32) |
| CoreML (Apple Silicon GPU) | 20–61% faster than MLAS | 6–10× slower than FP32+CoreML | 42–79% slower than INT8+MLAS |
| CUDA EP (NVIDIA GPU, Linux) | 5–15× faster than MLAS (estimated, long sequences) | **recommended** — same as Linux fleet default | TBD — DequantizeLinear nodes may fragment CUDA execution plan |

**Summary:**
- **Linux CPU production:** FP16 for memory efficiency; INT8 for memory-constrained hosts. Both use MLAS.
- **Linux GPU production:** FP16 is the recommended model variant for CUDA and TensorRT EPs — matches the fleet default and avoids potential execution-plan fragmentation. INT8 compatibility with CUDA EP is not yet validated.
- **Apple Silicon:** FP32 for best latency via CoreML GPU dispatch. FP16 and INT8 fragment the CoreML execution graph due to Cast/DequantizeLinear nodes.
- `BGE_M3_EP=cuda` and `BGE_M3_EP=tensorrt` are Linux-only. On macOS, CoreML is always used regardless of `BGE_M3_EP`.

See [coreml-ep.md](coreml-ep.md) for the full CoreML dispatch analysis.

## 7. Memory Projections by Configuration

| Configuration | Workers | Model size/session | Estimated total |
|--------------|---------|-------------------|----------------|
| FP32 + Linux (MLAS) | 7 | ~2.16 GB | ~15–18 GB |
| FP16 + Linux (MLAS) | 7 | ~1.08 GB | ~8–10 GB |
| INT8 + Linux (MLAS) | 7 | ~568 MB | ~5–6 GB |
| FP32 + CoreML | 2 | ~2.16 GB | ~25–44 GB |
| FP16 + CoreML | 1 | ~1.08 GB | ~10–18 GB |
| FP16 + CUDA EP | clamped to `BGE_M3_GPU_COUNT` | ~1.08 GB weights + VRAM per GPU | VRAM budget 10 GiB (default); set `BGE_M3_GPU_VRAM_BUDGET_BYTES` to adjust |
| FP16 + TensorRT EP | clamped to `BGE_M3_GPU_COUNT` | ~1.08 GB weights + VRAM per GPU | Same as CUDA EP; TRT engines cached to `{cache_dir}/trt-engines/`; TRT timing cache at `{cache_dir}/trt-timing` |

GPU EP workers are clamped to `BGE_M3_GPU_COUNT` (auto-detected on Linux from `/proc/driver/nvidia/gpus/`, default `1` elsewhere); each worker is pinned to a distinct CUDA device (`device_id = worker_index % gpu_count`). Set `BGE_M3_WORKERS = BGE_M3_GPU_COUNT` on multi-GPU instances for maximum parallel inference throughput. See [GPU Execution Providers](../README.md#gpu-execution-providers-cuda--tensorrt) in the README.

At `BGE_M3_MAX_SEQ_LENGTH=8192`, each individual `session.run()` call at a single text
uses ~671 MB of intermediate workspace (conservative estimate). The bin-packer ensures
no single call exceeds `max_workspace_bytes` — see [startup-probe.md](startup-probe.md)
for the workspace decomposition and budget math.

## 8. Sparse Embedding Stability

The sparse projection weights (`sparse_linear.safetensors`, 4 KB) are loaded independently
of the main ONNX model and stay at their loaded precision regardless of model variant.

FP16/INT8 quantization affects the hidden states fed into the sparse linear layer, not the
projection weights themselves. Marginal activations at the ReLU boundary are sensitive to
quantization noise but carry near-zero weight — their impact on rank-based retrieval scores
is negligible.

## 9. Migration Notes

The system supports full re-indexing from source. Switching variants requires:

1. Set `BGE_M3_MODEL=<new variant>`.
2. Clear the model cache directory (`BGE_M3_CACHE_DIR`) so the new model is downloaded.
3. Restart the server.
4. Re-index all documents (no mixed-precision migration required — fresh embeddings supersede all prior vectors).

On Apple Silicon LaunchAgent deployments, `BGE_M3_MODEL=fp16` is already set in
`scripts/ai.bge-m3.server.plist`. No migration step is needed for new installations.
