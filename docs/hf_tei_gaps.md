# Hugging Face TEI — BGE-M3 Capability Gaps

## Summary

[Hugging Face `text-embeddings-inference`](https://github.com/huggingface/text-embeddings-inference)
(TEI) is the obvious "buy not build" alternative to this server: a small
Rust + Candle/cuBLASLt embedding server with Flash Attention, an
OpenAI-compatible HTTP API, and a published CUDA Docker image
(`ghcr.io/huggingface/text-embeddings-inference:cuda-1.9` as of early 2026).
On paper it covers the same problem space as `bge-m3-embedding-server`.

In practice **TEI cannot replace this server for any consumer that depends
on BGE-M3's sparse or ColBERT outputs**. This document enumerates the gaps,
links the upstream issues, and explains the architectural reason each gap
exists. It is intended to be read alongside [`bge-m3-model.md`](bge-m3-model.md),
which describes BGE-M3's three retrieval heads.

| Capability | This server | TEI 1.9.x | Gap |
|---|---|---|---|
| Dense embeddings (`/v1/embeddings`) | Yes | Yes | None |
| Sparse embeddings (`/v1/sparse-embeddings`) | Yes | **No** | `ForMaskedLM` arch requirement; BGE-M3 does not ship one |
| Combined dense+sparse single-forward (`/v1/embeddings:both`) | Yes | **No** | Sparse head missing entirely |
| ColBERT multi-vector | Not exposed (model supports it) | **No** | Requires `last_hidden_state`; TEI does not surface it |
| 8192-token context | Yes (FP32 export) | Yes | None |
| Multilingual (100+ languages) | Yes | Yes | None |
| Flash Attention on NVIDIA | Partial — CUDA EP available; Flash Attention kernel not yet (see note) | Yes | TEI kernel-level advantage remains; CUDA EP closes the basic GPU inference gap |
| Apple Silicon CoreML execution | Yes | No | This server's advantage on macOS dev |

---

## Why the sparse head is missing in TEI

### BGE-M3's sparse head is not a `ForMaskedLM`

BGE-M3 ships with three retrieval heads built on a shared XLM-RoBERTa
backbone:

1. **Dense** — `[CLS]`-pooled, normalized.
2. **Sparse** — `sparse_linear.safetensors` (a 4 KB linear projection from
   1024-dim hidden states down to a per-token scalar), followed by ReLU and
   max-pool over the sequence dimension. This produces a SPLADE-style
   importance distribution over the model's 250 002-token vocabulary.
3. **ColBERT (multi-vector)** — a separate per-token projection matrix
   (`colbert_linear.safetensors`) producing one vector per token for
   late-interaction scoring.

The sparse projection is **not** an MLM head. It is a learned linear layer
that BAAI ships alongside the model weights and applies after the encoder.

### TEI's SPLADE pooling requires `ForMaskedLM`

TEI implements SPLADE via the `Pool::Splade` pooling mode, which assumes the
model exposes a standard Hugging Face `*ForMaskedLM` head — i.e. the
encoder's hidden states feed a known MLM-shaped logits layer that TEI can
project through. BGE-M3 does not register itself as `ForMaskedLM`; loading
it with SPLADE pooling fails with:

> `Splade pooling is not supported: model is not a ForMaskedLM model`

See [discussion #796](https://github.com/huggingface/text-embeddings-inference/discussions/796)
for the exact reproduction.

### `last_hidden_state` is not exposed

The structural fix would be for TEI to expose raw `last_hidden_state` so a
client could apply the BGE-M3 sparse projection itself, or for TEI to learn
to load the BAAI projection weight directly. Neither has happened.
[Issue #141](https://github.com/huggingface/text-embeddings-inference/issues/141)
tracks this; opened 2023, ~42 reactions, still open in 2026 — three years
without a code-level resolution and no public roadmap commitment.

---

## Why the gap matters in practice

Hybrid retrieval — fusing dense semantic similarity with sparse lexical
weights, typically via Reciprocal Rank Fusion — consistently outperforms
either method alone on keyword-heavy, code-token, and out-of-distribution
queries. This is BGE-M3's headline capability and the reason most users
choose it over a pure dense model.

A TEI deployment of BGE-M3 silently degrades to **dense-only**:

- Lexical matching on rare tokens, identifiers, code symbols, and
  out-of-vocabulary terms regresses to whatever similarity falls out of the
  pure dense embedding — typically the worst case for hybrid retrieval.
- Vector stores provisioned for both dense and sparse columns (e.g.
  `pgvector`'s `halfvec` + `sparsevec`) end up carrying dead schema with no
  way to populate the sparse side.
- Any downstream RRF / weighted-score fusion logic loses one of its two
  ranking signals.

The severity of the regression scales with how lexical the workload is.
Pure-prose multilingual retrieval may barely notice; code search,
identifier lookup, and exact-phrase queries lose the most.

---

## What TEI gets right (when the gaps don't apply)

Acknowledging where TEI would be the better tool, if the gaps could be
closed:

- **Flash Attention on NVIDIA.** TEI's `cuda-flash-attn` build dispatches
  attention to dedicated CUDA kernels rather than ORT's generic graph
  execution. For long-sequence workloads (the `O(seq²)` regime that
  motivates the workspace probe in this server), Flash Attention is a
  meaningful per-token throughput advantage. This server now includes a
  CUDA EP build (`BGE_M3_EP=cuda`, `-cuda` Docker image tag) that runs
  standard ORT CUDA attention ops — which closes the basic GPU inference gap
  but does not match TEI's Flash Attention kernel-level efficiency. A
  TensorRT EP build is also available (`BGE_M3_EP=tensorrt`), which can
  compile engine plans using TRT's fused attention op, but this has not
  been benchmarked at Flash Attention parity.
- **Engineering surface area.** TEI is maintained by Hugging Face's
  inference team with significant test coverage and a stable v1 API.
- **gRPC + OpenAI-compatible REST out of the box.** This server is
  HTTP/JSON only.
- **Model coverage.** TEI handles `bge-large-en-v1.5`, `mxbai-embed-large-v1`,
  `nomic-embed-text-v1`, rerankers, and the rest of the standard catalog
  with no per-model code. This server is BGE-M3-specific.

The conclusion is therefore not "TEI is bad" — it is "TEI is excellent at
the embedding-server problem it solves, but BGE-M3's sparse + ColBERT
heads are outside that problem".

---

## Workaround paths (and why each is worse)

### A. Drop sparse entirely → run TEI

The deployment falls back to dense + a separate lexical backend (Postgres
`tsvector`/BM25, Elasticsearch, or similar) for the keyword leg. Any
schema provisioned for SPLADE-shaped sparse vectors becomes dead weight.
Retrieval-quality regression on code-token and lexical queries is the
main cost. Operationally simplest, but the largest quality hit.

### B. TEI for dense + this server for sparse

Two embedding services to operate, monitor, and autoscale. The
shared-transformer `/v1/embeddings:both` optimization is lost — every
hybrid index call now forwards through the encoder twice (once on each
service) and pays double tokenization. Network-side cost goes up because
of the extra hop. Strictly worse than a single-server deployment on
both throughput and ops surface area.

### C. Switch model

Move to a pure-dense model TEI handles natively, e.g. `bge-large-en-v1.5`,
`mxbai-embed-large-v1`, or `nomic-embed-text-v1`. This loses BGE-M3's two
distinguishing features:

- **Multilingual coverage**: BGE-M3 is trained on 100+ languages; the
  English-only alternatives regress hard on non-English content.
- **8192-token context**: most TEI-native dense models top out at 512 or
  2048 tokens.

Acceptable only if neither feature is being used in practice — a
measurable question, not a guess.

### D. Wait for upstream

Issue #141 has been open since 2023 with no public roadmap movement. Not
a near-term bet. Worth re-checking annually.

---

## When to re-evaluate

This document should be revisited if any of the following changes:

- TEI adds `last_hidden_state` exposure, or learns to load BGE-M3's
  `sparse_linear.safetensors` projection directly (track [issue #141](https://github.com/huggingface/text-embeddings-inference/issues/141)
  and [discussion #796](https://github.com/huggingface/text-embeddings-inference/discussions/796)).
- BAAI publishes a BGE-M3 successor that registers as `ForMaskedLM` so
  TEI's SPLADE pooling works out of the box.
- Retrieval-quality measurement on the actual workload concludes that the
  sparse leg is not pulling its weight, making path A viable.
- A new model (e.g. `naver/splade-v3`, future `bge-m4`) becomes a credible
  multilingual + long-context replacement that TEI handles natively.

Until one of those holds, this server remains the practical option for
anyone deploying BGE-M3 with both dense and sparse outputs in production.

---

## References

- [`huggingface/text-embeddings-inference`](https://github.com/huggingface/text-embeddings-inference) — TEI source
- [Issue #141 — "Support for BAAI/bge-m3 model"](https://github.com/huggingface/text-embeddings-inference/issues/141)
- [Discussion #796 — "Splade pooling error: model is not a ForMaskedLM model"](https://github.com/huggingface/text-embeddings-inference/discussions/796)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) — head structure and `sparse_linear.safetensors` provenance
- [`bge-m3-model.md`](bge-m3-model.md) — this repo's description of the
  three retrieval heads and how they're served
