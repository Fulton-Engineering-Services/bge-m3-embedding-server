# Long-Context Security Analysis

**Date:** 2026-05-08
**Supersedes:** `pr9-re-review-findings.md` (specifically the finding that cited `max_length=512` as a DoS mitigation)

## Background

The previous security review (PR9) documented `tokenizer max_length=512` as a DoS
mitigation: a single request bounded at 512 tokens caps per-request ONNX workspace
at roughly `1 × 512 × constant ≈ ~50 MB`. This analysis updates that position as the
default `MAX_SEQ_LENGTH` is raised from 512 to 8192.

Attention is `O(seq²)`, so going from 512 to 8192 multiplies the single-text worst-case
workspace by `(8192/512)² = 256×`. This requires a new security envelope, documented here.

---

## New Security Envelope

The protection model shifts from "cap sequence length" to "cap total request workspace
via the cost-model admission control and the bin-packer".

### Per-text bound (unchanged)

`MAX_STRING_CHARS = 32_768` (32 KiB per input string, enforced in `handler.rs`)
limits the raw character count per text. At the typical English token density of
~3-4 chars/token, a 32 KiB string tokenizes to ~8 000-10 000 tokens, which the
tokenizer silently truncates to `MAX_SEQ_LENGTH` (default 8192).

For high-entropy inputs (code, base64, minified JSON) the token density drops toward
1 char/token, so a 32 KiB string could produce ~32 000 tokens — still truncated to
`MAX_SEQ_LENGTH`. The character cap plus tokenizer truncation together bound per-text
workspace at `cost_model.chunk_cost(1, MAX_SEQ_LENGTH)`.

**Per-text workspace bound (conservative defaults at max_seq=8192):**
```
a × max_seq + b × max_seq² = 16384 × 8192 + 8 × 8192²
                            = 134 MB + 537 MB
                            ≈ 671 MB per single-text chunk
```

This is the worst-case workspace for one text at maximum length. A single worker
holds at most one `session.run()` at a time, so 671 MB is the upper bound per worker
per call. On the 28 GB Fargate task with 7 workers and 0.7 safety factor, the
auto-budget probe derives `max_workspace_per_worker ≈ 2.5 GB`, which bounds
multi-text chunk workspace well within hardware limits.

### Per-request bound

`MAX_BATCH = 256` (existing, unchanged) caps batch count. The bin-packer ensures
no single `session.run()` call processes more than `max_workspace_bytes` across
all texts in a chunk. Large batches of short texts pack into one ONNX call (safe);
large batches of long texts are split into many small ONNX calls (safe, just slower).

A request with 256 texts each at `MAX_SEQ_LENGTH = 8192` is the pathological worst
case. The bin-packer would split this into many single-text chunks (since
`chunk_cost(2, 8192) > max_workspace`). The server would run 256 sequential
`session.run(1, 8192)` calls per worker, each consuming ~671 MB transiently.

This is the correct behavior: each call is bounded, just slow. The appropriate
mitigation for this adversarial pattern is network-level rate limiting (existing),
not sequence-length capping.

### Per-server bound

The bin-packer + cost model is the admission control layer. Any request that would
require a single `session.run()` call exceeding `max_workspace_bytes` is automatically
split into multiple smaller calls. There is no path to a single catastrophic OOM.

The conservative fallback defaults ensure the server behaves safely even when the
startup probe fails to run (no RSS measurement available, probe shapes error):
- `a = 16 KiB/token` and `b = 8 B/token²` deliberately over-estimate workspace cost.
- `DEFAULT_MAX_WORKSPACE = 2 GiB` is conservative for the Fargate 28 GB task.
- A `(1, 8192)` chunk at conservative defaults costs ~671 MB, safely under 2 GiB.

---

## What Changed vs PR9

| Property | Before (PR9) | After (this release) |
|----------|-------------|---------------------|
| `MAX_SEQ_LENGTH` default | 512 | 8192 |
| Per-text ONNX workspace (worst case) | ~50 MB | ~671 MB |
| Per-chunk safety mechanism | Static count `BGE_M3_ONNX_BATCH_SIZE` | Quadratic cost model + bin-packer |
| Per-request worst-case calls | 1 `session.run(n, 512)` | N `session.run(1, 8192)` splits |
| Memory detection | None | cgroup v2/v1 → host RAM fallback |
| Budget source | Operator-set env var | Auto-probe at startup (overridable) |
| Single OOM path exists? | Possible (large batch + long seq) | No — bin-packer prevents it |

---

## Residual Risks and Mitigations

### 1. Slow request amplification
An adversary can send 256 texts of 32K chars each. The server will serve the request
correctly but will process 256 sequential ONNX inference calls per worker, each
~30-60s at max length. This delays other requests for minutes.

**Current mitigation:** request-level connection timeouts (configured at the reverse
proxy / ALB). Each ONNX call is bounded in memory; CPU starvation is the actual risk.

**Future mitigation (recommended):** add a token-aware 413 reject:
```rust
let estimated_tokens: usize = texts.iter().map(|t| t.chars().count() / 3 + 1).sum();
if estimated_tokens > MAX_REQUEST_TOKENS { return 413; }
```
This is called out as future work in the plan and is NOT blocking this release.

### 2. Xenova FP16/INT8 positional embedding limit
The Xenova/bge-m3 FP16 and INT8 ONNX exports (pinned at `XENOVA_REPO_REVISION`) may
have been exported with `max_position_embeddings=512` rather than 8192. Inference at
seq>512 with these models would either fail with an ORT error (caught and logged) or
silently produce degraded embeddings.

**Current mitigation:** the startup probe runs a `(1, max_seq)` inference on the leader
worker. If this fails, the server fails to start with a clear error message naming the
responsible env var (`BGE_M3_MAX_SEQ_LENGTH`) and suggesting `BGE_M3_MODEL=fp32`.

The equivalence integration test (`BGE_M3_EQUIVALENCE_TEST=1`) validates cosine
similarity at each sequence length against the FP32 reference; running it on Xenova
variants before production deployment is strongly recommended.

---

## Pre-existing Security Controls (unchanged)

- `MAX_STRING_CHARS = 32_768` per input text (`handler.rs`)
- `MAX_BATCH = 256` (`handler.rs`)
- Body size limit 2 MiB (`main.rs` `DefaultBodyLimit`)
- HTTP 413 on oversized body (existing Axum middleware)
- HTTP 400 on parse failure; 422 on wrong input type
- `ort` session `with_intra_threads(1)` prevents ORT internal thread explosion

All pre-existing security tests in `handler.rs` and `main.rs` are preserved.
