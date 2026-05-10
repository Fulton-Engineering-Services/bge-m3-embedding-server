# 12. Operator Quick Reference

When something looks off in production — slower than expected throughput, a startup `WARN` about utilisation, an unexplained OOM kill — the first place to look is `/health`. It exposes everything the probe knows about itself: the fitted coefficients, the budget formula's inputs, the probe lifecycle state, and the current memory headroom. Reading it correctly determines in seconds whether the probe is healthy, whether it is running on conservative defaults, whether the container is over-subscribed, or whether a model variant is incompatible with the configured `MAX_SEQ_LENGTH`.

This page is structured diagnose-first, fix-second. Each section starts with a symptom, walks through the diagnosis path, and recommends fixes. The fixes are env-var oriented because that is the only operator-facing knob — the probe has no admin API.

## Diagnosing probe state

`curl http://host:8081/health | jq '.tuning'` shows:

| Field | Typical value (fp16, 7 workers, 28 GB) | Meaning |
|-------|----------------------------------------|---------|
| `probe_status` | `complete` | Which probe path was taken: `cache_hit`, `complete`, `failed`, `running`, `disabled`. |
| `a_bytes_per_token` | ${\sim}18\,432$ | Fitted linear coefficient (FFN term). |
| `b_bytes_per_token_sq` | ${\sim}6.2$ | Fitted quadratic coefficient (attention term). |
| `max_workspace_bytes` | ${\sim}2.0\,\text{GB}$ | Per-worker bin-packing budget derived from available memory. |
| `model_rss_bytes_per_worker` | ${\sim}1\,100\,000\,000$ | Peak RSS delta measured by each worker during `load_models()`. Was 0 before the 2026-05-09 fix. |
| `worst_case_peak_bytes` | ${\sim}21.5\,\text{GB}$ | $N \times \text{workspace} + N \times \text{model\_rss} + \text{OS\_HEADROOM}$. Must be below `available_bytes`. |
| `utilization_pct` | ${\sim}74\%$ | $\text{worst\_case\_peak} / \text{available} \times 100$. A `WARN` fires at startup if $> 90\%$. |

If `model_rss_bytes_per_worker` is 0 on Linux, the worker RSS measurement failed (non-Linux platform or `/proc/self/statm` unreadable). The budget formula still runs but deducts 0 for model weights — use `BGE_M3_MEMORY_SAFETY_FACTOR=0.5` as a stopgap and file a bug.

If `worst_case_peak_bytes > available_bytes`, the container is over-subscribed. Reduce `BGE_M3_WORKERS` or `BGE_M3_MEMORY_SAFETY_FACTOR` before the OOM kill happens.

## Symptom → diagnosis matrix

| Symptom | Likely cause | First check | Fix |
|---------|--------------|-------------|-----|
| `probe_status: failed` | OLS rejected fit (singular, negative $b$, all-zero RSS) | Container logs for `Probe shape failed` warnings; non-Linux platform; concurrent load contamination | Fall back to `BGE_M3_DISABLE_AUTO_BUDGET=1` until the root cause is identified |
| `probe_status: running` for $> 5$ min | Probe hung on a shape that cannot run | `BGE_M3_MAX_SEQ_LENGTH` exceeds model variant's positional embedding limit | Lower `MAX_SEQ_LENGTH` or switch `BGE_M3_MODEL` to `fp32` |
| `utilization_pct > 90` startup `WARN` | Container too small for the worker count | `BGE_M3_WORKERS`, container memory, `MEMORY_SAFETY_FACTOR` | Reduce workers or raise container memory; see [Resizing workers](#resizing-workers) below |
| `a` looks like $16\,384$ and `b` looks like $8$ | Conservative defaults are active (probe did not run or failed) | `probe_status` field | If `cache_hit`, fine. If `failed`/`disabled`, see those rows. |
| OOM kills mid-traffic at long context | Per-worker workspace under-budgeted | `worst_case_peak_bytes` should be $<$ `available_bytes` | Reduce `BGE_M3_WORKERS` or `BGE_M3_MEMORY_SAFETY_FACTOR` |
| Throughput unexpectedly low at short sequences | $a$ clamped high, conservative $b$ | Fitted $a$ looks $> 50\,000$? Probe captured a kernel-switch artefact | Force re-probe (`BGE_M3_DISABLE_PROBE_CACHE=1`); if persistent, file a bug with the cold-start log |
| $b$ is 0 or extremely small | Fit corrupted by RSS noise | `model_rss_bytes_per_worker` should be ${\sim}1.1\,\text{GB}$; if it is 0, see that row | Force re-probe; investigate RSS reading |

## Forcing a fresh probe

```bash
BGE_M3_DISABLE_PROBE_CACHE=1 ./bge-m3-embedding-server
```

Bypasses the cache without affecting other behaviour. Use when validating a new deployment, after manual edits to the cache file, or when debugging probe behaviour. The fresh fit is still written to the cache (overwriting the previous entry); subsequent starts can use the cache normally — set the env var only on the validation start.

## Skipping the probe entirely

```bash
BGE_M3_DISABLE_AUTO_BUDGET=1 ./bge-m3-embedding-server
```

Server boots with conservative defaults immediately. Use for fast dev-loop iteration on macOS or when running smoke tests where probe time matters more than packing optimality. `probe_status` becomes `disabled` to make the deliberate skip visible in `/health`.

## Pinning explicit coefficients

```bash
BGE_M3_COST_MODEL_A=20000 BGE_M3_COST_MODEL_B=5.0 \
  BGE_M3_AVAILABLE_MEMORY_BYTES=10737418240 \
  ./bge-m3-embedding-server
```

All three must be set together — partial overrides are intentionally rejected (see `Config::from_env`). Use when reproducing a production incident locally with the same coefficients. Useful when fitted values from a production trace need to be replayed in a controlled environment.

This is the only path that gives direct control over $(a, b)$. The probe is bypassed; `probe_status` becomes `disabled`.

## Resizing workers

The most common production tuning is changing `BGE_M3_WORKERS`. The trade-off:

- **More workers → more parallelism, less per-worker budget.** Each worker gets $(\text{available} - N \times \text{model\_rss} - \text{OS\_HEADROOM}) \times \text{safety} / N$. Doubling $N$ roughly halves the per-worker budget.
- **Fewer workers → larger per-worker budget but lower throughput ceiling.** Inference is single-threaded inside an ORT session, so total inference throughput scales linearly with worker count.

A rule of thumb that has worked well in production:

| Container memory | Recommended workers (fp16) | Per-worker workspace | Notes |
|------------------|----------------------------|---------------------|-------|
| 14 GB | 2 | ${\sim}3\,\text{GB}$ | Comfortable for `MAX_SEQ = 8192` |
| 28 GB | 4–5 | ${\sim}3\,\text{GB}$ | Production sweet spot |
| 28 GB | 7 | ${\sim}1\,\text{GB}$ | Throughput-tuned; tighter packing |
| 56 GB | 10–14 | ${\sim}3\,\text{GB}$ | High-throughput tier |

Re-probe after changing `BGE_M3_WORKERS`. Although the worker count does not invalidate the cache (§9), the *budget* changes, and `utilization_pct` should be verified to stay under $90\%$.

## Reading log lines for diagnosis

The most informative log lines for probe diagnosis:

| Log line | What it indicates |
|----------|-------------------|
| `Memory detected available_bytes=... source=...` | Whether cgroup detection worked. `source=cgroup_v2` is normal in Fargate; `source=sysctl` means macOS or cgroups failed. |
| `Workspace budget computed ... utilization_pct=...` | Final budget formula output; $> 90\%$ triggers a startup `WARN`. |
| `Probe cache fingerprint mismatch; will re-probe` | Cache was present but did not match — what changed? (model, max_seq, arch, server_version) |
| `Probe: skipping shape (estimated to exceed rss_ceiling)` | OOM-protection layer in §6 — typically benign on small containers. |
| `Probe shape failed; skipping batch=N seq=N error=...` | A shape errored at runtime; if `seq` matches `MAX_SEQ`, the model variant is incompatible with that length. |
| `Probe: fitted cost model a=... b=... data_points=...` | Successful fit; record $a$, $b$, and the data-point count for future comparison. |
| `Probe coefficients cached to EFS` | Atomic write succeeded; subsequent starts will hit the cache. |
| `Cost model: using conservative defaults` | Probe did not run (disabled) or failed; conservative $(16384, 8)$ are active. |

## Legacy translation

`BGE_M3_ONNX_BATCH_SIZE` is deprecated; setting it logs a `WARN` and translates internally to `BGE_M3_TOKEN_BUDGET` (a workspace ceiling). Migrate to:

- `BGE_M3_TOKEN_BUDGET` for "give me roughly the same packing as before";
- the auto-budget (default) for "give me the best packing my container can support".

## Verifying the probe end-to-end

A post-deployment smoke test:

```bash
# 1. Start container, wait for /health to return 200 with probe_status:complete or cache_hit
curl -sf http://host:8081/health | jq -r '.tuning.probe_status'
# Expected: "complete" or "cache_hit"

# 2. Verify utilization is below 90%
curl -s http://host:8081/health | jq '.tuning.utilization_pct'
# Expected: < 90.0

# 3. Verify a and b look reasonable for the deployment
curl -s http://host:8081/health | jq '{a: .tuning.a_bytes_per_token, b: .tuning.b_bytes_per_token_sq}'
# Expected for fp16 amd64: a ≈ 18000, b ≈ 6
# Expected for conservative defaults: a == 16384, b == 8

# 4. Verify model_rss is non-zero (Linux only)
curl -s http://host:8081/health | jq '.tuning.model_rss_bytes_per_worker'
# Expected: > 1_000_000_000 (≈1 GB)

# 5. Run a real embedding request to validate the bin-packer
curl -X POST http://host:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["test text"], "model": "bge-m3"}' | jq '.data[0].embedding | length'
# Expected: 1024
```

Any failure is a signal to investigate before declaring the deployment ready.

## When the probe is the wrong tool

Some operating modes do not benefit from the probe:

- **Local macOS dev.** RSS reads do not work and the probe falls back to defaults anyway. Set `BGE_M3_DISABLE_AUTO_BUDGET=1` to skip the probe machinery entirely and save a few seconds at startup.
- **CI smoke tests.** Every probe run takes ${\sim}120\,\text{s}$. For tests that just verify the server boots and accepts requests, `BGE_M3_DISABLE_AUTO_BUDGET=1` is much faster.
- **Reproducing a production fit locally.** Pin the exact $(a, b)$ from production via `BGE_M3_COST_MODEL_{A,B}` plus `BGE_M3_AVAILABLE_MEMORY_BYTES` to avoid local-machine variation.

For everything else — production deployments, performance tuning, model-variant evaluation — let the probe run.

## Quick env-var reference

Every probe-related env var, sorted by likelihood of needing to touch it.

| Env var | Default | When to set it |
|---------|---------|---------------|
| `BGE_M3_WORKERS` | `2` | Always — tune for the container size |
| `BGE_M3_MAX_SEQ_LENGTH` | `8192` | Only if the model variant does not support 8192 (lower it) |
| `BGE_M3_MODEL` | `fp16` | If on Apple Silicon (use `fp32`) or if the probe shows positional-embedding errors at `MAX_SEQ` (try `fp32`) |
| `BGE_M3_DISABLE_PROBE_CACHE` | unset | One-shot, when validating a new deployment |
| `BGE_M3_DISABLE_AUTO_BUDGET` | unset | macOS dev, CI, smoke tests |
| `BGE_M3_MEMORY_SAFETY_FACTOR` | `0.7` | Lower (e.g., `0.5`) as a stopgap when `model_rss_bytes_per_worker` is unexpectedly 0 |
| `BGE_M3_AVAILABLE_MEMORY_BYTES` | detected | Override only when reproducing a production budget locally |
| `BGE_M3_COST_MODEL_A` / `BGE_M3_COST_MODEL_B` | unset | Reproducing a production fit; must set both + `AVAILABLE_MEMORY_BYTES` |
| `BGE_M3_TOKEN_BUDGET` | unset | Legacy translation target for the deprecated `BGE_M3_ONNX_BATCH_SIZE` |
| `BGE_M3_ONNX_BATCH_SIZE` | unset | **Deprecated** — translates to `TOKEN_BUDGET`, will be removed in a future release |

---

← [Previous: End-to-end](11-end-to-end.md) | [↑ Series overview](../startup-probe.md) | [Next: References →](13-references.md)
