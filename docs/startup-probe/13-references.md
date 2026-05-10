# 13. References

> Sources for everything in this series — the numerical-linear-algebra theory behind OLS and Jacobi preconditioning, the transformer-architecture papers that justify the cost model's two-term form, and the systems documentation for the ONNX Runtime, ArcSwap, and POSIX primitives the implementation relies on.

## Numerical linear algebra

The probe combines standard numerical-linear-algebra techniques with transformer-architecture-specific cost reasoning. Useful background reading for contributors:

- **Ordinary least squares and the normal equations.** Trefethen & Bau, *Numerical Linear Algebra*, Lectures 11 (least squares) and 18 (conditioning of least squares). Chapter 11 derives the normal equations `Xᵀ X θ = Xᵀ y` and the closed-form Cramer's-rule solution we use. Chapter 18 explains why ill-conditioned `XᵀX` causes the determinant-vs-max-diagonal stability check to fail — exactly what page [Conditioning §5](05-conditioning.md) describes for our 8000× column-magnitude gap at `MAX_SEQ=8192`.

- **Diagonal preconditioning / Jacobi scaling.** Saad, *Iterative Methods for Sparse Linear Systems*, ch. 10 — covers preconditioners; the simplest case is diagonal scaling, which is exactly what we do for the 2-column normal equations. The Jacobi preconditioner is usually presented in the iterative-methods context (where it accelerates convergence), but the same coordinate-change argument applies to the direct-solve case: rescaling columns to similar magnitudes makes the conditioning of `XᵀX` independent of column scale.

- **Condition number of a Gram matrix.** Golub & Van Loan, *Matrix Computations*, §5.3 (LS conditioning) and §3.5.4 (column scaling). Theorem 5.3.1 bounds the condition number of `XᵀX` by the squared condition number of `X` — which is why our `(S_max / S_min)⁴` blow-up appears: the column scale ratio in `X` is `O((S_max/S_min)²)`, and squaring that for the Gram matrix gives the quartic.

## Transformer architecture

- **Attention mechanism complexity.** Vaswani et al., *Attention Is All You Need* (NeurIPS 2017) — the original paper. §3.2 derives the `O(B·S²)` attention term and `O(B·S·D)` projection terms used in the cost decomposition on page [Cost decomposition §2](02-cost-decomposition.md). Specifically:
  - `Q · Kᵀ` produces a `[B, H, S, S]` tensor — the attention scores. This is the source of the quadratic-in-`S` term.
  - The QKV projections, attention output projection, and FFN intermediates are all `[B, S, hidden]` shaped — linear-in-`S` terms.

- **BGE-M3 specifics.** Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (2024) — the model paper. Confirms BGE-M3's architecture: 24 transformer layers, 16 attention heads, 1024 hidden dim, 4096 FFN intermediate dim, 8192 max position. These are the `(L, H, D, D_ff, max_S)` constants that go into the cost-decomposition tables.

## Runtime systems

- **ONNX Runtime arena allocator.** [`onnxruntime` docs on arena-based allocators](https://onnxruntime.ai/docs/api/c/struct_ort_arena_cfg.html) — explains why RSS deltas at first-touch overstate steady-state workspace and why we sample only once per shape. Background reading for [Measurement §7](07-measurement.md) (Layer 1 RSS deltas) and [Probe shapes §6](06-probe-shapes.md) (the `4×` multiplier in the absolute-RSS guard).

- **`ArcSwap` and lock-free pointer swaps.** [`arc-swap` crate documentation](https://docs.rs/arc-swap/latest/arc_swap/) — the wait-free read-many, write-rarely primitive used for the cost-model handoff in [Execution §10](10-execution.md). The crate's design notes explain why a single atomic pointer load is much cheaper than a `Mutex<Arc<T>>` acquire-release roundtrip when reads dominate writes.

- **Tokio semaphores and owned permits.** [`tokio::sync::Semaphore`](https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html) and [`OwnedSemaphorePermit`](https://docs.rs/tokio/latest/tokio/sync/struct.OwnedSemaphorePermit.html). The `acquire_many_owned` + `forget()` + `add_permits` pattern from [Execution §10](10-execution.md) is the idiomatic way to hold a permit across an `async move` boundary.

- **POSIX atomic rename semantics.** `rename(2)` on Linux man page — the basis for the cache file's atomic-write strategy in [Cache §9](09-cache.md). Specifically, the manpage guarantees that "if newpath already exists, it will be atomically replaced" — ensuring concurrent readers always see either the complete old file or the complete new file, never a half-written file.

- **`/proc/self/statm` semantics.** `proc(5)` Linux manpage, "/proc/[pid]/statm" section. Field 1 is *resident set size* in pages; multiplied by the page size (4096 on Linux/x86_64 and Linux/aarch64) it gives RSS in bytes. This is the basis for `read_process_rss_bytes()` in [Measurement §7](07-measurement.md).

- **cgroup memory accounting.** [Linux Kernel cgroup-v2 documentation, "Memory Controller" section](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory). The probe walks `/sys/fs/cgroup/memory.max` and falls back to cgroup-v1 paths when v2 is unavailable. See `src/sysinfo.rs` for the path-walk implementation.

## Statistical considerations

- **Asymmetric loss in measurement systems.** [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, §5.8 "Evaluating point forecast accuracy"](https://otexts.com/fpp3/accuracy.html) — discusses asymmetric loss functions where over-prediction and under-prediction have different costs. The probe's asymmetric clamping in [Clamps & fallback §8](08-clamps-fallback.md) reflects exactly this: under-counting `b` is fatal (OOM) while over-counting `a` is merely slow.

- **Box's "all models are wrong" aphorism.** Box & Draper, *Empirical Model-Building and Response Surfaces* (1987), p. 424: "Essentially, all models are wrong, but some are useful." Cited on page [Cost decomposition §2](02-cost-decomposition.md) — the two-coefficient model is *deliberately* a simplification of transformer workspace, sufficient to drive the bin-packer but not a complete physical model.

## Related project documentation

For BGE-M3-specific details (model variants, hybrid retrieval, fp16/int8 trade-offs):

- [`bge-m3-model.md`](../bge-m3-model.md) — Model provenance, vocabulary, dense/sparse capabilities, hybrid scoring, vector storage compatibility
- [`model-variants.md`](../model-variants.md) — FP32 vs FP16 precision evaluation, quantized model table, production recommendation
- [`performance.md`](../performance.md) — MLAS vs CoreML benchmarks, CoreML workspace analysis, embedding quality, memory footprint, RAM reduction options
- [`coreml-ep.md`](../coreml-ep.md) — Apple Silicon compute units, custom ORT build, ENOTDIR fix, BGE-M3 op coverage, CoreML EP configuration

For the surrounding system architecture:

- [`architecture.md`](../architecture.md) — Component diagram, module layout, worker pool design, middleware stack
- [`cold-start.md`](../cold-start.md) — Leader–follower startup pattern, failure modes, idle-reload comparison
- [`request-flow.md`](../request-flow.md) — End-to-end request lifecycle for dense and sparse endpoints
- [`health-state-machine.md`](../health-state-machine.md) — Health endpoint states, decision logic, Docker HEALTHCHECK integration
- [`deployment.md`](../deployment.md) — `install-bge-m3-apple.sh`, LaunchAgent configuration, service management

## Visual companions

The figures embedded in this series are generated by the `tools/visuals/` package. To regenerate or modify them, see [`tools/visuals/README.md`](../../tools/visuals/README.md). The package also ships three interactive Jupyter notebooks for hands-on exploration of the math:

| Notebook | What it explores |
|----------|------------------|
| `01_cost_decomposition_explorer.ipynb` | Sliders for `a` and `b` — live crossover point, preset buttons for fitted vs conservative defaults |
| `02_workspace_budget_calculator.ipynb` | Deployment sizing tool — workers, model RSS, available memory, safety factor → utilization traffic light |
| `03_conditioning_visualiser.ipynb` | Column scale ratio slider — morphs OLS loss landscape from circular to elongated, shows condition number |

These notebooks are the most direct way to develop intuition for the trade-offs covered in pages [Cost decomposition §2](02-cost-decomposition.md), [Operator guide §12](12-operator-guide.md), and [Conditioning §5](05-conditioning.md) respectively.

---

← [Previous: Operator guide](12-operator-guide.md) | [↑ Series overview](../startup-probe.md)
