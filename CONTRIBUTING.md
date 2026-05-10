# Contributing to bge-m3-embedding-server

`bge-m3-embedding-server` is open source under the Apache License 2.0.
Contributions are welcome — bug reports, fixes, and new features alike.

By submitting a pull request you certify that your contribution is made under
the [Developer Certificate of Origin (DCO)](https://developercertificate.org/)
and the same Apache 2.0 license as the project (inbound = outbound). Every
commit must carry a `Signed-off-by` trailer matching your author identity:

```
git commit -s -m "feat: your change"
```

To add sign-off to all commits on an existing branch:

```bash
git rebase origin/main --signoff
git push --force-with-lease
```

---

## Development environment

Requires Rust **1.88** or later (MSRV) and
[cargo-nextest](https://nexte.st/) for the test suite.

```bash
# Clone and enter the repo
git clone git@github.com:Fulton-Engineering-Services/bge-m3-embedding-server.git
cd bge-m3-embedding-server

# Run the server locally (downloads model files on first run)
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```

---

## Running the checks

All of the following must pass before a PR is merged.

```bash
# Build
cargo build

# Lint (warnings are errors)
cargo clippy --all-targets -- -D warnings

# Format check
cargo fmt --check

# Tests (requires cargo-nextest)
cargo nextest run --all-features --no-tests=warn

# Supply chain audit (requires cargo-deny)
cargo deny check

# License header check (requires hawkeye)
hawkeye check
```

---

## Project layout

```
src/
  lib.rs                    # library crate root (re-exports for tests)
  main.rs                   # thin binary entry point
  bootstrap.rs              # startup orchestration (router, budget, probe task, readiness)
  bootstrap/
    budget.rs               # workspace-budget arithmetic
    probe_task.rs           # background probe task
    readiness.rs            # foreground readiness probe
    router.rs               # Axum router + middleware
  config.rs                 # env-var configuration (Config::from_env)
  state.rs                  # AppState, TuningInfo, ProbeStatus
  handler.rs                # HTTP handler module (facade)
  handler/
    common.rs               # shared validation + readiness helpers
    dense.rs                # POST /v1/embeddings
    sparse.rs               # POST /v1/sparse-embeddings
    both.rs                 # POST /v1/embeddings:both
    health.rs               # GET /health
    models.rs               # GET /v1/models
  embedder.rs               # worker-pool module (facade)
  embedder/
    pool.rs                 # EmbedPool async wrapper
    worker.rs               # blocking worker thread + probe wiring
    dense.rs                # dense embedding pipeline
    sparse.rs               # sparse embedding pipeline
    dual.rs                 # paired dense + sparse pipeline
    session.rs              # ORT session loading + execution providers
    tokenize.rs             # tokenizer load + no-pad tokenization
    math.rs                 # pure dense/sparse math helpers
    model_files.rs          # hf-hub download / cache layout
    types.rs                # EmbedRequest, ProbeResult, SparseEmbedding
    error.rs                # ort::Error → anyhow::Error adapter
  binpack.rs                # quadratic-aware workspace bin-packer + CostModel
  probe.rs                  # cost-model probe module (facade)
  probe/
    runner.rs               # (batch, seq) shape-sweep driver
    fit.rs                  # OLS cost-model fitter
    cache.rs                # persistent EFS coefficient cache
    corpus.rs               # probe text synthesis
    validate.rs             # tokenizer + ndarray shape check
  sysinfo.rs                # memory detection (cgroup v2/v1 → /proc/meminfo)
  weights.rs                # bundled sparse_linear.safetensors weights
  models.rs                 # request / response serde types
  error.rs                  # AppError → HTTP status mapping
benches/                    # Criterion benchmarks
tests/                      # integration tests
```

---

## Releases

Releases are managed via the GitHub Actions release workflow.

- Use [Conventional Commits](https://www.conventionalcommits.org/) on every PR:
  `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `perf:`, etc.
- `feat!:` or a `BREAKING CHANGE:` footer triggers a major version bump.
- To cut a release: bump `version` in `Cargo.toml`, commit, push to `main`.
  The release workflow creates the git tag, builds multi-arch Docker images,
  and publishes a GitHub Release automatically.
- Do **not** create tags locally — let the workflow handle them.
