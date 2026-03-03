# Contributing

Thank you for your interest in contributing to bge-m3-embedding-server!

## Quick Start

```bash
# Clone and build
git clone https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server
cd bge-m3-embedding-server
cargo build

# Run the full test suite (does not require model download)
cargo nextest run --all-features --no-tests=warn

# Lint
cargo clippy --all-targets -- -D warnings

# Format check
cargo fmt --check

# Supply chain audit
cargo deny check
```

## Before Opening a PR

All of the following must pass:

- [ ] `cargo nextest run --all-features --no-tests=warn`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo fmt --check`
- [ ] `cargo deny check`

The CI pipeline enforces all of the above automatically.

## Coding Standards

- No `unsafe` code (`[lints.rust] unsafe_code = "forbid"` is enforced)
- All Clippy pedantic warnings must be resolved (not just suppressed unless justified)
- `fastembed::SparseEmbedding` does not implement `Debug` — use `.err().expect(msg)`
  instead of `.unwrap_err()` on `Result<Vec<SparseEmbedding>>`

## Running Locally with Models

```bash
# Download models to /tmp/bge-m3-cache and start the server
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```

First run downloads ~2 GB of ONNX model files. Subsequent runs reuse the cache.

## Releasing

Maintainers only. See `CLAUDE.md` for the release workflow.
