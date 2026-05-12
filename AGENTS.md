# AGENTS.md — bge-m3-embedding-server

Full project context is in [`CLAUDE.md`](CLAUDE.md). This file adds cloud-agent-specific
notes that differ from the local development workflow.

## Cloud Environment

The Cursor cloud environment (`.cursor/Dockerfile`) provides:
- Rust stable with `rustfmt` and `clippy`
- `cargo-nextest`, `cargo-deny`, `hawkeye`
- No prebuilt ONNX Runtime library — use the `download-ort` feature for all builds

## Build & Test Commands (Cloud)

Always append `--features download-ort` when building or testing. The `download-ort` feature
fetches the prebuilt ORT static library at build time; without it, `cargo build` fails because
`ORT_LIB_LOCATION` is not set in this environment.

```bash
# Build
cargo build --features download-ort

# Test
cargo nextest run --features download-ort --no-tests=warn

# Lint
cargo clippy --all-targets --features download-ort -- -D warnings

# Format check
cargo fmt --all --check

# Supply-chain audit
cargo deny check

# License headers
hawkeye check
```

## Model Downloads

The server downloads model files from Hugging Face on first run. For quick functional checks
that do not require a running server, rely on unit tests (`cargo nextest run`) — they do not
require model files.

To run the server locally in the cloud environment, set `BGE_M3_CACHE_DIR` and disable the
startup probe to avoid the multi-minute probe sweep:

```bash
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache BGE_M3_DISABLE_AUTO_BUDGET=1 \
  cargo run --features download-ort
```

## Key Notes

- **`--features download-ort` is mandatory** in this environment — do not omit it.
- **`BGE_M3_DISABLE_AUTO_BUDGET=1`** skips the Linux startup probe; use it for dev/test
  runs where the 2-minute probe sweep would waste time. Leave it unset for production.
- **`cargo fmt --all` before pushing** — CI rejects any formatting drift.
- Do not create git tags manually; see `CLAUDE.md` for the release workflow.
