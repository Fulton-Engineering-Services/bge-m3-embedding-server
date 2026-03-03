# Suggested Commands

## Build
```bash
cargo build
cargo build --release
```

## Test (requires cargo-nextest)
```bash
cargo nextest run --all-features --no-tests=warn
```
67 unit tests. Integration testing requires a running instance with models downloaded.

## Lint
```bash
cargo clippy --all-targets -- -D warnings
```

## Format
```bash
cargo fmt --all            # apply formatting
cargo fmt --all --check    # check only (CI uses this — always format before pushing)
```

## Supply Chain Audit (requires cargo-deny)
```bash
cargo deny check
```

## Run Locally
```bash
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache cargo run
```
First run downloads ~2GB of BGE-M3 ONNX models. Server ready when logs show "ready".

## Docker
```bash
docker build -t bge-m3-embedding-server .
docker run --rm -p 8081:8081 -v /path/to/model-cache:/cache bge-m3-embedding-server
```

## Benchmarks
```bash
cargo bench --bench embeddings   # requires running model instance
```

## Git Push (multi-account — FES repos require jpfulton-fultonengineeringservices)
```bash
TOKEN=$(gh auth token --user jpfulton-fultonengineeringservices)
git push "https://jpfulton-fultonengineeringservices:${TOKEN}@github.com/Fulton-Engineering-Services/bge-m3-embedding-server.git"
```

## Release
Bump version in Cargo.toml → commit → push to main. Release workflow creates the tag automatically.
Do NOT create tags locally.
