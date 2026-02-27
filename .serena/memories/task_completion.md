# Task Completion Checklist

When a coding task is complete, run the following in order:

1. **Format** (required — CI fails on fmt violations):
   ```bash
   cargo fmt --all
   ```

2. **Lint**:
   ```bash
   cargo clippy --all-targets -- -D warnings
   ```

3. **Test**:
   ```bash
   cargo nextest run --all-features --no-tests=warn
   ```
   Expected: 67 tests pass. No integration tests (require running model instance).

4. **Supply chain** (if deps changed):
   ```bash
   cargo deny check
   ```

5. **Build check** (if structural changes):
   ```bash
   cargo build
   ```

6. **Push** (uses multi-account pattern — see suggested_commands.md):
   ```bash
   TOKEN=$(gh auth token --user jpfulton-fultonengineeringservices)
   git push "https://jpfulton-fultonengineeringservices:${TOKEN}@github.com/Fulton-Engineering-Services/bge-m3-axum-fastembed-rs.git"
   ```

## For Releases
1. Bump `version` in `Cargo.toml`
2. `cargo fmt --all && cargo clippy ... && cargo nextest run ...`
3. Commit and push to main — Release workflow creates the tag automatically
