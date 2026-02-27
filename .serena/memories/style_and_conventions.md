# Style and Conventions

## Rust Style
- **Edition**: 2021
- **MSRV**: 1.88
- **Formatter**: rustfmt with `imports_granularity = "Crate"` and `group_imports = "StdExternalCrate"` (rustfmt.toml)
- **Lints**: `unsafe_code = "forbid"`; clippy `all` + `pedantic` at warn level — all clippy warnings are enforced in CI

## Naming Conventions
- Standard Rust conventions: `snake_case` for functions/variables, `PascalCase` for types
- Handler functions are named after the endpoint (e.g. `dense_embeddings`, `sparse_embeddings`, `health`, `models`)
- Test helper constructors use descriptive suffixes: `_for_test()`, `closed_for_test()`, `idle_for_test()`

## Error Handling
- Public error type: `AppError` enum with `InvalidRequest` (400), `ServiceUnavailable` (503), `InternalError` (500)
- Implements `IntoResponse` for Axum; implements `From<anyhow::Error>`
- Internal errors use `anyhow::Error`; never expose internal details in HTTP responses

## Testing Conventions
- Unit tests in `#[cfg(test)] mod tests` at the bottom of each file
- Config tests use `Config::from_lookup()` with a closure (not `env::set_var`) to avoid global state pollution
- Handler tests use `EmbedPool::closed_for_test()` / `with_fixed_responses()` / `idle_for_test()` to avoid model loading
- Router-level tests use `tower::ServiceExt::oneshot()` (never bind to a port)
- Property tests use `proptest`

## Code Organization
- One concern per file; `models.rs` = wire types only, `handler.rs` = HTTP handlers only, `embedder.rs` = pool + workers
- `TextInput` uses a custom `Deserialize` impl to accept both `String` and `Vec<String>`
- All public API surface is minimal — use `pub(crate)` where possible

## Important Gotchas
- `fastembed::SparseEmbedding` does not implement `Debug` → use `.err().expect(msg)` not `.unwrap_err()` on Results containing it
- Always run `cargo fmt --all` before pushing — CI enforces `cargo fmt --all --check`
