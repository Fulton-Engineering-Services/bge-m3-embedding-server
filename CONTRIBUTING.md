# Contributing to bge-m3-embedding-server

Contributions are welcome — bug reports, fixes, new features, and documentation improvements.

By submitting a pull request you agree your contribution is made under the
Apache License 2.0 (inbound = outbound), and you certify the
[Developer Certificate of Origin](https://developercertificate.org/) by signing off
each commit with `git commit -s`.

---

## Dev environment

### Rust toolchain

Requires **Rust 1.88** or later (MSRV). Install or update via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### cargo nextest

The test suite uses [cargo-nextest](https://nexte.st/). Install it once:

```bash
cargo install cargo-nextest --locked
```

### Build

```bash
cargo build
```

---

## Running checks

All five commands must pass before a PR is merged.

```bash
# Lint — no warnings permitted
cargo clippy --all-targets -- -D warnings

# Format check (CI gate)
cargo fmt --check

# Auto-format
cargo fmt

# Tests
cargo nextest run --all-features --no-tests=warn

# Supply-chain audit (licenses + advisories)
cargo deny check

# License header check (.rs files)
hawkeye check
```

Install `cargo-deny` and `hawkeye` once if needed:

```bash
cargo install cargo-deny --locked
cargo install hawkeye --locked
```

---

## All checks must pass

CI runs `cargo clippy`, `cargo fmt --check`, `cargo nextest run --all-features`,
`cargo deny check`, and `hawkeye check` on every pull request. A PR will not be merged
until all gates are green.

---

## Visualisation tools

The `tools/probe-visuals/` directory contains Python scripts that generate the figures in
[`docs/startup-probe.md`](docs/startup-probe.md). If you are editing that document or adding
new figures, see [`tools/probe-visuals/README.md`](tools/probe-visuals/README.md) for
prerequisites, quick start, and the figure index.

```bash
cd tools/probe-visuals
uv sync
uv run python scripts/render_all.py
```

---

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) titles:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New user-facing feature |
| `fix:` | Bug fix |
| `perf:` | Performance improvement |
| `docs:` | Documentation only |
| `chore:` | Build, CI, tooling, dependency bumps |
| `refactor:` | Internal restructure, no behaviour change |
| `test:` | Test-only change |

Breaking changes: add `!` after the prefix (e.g. `feat!:`) **and** a `BREAKING CHANGE:` footer.

Sign every commit with DCO:

```bash
git commit -s -m "feat: add sparse embedding batch endpoint"
```

---

## Pull requests

- Target the `main` branch.
- Reference any related issue in the PR description (`Closes #N`).
- DCO sign-off is required on every commit — CI enforces this.
- Keep PRs focused; split unrelated changes into separate PRs.
- Update `CHANGELOG.md` (if present) or note changes in the PR description.

---

## Releases

1. Bump the version string in **`Cargo.toml`**.
2. Commit: `chore: bump version to X.Y.Z`
3. Push to `main`.

The Release workflow detects the version bump, creates the git tag `vX.Y.Z`, builds
multi-arch Docker images, and publishes a GitHub Release automatically. Do **not** create
tags locally.
