# Deployment — macOS LaunchAgent (Apple Silicon)

## Overview

The server deploys on Apple Silicon Macs as a launchd UserAgent on port **8089**.
The install script handles the full lifecycle: custom ORT build, binary compilation,
installation, and LaunchAgent bootstrap. Service label: `ai.bge-m3.server`.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Xcode Command Line Tools: `xcode-select --install`
- Rust toolchain: [rustup](https://rustup.rs)
- Python 3 — already present on macOS; used by the ORT build system
- CMake — installed automatically by the script via pip into an isolated venv
  (do NOT install from Homebrew — see [coreml-ep.md](coreml-ep.md) for why Homebrew CMake is excluded)

## Installation

```bash
# From the repo root (builds ORT from source — takes 15–30 min on first run):
./scripts/install-bge-m3-apple.sh

# Or, install with a pre-built binary (skips ORT and Rust build steps):
./scripts/install-bge-m3-apple.sh /path/to/bge-m3-apple
```

What the script does:

1. Verifies Apple Silicon (arm64)
2. Clones and builds ONNX Runtime from the FES fork with CoreML EP enabled — takes 15–30 minutes
   (see [coreml-ep.md](coreml-ep.md) for details on the fork and the external-data-path fix)
3. Compiles `bge-m3-embedding-server` with `ORT_LIB_LOCATION` pointing at the source-built ORT
   and `RUSTFLAGS="-C target-cpu=native"` for Apple Silicon instruction-set optimizations
4. Installs the binary to `~/.local/bin/bge-m3-apple`
5. Creates the model cache directory at `~/.cache/bge-m3/`
6. Creates the log directory at `~/Library/Logs/bge-m3-apple/`
7. Installs the LaunchAgent plist to `~/Library/LaunchAgents/ai.bge-m3.server.plist`,
   substituting `__HOME__` with the actual home path
8. Bootstraps the agent via `launchctl bootstrap gui/$(id -u)`
9. Runs a health check on `http://localhost:8089/health`

The script is idempotent — safe to re-run to update the binary or plist.

## LaunchAgent Configuration

Source: [`../scripts/ai.bge-m3.server.plist`](../scripts/ai.bge-m3.server.plist)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `BGE_M3_BIND` | `0.0.0.0:8089` | Port 8089 avoids conflict with llama-server (8088) |
| `BGE_M3_WORKERS` | `2` | Two worker threads, each with its own ORT session |
| `BGE_M3_MAX_BATCH` | `256` | Maximum texts per request |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `0` | Models stay resident permanently — dedicated server |
| `BGE_M3_MAX_SEQ_LENGTH` | `2048` | Sequence length capped to match consumer `max-tokens=2048`; avoids the full 8192 probe cost on Apple Silicon where auto-budget uses conservative defaults. |
| `BGE_M3_DISABLE_AUTO_BUDGET` | `1` | Startup probe uses Linux cgroup/proc APIs; skip on macOS to avoid unnecessary delay. |
| `BGE_M3_MODEL` | `fp16` | Xenova/bge-m3 FP16 (~1.08 GB/session); halves peak memory vs fp32. See [model-variants.md](model-variants.md) |
| `BGE_M3_CACHE_DIR` | `~/.cache/bge-m3` | Model cache location (resolved from `__HOME__` by the install script; `launchd` does not expand `~`) |
| `BGE_M3_LOG_FORMAT` | `json` | Structured logging |
| `RUST_LOG` | `info` | Log level |
| `KeepAlive` | `true` | Restart on crash |
| `RunAtLoad` | `true` | Start at login |
| `ThrottleInterval` | `10` s | Minimum seconds between restart attempts |

## Service Management

```bash
# Status
launchctl list ai.bge-m3.server

# Stop
launchctl bootout gui/$(id -u)/ai.bge-m3.server

# Start (after a bootout, or on a machine where RunAtLoad has not yet triggered)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.bge-m3.server.plist

# Force restart (kill and relaunch)
launchctl kickstart -k gui/$(id -u)/ai.bge-m3.server

# Logs (live tail)
tail -f ~/Library/Logs/bge-m3-apple/stderr.log
```

## Log Locations

| Stream | Path |
|--------|------|
| stdout | `~/Library/Logs/bge-m3-apple/stdout.log` |
| stderr | `~/Library/Logs/bge-m3-apple/stderr.log` |

Most useful output goes to stderr (Rust `tracing` writes to stderr by default).
Use the stderr log for troubleshooting.

## Upgrading

Re-run `./scripts/install-bge-m3-apple.sh`. The script is idempotent — it replaces
the binary and plist in-place and restarts the agent.

If a new binary is already built elsewhere, pass it directly to skip the ORT and Rust build steps:

```bash
./scripts/install-bge-m3-apple.sh /path/to/new-bge-m3-apple
```
