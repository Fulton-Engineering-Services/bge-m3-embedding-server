#!/usr/bin/env bash
# install-bge-m3-apple.sh — Install bge-m3-embedding-server as a launchd UserAgent
#
# Usage: ./scripts/install-bge-m3-apple.sh [/path/to/bge-m3-apple-binary]
#
# Builds (or uses provided) bge-m3-apple binary compiled for aarch64-apple-darwin
# with Apple Silicon optimizations (CoreML EP, target-cpu=native), installs it to
# ~/.local/bin/bge-m3-apple, then bootstraps the LaunchAgent on port 8089.
# Idempotent — safe to re-run to update the binary or plist.
#
# Prerequisites:
#   - Apple Silicon Mac (M1/M2/M3/M4)
#   - Xcode Command Line Tools (xcode-select --install)
#   - Rust toolchain (rustup)
#   - CMake (brew install cmake)
#   - Python 3 (for ORT build system)
#
# Note on protobuf: ORT v1.23.2 fetches and builds its own protobuf v21.12 from
# source. A system protoc on PATH (e.g. from 'brew install protobuf') causes
# CMake's FindProtobuf to pick it up, producing a version mismatch with ORT's
# bundled libprotobuf. This script automatically removes system protoc from PATH
# for the ORT build step — no manual action required.
#
# What this script builds:
#   1. ONNX Runtime from source with CoreML EP enabled (FES fork with
#      external-data-path fix required for BGE-M3 CoreML EP support)
#   2. bge-m3-embedding-server with target-cpu=native and ORT_LIB_LOCATION
#      pointing at the source-built ORT
#
# Run as the user who will own the agent.

set -euo pipefail

info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
warn()  { echo "[WARN]  $*"; }

# ── Locate plist template (bundled alongside this script) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_TEMPLATE="$SCRIPT_DIR/ai.bge-m3.server.plist"
[[ -f "$PLIST_TEMPLATE" ]] || error "Plist template not found: $PLIST_TEMPLATE"

# ── Platform check ───────────────────────────────────────────────────────────
[[ "$(uname)" == "Darwin" ]] || error "This script only runs on macOS."
[[ "$(uname -m)" == "arm64" ]] || warn "Not running on arm64 — bge-m3-apple is optimised for Apple Silicon."

# ── Configuration ────────────────────────────────────────────────────────────
# ORT source: FES fork of v1.23.2 with the CoreML external-data-path fix
# (TensorProtoWithExternalDataToTensorProto receives a file path; fix passes
# parent_path() so CoreML EP can read BGE-M3 external weight data).
# Upstream PR: https://github.com/microsoft/onnxruntime/issues/<pending>
ORT_BASE_TAG="v1.23.2"
ORT_FORK_URL="https://github.com/Fulton-Engineering-Services/onnxruntime.git"
ORT_FORK_BRANCH="fix/coreml-tensorproto-external-data-path"
ORT_FORK_COMMIT="1e37c3583d05992bc1419269f87d941e8642248c"
ORT_BUILD_DIR="$HOME/.local/share/ort-build"
ORT_SOURCE_DIR="$ORT_BUILD_DIR/onnxruntime"
ORT_OUTPUT_DIR="$ORT_BUILD_DIR/output"
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

# The repo root is the parent of this script's ops/ directory.
BGE_M3_REPO="$SCRIPT_DIR/.."

# ── Binary ──────────────────────────────────────────────────────────────────
BINARY_PATH="${1:-}"

if [[ -n "$BINARY_PATH" ]]; then
    [[ -f "$BINARY_PATH" ]] || error "Binary not found: $BINARY_PATH"
    info "Installing binary from $BINARY_PATH..."
    install -m 0755 "$BINARY_PATH" "$INSTALL_DIR/bge-m3-apple"
else
    # ── Step 1: Build ONNX Runtime from source with CoreML EP ────────────
    # Static build produces per-component archives (libonnxruntime_common.a,
    # libonnxruntime_providers.a, etc.) — not a single libonnxruntime.a.
    # ort-sys auto-discovers the full set via its Layout 2 search.
    if [[ -f "$ORT_OUTPUT_DIR/Release/libonnxruntime_common.a" ]]; then
        info "Using cached ORT build at $ORT_OUTPUT_DIR/Release/"
    else
        info "Building ONNX Runtime $ORT_BASE_TAG+FES-patch ($ORT_FORK_BRANCH) from source with CoreML EP..."
        mkdir -p "$ORT_BUILD_DIR"

        # Clone ORT fork if not already present
        if [[ ! -d "$ORT_SOURCE_DIR" ]]; then
            info "Cloning FES onnxruntime fork (shallow, branch $ORT_FORK_BRANCH)..."
            git clone --depth 1 --branch "$ORT_FORK_BRANCH" \
                "$ORT_FORK_URL" "$ORT_SOURCE_DIR"
            CLONED_COMMIT=$(git -C "$ORT_SOURCE_DIR" rev-parse HEAD)
            if [[ "$CLONED_COMMIT" != "$ORT_FORK_COMMIT" ]]; then
                warn "Cloned HEAD $CLONED_COMMIT does not match pinned commit $ORT_FORK_COMMIT — branch may have advanced"
            else
                info "Commit verified: $CLONED_COMMIT"
            fi
        else
            info "ORT source already at $ORT_SOURCE_DIR"
        fi

        # Patch: ORT v1.23.2 CoreML static build fails CMake Generate because
        # coreml_proto isn't in the install EXPORT set. We never run cmake
        # --install (we only need the .a files from the build tree), so just
        # comment out the 3-line install(EXPORT) block.
        ORT_TOP_CMAKE="$ORT_SOURCE_DIR/cmake/CMakeLists.txt"
        if grep -q '^  install(EXPORT' "$ORT_TOP_CMAKE"; then
            info "Patching ORT cmake to disable install(EXPORT) validation..."
            sed -i '' \
                -e 's/^  install(EXPORT/  # PATCHED: disabled for static CoreML build\n  # install(EXPORT/' \
                -e 's/^    NAMESPACE ${PROJECT_NAME}::/  #   NAMESPACE ${PROJECT_NAME}::/' \
                -e 's/^    DESTINATION ${CMAKE_INSTALL_LIBDIR}/  #   DESTINATION ${CMAKE_INSTALL_LIBDIR}/' \
                "$ORT_TOP_CMAKE"
        fi

        # Verify prerequisites
        command -v python3 >/dev/null 2>&1 || error "python3 not found — install Xcode CLI tools or Homebrew Python"

        # ORT v1.23.2 + CoreML + static build hits a CMake 4.x bug where
        # install(EXPORT) validation fails on the coreml_proto target.
        # Work around by using CMake 3.31 from pip in an isolated venv.
        ORT_VENV="$ORT_BUILD_DIR/.venv"
        if [[ ! -x "$ORT_VENV/bin/cmake" ]]; then
            info "Creating build venv with CMake 3.31..."
            python3 -m venv "$ORT_VENV"
            "$ORT_VENV/bin/pip" install --quiet "cmake>=3.31,<4"
        fi
        ORT_CMAKE="$ORT_VENV/bin/cmake"
        info "Using CMake: $ORT_CMAKE ($("$ORT_CMAKE" --version | head -1))"

        # ORT v1.23.2 builds its own protobuf v21.12 from source. A system protoc on
        # PATH (e.g. from 'brew install protobuf') causes CMake's FindProtobuf to find
        # it, mixing headers/libraries from different protobuf versions. Strip any
        # directory containing a system protoc from PATH for this build only.
        ORT_BUILD_PATH="$PATH"
        if command -v protoc >/dev/null 2>&1; then
            PROTOC_DIR="$(dirname "$(command -v protoc)")"
            warn "System protoc found at $PROTOC_DIR/protoc — excluding from ORT build PATH to prevent protobuf version conflict"
            ORT_BUILD_PATH="$(echo "$PATH" | tr ':' '\n' | grep -Fxv "$PROTOC_DIR" | tr '\n' ':' | sed 's/:$//')"
        fi

        info "Running ORT build (this takes 15-30 minutes on first run)..."
        cd "$ORT_SOURCE_DIR"
        PATH="$ORT_BUILD_PATH" python3 tools/ci_build/build.py \
            --cmake_path "$ORT_CMAKE" \
            --build_dir "$ORT_OUTPUT_DIR" \
            --config Release \
            --parallel \
            --compile_no_warning_as_error \
            --skip_tests \
            --osx_arch arm64 \
            --apple_deploy_target 13.0 \
            --use_coreml \
            --cmake_extra_defines \
                CMAKE_OSX_ARCHITECTURES=arm64 \
                onnxruntime_BUILD_UNIT_TESTS=OFF \
                onnxruntime_BUILD_SHARED_LIB=OFF
        cd "$BGE_M3_REPO"

        [[ -f "$ORT_OUTPUT_DIR/Release/libonnxruntime_common.a" ]] || \
            error "ORT build completed but libonnxruntime_common.a not found at $ORT_OUTPUT_DIR/Release/"
        [[ -f "$ORT_OUTPUT_DIR/Release/libonnxruntime_providers_coreml.a" ]] || \
            warn "CoreML provider library not found — CoreML EP will not be available"

        # re2 is configured by ORT's CMake but never compiled in static builds
        # (no production ORT target depends on it directly; it's only used in tests
        # and some contrib ops). ort-sys links it unconditionally in full static
        # builds, so we must compile it explicitly after the main build.
        RE2_BUILD_DIR="$ORT_OUTPUT_DIR/Release/_deps/re2-build"
        if [[ ! -f "$RE2_BUILD_DIR/libre2.a" ]]; then
            info "Building re2 (configured but not compiled by ORT static build)..."
            make -C "$RE2_BUILD_DIR" re2 -j"$(sysctl -n hw.logicalcpu)"
        fi

        info "ORT build complete: $ORT_OUTPUT_DIR/Release/ ($(ls "$ORT_OUTPUT_DIR/Release"/libonnxruntime_*.a | wc -l | tr -d ' ') component archives)"
    fi

    # ── Step 2: Build bge-m3-embedding-server with optimizations ────────
    info "Building bge-m3-apple with Apple Silicon optimizations..."
    info "  ORT_LIB_LOCATION=$ORT_OUTPUT_DIR"
    info "  target-cpu=native (enables i8mm, bf16, bti on M2+)"

    ORT_LIB_LOCATION="$ORT_OUTPUT_DIR" \
    ORT_LIB_PROFILE=Release \
    RUSTFLAGS="-C target-cpu=native" \
    cargo build --release \
        --target aarch64-apple-darwin \
        --manifest-path "$BGE_M3_REPO/Cargo.toml"

    BUILT_BINARY="$BGE_M3_REPO/target/aarch64-apple-darwin/release/bge-m3-embedding-server"
    [[ -f "$BUILT_BINARY" ]] || error "Build succeeded but binary not found at $BUILT_BINARY"
    info "Installing built binary..."
    install -m 0755 "$BUILT_BINARY" "$INSTALL_DIR/bge-m3-apple"
fi
info "Binary: $INSTALL_DIR/bge-m3-apple ($(file "$INSTALL_DIR/bge-m3-apple" | grep -o 'arm64[^,]*' || echo 'installed'))"

# ── Verify linked frameworks ─────────────────────────────────────────────────
if otool -L "$INSTALL_DIR/bge-m3-apple" 2>/dev/null | grep -q CoreML; then
    info "CoreML.framework linked — Neural Engine / GPU dispatch available"
else
    warn "CoreML.framework NOT linked — running CPU-only (MLAS NEON kernels)"
fi

# ── Cache dir ───────────────────────────────────────────────────────────────
mkdir -p "$HOME/.cache/bge-m3"
info "Cache directory: $HOME/.cache/bge-m3"

# ── Log dir ─────────────────────────────────────────────────────────────────
LOG_DIR="$HOME/Library/Logs/bge-m3-apple"
mkdir -p "$LOG_DIR"
info "Log directory: $LOG_DIR"

# ── Install plist ───────────────────────────────────────────────────────────
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS_DIR"
DEST_PLIST="$LAUNCH_AGENTS_DIR/ai.bge-m3.server.plist"
info "Installing plist to $DEST_PLIST..."
sed "s|__HOME__|$HOME|g" "$PLIST_TEMPLATE" > "$DEST_PLIST"

# ── Reload service ──────────────────────────────────────────────────────────
UID_VAL=$(id -u)
if launchctl list "ai.bge-m3.server" &>/dev/null; then
    info "Unloading existing ai.bge-m3.server..."
    launchctl bootout "gui/$UID_VAL/ai.bge-m3.server" 2>/dev/null || true
    sleep 1
fi
info "Bootstrapping ai.bge-m3.server for gui/$UID_VAL..."
launchctl bootstrap "gui/$UID_VAL" "$DEST_PLIST"

# ── Verify ──────────────────────────────────────────────────────────────────
info "Waiting 10s for bge-m3-apple to load models..."
sleep 10
if curl -sf http://localhost:8089/health | grep -q '"status"'; then
    info "Health check passed: http://localhost:8089/health"
    echo
    echo "bge-m3-apple is running. Test with:"
    echo "  curl -s http://localhost:8089/v1/models | jq ."
    echo "  curl -s -X POST http://localhost:8089/v1/embeddings -H 'Content-Type: application/json' -d '{\"input\":[\"test\"]}' | jq '.data[0].embedding | length'"
else
    warn "Health check not yet passing — models may still be loading (BGE-M3 takes ~30s on first run)."
    echo "Check logs: tail -f $LOG_DIR/stderr.log"
    echo "Retry:      curl http://localhost:8089/health"
fi

echo
echo "Service management:"
echo "  Status:  launchctl list ai.bge-m3.server"
echo "  Stop:    launchctl bootout gui/$UID_VAL/ai.bge-m3.server"
echo "  Restart: launchctl kickstart -k gui/$UID_VAL/ai.bge-m3.server"
echo "  Logs:    tail -f $LOG_DIR/stderr.log"
