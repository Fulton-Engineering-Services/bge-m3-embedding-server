# Apple Silicon Build Target

This document describes how `bge-m3-axum-fastembed-rs` builds, links, and
executes on Apple Silicon (aarch64-apple-darwin). It covers the hardware
compute units available on M-series chips, what the current build chain
actually uses (and what it leaves on the table), the Rust release profile,
and the macOS `launchd` deployment used in production.

## Apple Silicon Compute Units

Apple Silicon SoCs contain four independent compute units relevant to ML
inference. Understanding which units are actually exercised by this project
is key to evaluating optimization opportunities.

```mermaid
graph LR
    subgraph "Apple Silicon SoC"
        CPU["CPU Cores<br/>(P + E clusters)<br/>NEON SIMD · AMX matrix"]
        GPU["GPU Cores<br/>Metal Compute Shaders"]
        ANE["Neural Engine<br/>16-core NPU (M4)<br/>Fixed-precision INT8/FP16"]
        UMA["Unified Memory<br/>Shared by all units<br/>Zero-copy between CPU/GPU/ANE"]
    end

    CPU --- UMA
    GPU --- UMA
    ANE --- UMA

    classDef active fill:#6f6,stroke:#333,stroke-width:2px
    classDef idle fill:#ccc,stroke:#999,stroke-width:1px

    class CPU active
    class GPU,ANE idle
```

### CPU — NEON SIMD

Every ARM64 core (performance and efficiency) has 32 x 128-bit SIMD vector
registers (V0–V31) implementing the NEON (Advanced SIMD) instruction set.
NEON vectorizes operations over 4 x FP32, 8 x FP16, 16 x INT8, etc.

**This is the only compute unit currently used by this project.** ONNX
Runtime's MLAS library contains hand-tuned NEON assembly kernels for GEMM,
convolution, softmax, pooling, and rotary embeddings.

### CPU — AMX (M1–M3) / SME (M4+)

The Apple Matrix coprocessor (AMX) is a proprietary extension tightly
coupled to each CPU cluster. It performs outer-product matrix
multiplication (`Z += X ⊗ Y`) using 512-bit registers at peak throughput
of ~1.6 TFLOPS FP32 on M1 at 3.2 GHz — roughly 3.7x faster than scalar
CPU for matrix-matrix multiply.

Key constraints:

| Aspect | Detail |
|--------|--------|
| Access path | Apple Accelerate framework only (M1–M3); no public intrinsics |
| M4+ | Uses ARM SME (Scalable Matrix Extension), an industry-standard ISA |
| MLAS | **Explicitly disabled** — `platform.cpp` guards AMX init with `#ifndef __APPLE__` |
| Status | **Idle in this project** |

### GPU — Metal Compute Shaders

Apple Silicon uses unified memory, eliminating PCIe copy overhead between
CPU and GPU. CoreML dispatches to the GPU via Metal Performance Shaders
(MPS). M4 chips add dedicated "Neural Accelerators" within the GPU die.

**Not currently used** — requires CoreML EP registration (see below).

### Neural Engine (ANE)

The ANE is a discrete NPU on the SoC targeting fixed-precision inference
(INT8, FP16). The M4 ANE delivers 38 TOPS across 16 cores — roughly 3x
faster than the GPU for compatible models.

Hard constraints:
- Static tensor shapes required — dynamic dimensions force CPU fallback
- Only accessible through CoreML
- **Not currently used** — requires CoreML EP registration

## Dependency Chain

The path from Rust source to ONNX inference traverses four crates before
reaching the prebuilt static library:

```mermaid
graph TD
    App["bge-m3-axum-fastembed-rs<br/><code>Cargo.toml</code>"]
    FE["fastembed = &quot;5&quot;<br/>features: default"]
    ORT["ort = &quot;=2.0.0-rc.11&quot;<br/>features: ndarray, std"]
    SYS["ort-sys = &quot;2.0.0-rc.9&quot;<br/>build.rs downloads binary"]
    LIB["libonnxruntime.a<br/>ORT 1.23.2 (pyke.io)<br/>aarch64-apple-darwin"]

    App -->|"depends on"| FE
    FE -->|"depends on"| ORT
    ORT -->|"depends on"| SYS
    SYS -->|"downloads at<br/>build time"| LIB

    classDef crate fill:#e8f0fe,stroke:#4285f4,stroke-width:2px
    classDef binary fill:#fce8e6,stroke:#ea4335,stroke-width:2px

    class App,FE,ORT,SYS crate
    class LIB binary
```

### The Prebuilt Binary

`ort-sys` downloads a precompiled static library from `cdn.pyke.io` at
build time. For `aarch64-apple-darwin`, this is ONNX Runtime 1.23.2
compiled with:

- **CPU EP** (default fallback) — always present
- **CoreML EP** — compiled in, 11,886+ CoreML symbols confirmed via `nm`
- **MLAS** — hand-tuned NEON assembly kernels for ARM64
- No CUDA, TensorRT, ROCM, or DirectML (as expected for macOS ARM64)

The dist URL is recorded in `ort-sys`'s `build/download/dist.txt`:

```
none  aarch64-apple-darwin  https://cdn.pyke.io/0/pyke:ort-rs/ms@1.23.2/aarch64-apple-darwin.tar.lzma2
```

### What Links at Build Time

When `cargo build --release --target aarch64-apple-darwin` runs, `ort-sys`'s
`build.rs` emits link directives. For `aarch64-apple-darwin` with the pyke
prebuilt, only Foundation is explicitly linked:

```
cargo:rustc-link-lib=framework=Foundation
```

However, because `libonnxruntime.a` contains CoreML EP object files that
reference CoreML symbols, the linker resolves these against the macOS SDK
frameworks. The resulting binary links:

| Framework | Why |
|-----------|-----|
| `CoreML.framework` | CoreML EP symbols in `libonnxruntime.a` |
| `Foundation.framework` | Explicitly linked by `ort-sys` build.rs |
| `CoreFoundation.framework` | Transitive dependency |
| `Security.framework` | TLS/certificate support |
| `SystemConfiguration.framework` | Network reachability |

This can be verified on the installed binary:

```bash
otool -L ~/.local/bin/bge-m3-apple
```

### CoreML EP — Present but Not Registered

This is the critical subtlety. The binary **contains** CoreML EP code and
**links** `CoreML.framework`, but the EP is **never registered** at runtime.

The chain of why:

1. `fastembed = "5"` depends on `ort` with features `["ndarray", "std"]` — no
   `coreml` feature
2. In the `ort` crate, `src/ep/coreml.rs` gates registration behind
   `#[cfg(any(feature = "load-dynamic", feature = "coreml"))]`
3. Without the feature flag, `register()` returns `Err(RegisterError::MissingFeature)`
4. fastembed never calls `register()` for CoreML anyway — it uses the default
   CPU EP

**Result:** All inference runs through the CPU EP using MLAS NEON kernels.
The CoreML symbols are dead code in the linked binary.

```mermaid
graph TD
    Session["ORT InferenceSession"]
    Partition["Graph Partitioner"]
    CoreML["CoreML EP<br/>(NOT registered)"]
    CPU["CPU EP<br/>(MLAS NEON kernels)"]
    AMX["Apple AMX<br/>(not accessed)"]
    ANE["Neural Engine<br/>(not accessed)"]

    Session --> Partition
    Partition -->|"All ops"| CPU
    Partition -.->|"Would dispatch<br/>if registered"| CoreML
    CoreML -.-> ANE
    CoreML -.-> AMX
    CPU -->|"NEON SIMD only"| Result["Embedding vectors"]

    classDef active fill:#6f6,stroke:#333,stroke-width:2px
    classDef inactive fill:#ccc,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5

    class Session,Partition,CPU,Result active
    class CoreML,AMX,ANE inactive
```

### MLAS on Apple Silicon

MLAS (Microsoft Linear Algebra Subprograms) is ONNX Runtime's built-in
BLAS-like library. It ships as part of `libonnxruntime.a` and provides:

- FP32 SGEMM (single-precision GEMM)
- FP16/HGEMM (half-precision GEMM)
- Quantized GEMM (INT8, Q4, QNBitGEMM)
- Convolution, softmax, pooling, rotary embeddings

For ARM64, MLAS includes NEON-optimized assembly kernels:

| Kernel | Source file |
|--------|------------|
| Half-precision GEMM | `hgemm_kernel_neon.cpp` |
| Quantized GEMM | `qgemm_kernel_neon.cpp` |
| FP32 cast | `cast_kernel_neon.cpp` |
| SQNBitGEMM | `sqnbitgemm_kernel_neon_fp32.cpp` |
| Rotary embedding | `rotary_embedding_kernel_neon.cpp` |

What MLAS does **not** use on macOS:

- **Apple Accelerate** — zero `cblas_*` or `vDSP_*` symbols in the binary
- **Apple AMX** — explicitly disabled with `#ifndef __APPLE__` guard in
  `platform.cpp`

MLAS implements its own optimized kernels rather than delegating to system
frameworks.

## Rust Release Profile

The project's `Cargo.toml` configures an aggressive release profile:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
panic = "abort"
```

### Setting Analysis

| Setting | Value | Effect |
|---------|-------|--------|
| `lto` | `"thin"` | Cross-crate inlining and optimization; parallelizable; ~80–90% of fat LTO's runtime gain without the severe compile-time penalty |
| `codegen-units` | `1` | Entire crate compiled as one LLVM module — maximizes intra-crate optimization, auto-vectorization, and constant propagation; often 5–15% runtime improvement over the default 16 units |
| `strip` | `"symbols"` | Removes symbol table; 30–60% binary size reduction on macOS; stack traces show hex addresses only |
| `panic` | `"abort"` | No unwinding tables; 5–15% binary size reduction; marginally tighter codegen without landing pads |

The resulting binary is ~22 MB (stripped, arm64 Mach-O).

### The `target-cpu` Gap

The project has **no** `.cargo/config.toml`. Builds use the default
`aarch64-apple-darwin` target features, which correspond to the M1 baseline.

On M2 and later, `target-cpu=native` unlocks three additional instruction
set extensions:

| Feature | Description | Added in |
|---------|-------------|----------|
| `i8mm` | INT8 Matrix Multiply — relevant for quantized ONNX ops | M2 (apple-a15) |
| `bf16` | BFloat16 — 16-bit brain float for ML inference | M2 (apple-a15) |
| `bti` | Branch Target Identification — security hardening | M2 (apple-a15) |

The full feature progression:

```mermaid
graph LR
    M1["M1 Baseline<br/>28 features<br/>neon, fp16, dotprod,<br/>sha2, sha3, aes, lse..."]
    M2["M2 Adds<br/>+bf16, +bti, +i8mm"]
    M3["M3<br/>(identical feature set)"]
    M4["M4<br/>Adds ARM SME<br/>(replaces AMX)"]

    M1 --> M2 --> M3 --> M4

    classDef chip fill:#e8f0fe,stroke:#4285f4,stroke-width:2px
    class M1,M2,M3,M4 chip
```

Note: Apple Silicon does **not** implement SVE or SVE2. The M-series chips
use fixed-width 128-bit NEON SIMD exclusively (until M4's SME).

### Recommended `.cargo/config.toml`

To unlock M2+ features for local builds without affecting CI or Docker:

```toml
# .cargo/config.toml
#
# Apple Silicon local builds — target-cpu=native enables i8mm, bf16, bti
# beyond the M1 baseline. This section is ignored when building for any
# other target triple (e.g., x86_64-unknown-linux-gnu in CI/Docker).
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]
```

**Why this is safe for CI:** GitHub Actions runners use
`x86_64-unknown-linux-gnu`. The Dockerfile builds for
`x86_64-unknown-linux-gnu` or `aarch64-unknown-linux-gnu`. Neither matches
`aarch64-apple-darwin`, so this section is never consulted.

**Historical note:** Rust versions before 1.82 (LLVM < 19) had a
[known bug](https://github.com/rust-lang/rust/issues/93889) where `native`
misidentified Apple Silicon as `cyclone` (A7 era). This was fixed in LLVM 19
(Rust 1.82, October 2024). Current rustc (1.93.0+) correctly resolves
`native` to the actual chip model.

## macOS Deployment

The project deploys on Apple Silicon Macs as a `launchd` UserAgent, managed
by the install script in the `dpos-ha-config` repository.

### Build and Install

The install script
([`dpos-ha-config/ops/install-bge-m3-service.sh`](https://github.com/Fulton-Engineering-Services/dpos-ha-config))
handles the full lifecycle:

```bash
# From the dpos-ha-config repo root:
./ops/install-bge-m3-service.sh
```

The script:

1. Verifies Apple Silicon (`uname -m == arm64`)
2. Builds from source if no binary is provided:
   ```bash
   cargo build --release --target aarch64-apple-darwin \
       --manifest-path ../bge-m3-axum-fastembed-rs/Cargo.toml
   ```
3. Installs the binary to `~/.local/bin/bge-m3-apple`
4. Creates the model cache directory at `~/.cache/bge-m3`
5. Creates the log directory at `~/Library/Logs/bge-m3-apple/`
6. Installs the `launchd` plist (substituting `__HOME__` paths)
7. Bootstraps the LaunchAgent via `launchctl bootstrap`
8. Runs a health check on port 8089

### LaunchAgent Configuration

The plist template
(`dpos-ha-config/config/bge-m3-apple/ai.bge-m3.server.plist`) configures:

| Setting | Value | Rationale |
|---------|-------|-----------|
| `BGE_M3_BIND` | `0.0.0.0:8089` | Port 8089 avoids conflict with llama-server (8088) |
| `BGE_M3_WORKERS` | `2` | Two worker threads, each with its own model instance |
| `BGE_M3_MAX_BATCH` | `256` | Maximum texts per request |
| `BGE_M3_IDLE_TIMEOUT_SECS` | `0` | Disabled — models stay resident permanently |
| `BGE_M3_CACHE_DIR` | `~/.cache/bge-m3` | Local model cache |
| `BGE_M3_LOG_FORMAT` | `json` | Structured logging for log aggregation |
| `RUST_LOG` | `info` | Log level |
| `KeepAlive` | `true` | Restart on crash |
| `RunAtLoad` | `true` | Start at login |
| `ThrottleInterval` | `10` | Minimum 10s between restart attempts |

Logs are written to `~/Library/Logs/bge-m3-apple/{stdout,stderr}.log`.

### Service Management

```bash
# Status
launchctl list ai.bge-m3.server

# Stop
launchctl bootout gui/$(id -u)/ai.bge-m3.server

# Restart
launchctl kickstart -k gui/$(id -u)/ai.bge-m3.server

# Logs
tail -f ~/Library/Logs/bge-m3-apple/stderr.log
```

## CoreML EP via Custom ORT Build (Current Production State)

CoreML EP is now enabled in production via a custom ORT build linked with
`ORT_LIB_LOCATION`.  This approach bypasses the pyke.io prebuilt entirely;
all inference runs through ORT built from source with
`-Donnxruntime_USE_COREML=ON`.

### ORT Bug Fix Required

The pyke.io prebuilt (and the upstream ORT release through v1.23.2) contains
a path-handling bug in `TensorProtoWithExternalDataToTensorProto` that causes
an immediate `ENOTDIR` crash when CoreML EP loads a model with external data
(`.onnx` + `.onnx_data` split format, as used by BGE-M3):

```
open file ".../onnx/model.onnx/model.onnx_data" failed: Not a directory
```

**Root cause:** the function receives `ModelPath()` (a file path, e.g.
`.../onnx/model.onnx`) but passes it directly to `ReadExternalDataForTensor`,
which expects a directory. `GetExternalDataInfo` then constructs
`model.onnx / model.onnx_data` — a path through a file — triggering ENOTDIR.

**Fix** (one logical line in `tensorprotoutils.cc`):

```cpp
// Derive the directory from model_path if it points to a file.
const auto tensor_proto_dir =
    model_path.has_filename() ? model_path.parent_path() : model_path;
ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(ten_proto, tensor_proto_dir, unpacked_data));
```

This mirrors the pattern used by every other external-data path in the same
file (`UnpackTensor`, `GetExtDataFromTensorProto`).  The bug is also present
in ORT `main` as of the time of writing.  An upstream PR against `main` is
planned (see Serena memory `ort-upstream-pr/coreml-external-data-path-fix`).

**Patched fork:**

| | |
|---|---|
| Fork | `https://github.com/Fulton-Engineering-Services/onnxruntime` |
| Branch | `fix/coreml-tensorproto-external-data-path` |
| Commit | `1e37c3583d05992bc1419269f87d941e8642248c` |
| Base tag | `v1.23.2` |

### Building the Custom ORT

```bash
# One-time: clone and configure (requires cmake 3.26+, Xcode CLT)
git clone --depth 1 --branch v1.23.2 \
  https://github.com/Fulton-Engineering-Services/onnxruntime.git \
  ~/.local/share/ort-build/onnxruntime
cd ~/.local/share/ort-build/onnxruntime
git checkout fix/coreml-tensorproto-external-data-path

mkdir -p ~/.local/share/ort-build/output && cd ~/.local/share/ort-build/output
cmake ../onnxruntime \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_BUILD_SHARED_LIB=OFF \
  -Donnxruntime_USE_COREML=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64

# Build (takes ~20–30 minutes on first run)
cmake --build . -j$(sysctl -n hw.logicalcpu)
```

### Building the Rust Service Against Custom ORT

```bash
ORT_LIB_LOCATION=~/.local/share/ort-build/output/Release \
  cargo build --release
```

The `ort-sys` build script detects the multi-library layout in the directory
and links against all individual `.a` files (onnxruntime_framework,
onnxruntime_session, onnxruntime_providers_coreml, protobuf-lite, onnx, etc.).
`ORT_LIB_LOCATION` is a **build-time** environment variable only; the
resulting binary is fully self-contained.

### Cargo Cache Gotcha

With `-C lto=thin`, Rust embeds ALL C++ object files from linked static
libraries into `libort_sys-*.rlib` (typically ~69–72 MB).  This rlib is
cached independently from the `.a` files.  If the custom ORT is rebuilt
(e.g. after patching), the stale rlib must be removed to force a relink:

```bash
rm target/release/deps/libort_sys-*.rlib
ORT_LIB_LOCATION=~/.local/share/ort-build/output/Release cargo build --release
```

Simply running `cargo clean -p ort-sys` does not remove the rlib; it must be
deleted directly from `target/release/deps/`.

## Future: Enabling CoreML EP (Standard Build Path)

Enabling the CoreML Execution Provider would allow ONNX Runtime to dispatch
model subgraphs to the Neural Engine (ANE) and GPU via Apple's CoreML
framework, rather than running everything through MLAS NEON kernels on the
CPU.

### What Would Change

```mermaid
graph TD
    Session["ORT InferenceSession"]
    Partition["Graph Partitioner"]
    CoreML["CoreML EP<br/>(registered)"]
    CPU["CPU EP<br/>(fallback)"]
    Compile["CoreML Model Compilation<br/>(at session creation)"]
    Dispatch["CoreML Dispatch"]
    ANE["Neural Engine"]
    GPU["GPU (Metal)"]
    CPUML["CPU (Accelerate → AMX)"]

    Session --> Partition
    Partition -->|"Supported ops"| CoreML
    Partition -->|"Unsupported ops"| CPU
    CoreML --> Compile
    Compile --> Dispatch
    Dispatch --> ANE
    Dispatch --> GPU
    Dispatch --> CPUML

    classDef active fill:#6f6,stroke:#333,stroke-width:2px
    classDef coreml fill:#e8f0fe,stroke:#4285f4,stroke-width:2px

    class Session,Partition,CoreML,Compile,Dispatch,ANE,GPU,CPUML,CPU active
    class CoreML,Compile,Dispatch coreml
```

CoreML's `MLComputeUnits` setting controls which hardware is available:

| Option | Dispatch targets |
|--------|-----------------|
| `All` (default) | ANE + GPU + CPU — CoreML chooses per-subgraph |
| `CPUAndNeuralEngine` | ANE + CPU |
| `CPUAndGPU` | GPU + CPU |
| `CPUOnly` | CPU only (via Accelerate → AMX) |

### Prerequisites

1. **Enable the `coreml` feature on `ort`** — fastembed-rs would need to
   depend on `ort` with `features = ["coreml"]`, or the feature would need
   to be enabled transitively

2. **Link `CoreML.framework` explicitly** — the pyke prebuilt's
   `static_link/mod.rs` only emits `framework=Foundation` for macOS (iOS
   gets CoreML automatically); a `build.rs` or Cargo config override would
   need to add `cargo:rustc-link-lib=framework=CoreML`

3. **Register the EP at session creation** — fastembed-rs would need to
   call `SessionBuilder::with_execution_providers([CoreMLExecutionProvider::default()])`
   before loading the model

### Known Risks

| Risk | Detail |
|------|--------|
| FP16 silent conversion | CoreML silently converts FP32 to FP16 when dispatching to GPU; this can affect embedding precision |
| Session creation overhead | ONNX-to-CoreML compilation happens at session creation, adding 5–15s startup cost per model unless `ModelCacheDirectory` is configured |
| Dynamic shapes | Any ONNX ops with dynamic dimensions fall back to CPU; BGE-M3's variable-length sequence handling may limit ANE eligibility |
| macOS version coupling | MLProgram format requires macOS 12+; NeuralNetwork format on macOS 10.15+ has limited op coverage |

### References

- [ONNX Runtime CoreML EP documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Apple MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [ort crate CoreML EP source](https://github.com/pykeio/ort) (`src/ep/coreml.rs`)
- [AMX reverse engineering (corsix/amx)](https://github.com/corsix/amx)
- [ONNX Runtime MLAS](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/mlas/lib)
- [Rust `native` CPU misdetection fix (rust-lang/rust#93889)](https://github.com/rust-lang/rust/issues/93889)
