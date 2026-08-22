FROM ubuntu:24.04

# System packages needed for Rust + ORT build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable with rustfmt + clippy (matches CI toolchain)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable \
       --component rustfmt,clippy
ENV PATH="/root/.cargo/bin:${PATH}"

# cargo-binstall: fast binary tool installer used for subsequent steps
RUN curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh \
    | bash

# cargo-nextest (test runner) and cargo-deny (supply-chain audit) — match CI
RUN cargo binstall -y cargo-nextest cargo-deny

# hawkeye: license-header checker — install the latest release, matching CI
RUN cargo binstall -y hawkeye && hawkeye --version
