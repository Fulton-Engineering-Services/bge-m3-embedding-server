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

# hawkeye 6.5.1: license-header checker — pinned to match the version in CI
ARG HAWKEYE_VERSION=6.5.1
RUN curl --proto '=https' --tlsv1.2 -sSfL \
    "https://github.com/korandoru/hawkeye/releases/download/v${HAWKEYE_VERSION}/hawkeye-x86_64-unknown-linux-musl.tar.xz" \
    -o /tmp/hawkeye.tar.xz \
    && echo "9a6454f7f4f587764b4cbcbb865026526bbadeda477f1ef0b65924af777fb172  /tmp/hawkeye.tar.xz" | sha256sum -c \
    && tar -xJf /tmp/hawkeye.tar.xz -C /tmp \
    && mv "/tmp/hawkeye-x86_64-unknown-linux-musl/hawkeye" /usr/local/bin/hawkeye \
    && rm -rf /tmp/hawkeye.tar.xz "/tmp/hawkeye-x86_64-unknown-linux-musl" \
    && hawkeye --version
