FROM ubuntu:24.04@sha256:c4a8d5503dfb2a3eb8ab5f807da5bc69a85730fb49b5cfca2330194ebcc41c7b AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download rustup-init, verify the official SHA-256 sidecar, then install.
# Using TARGETARCH so both amd64 and arm64 native CI runners work.
ARG TARGETARCH
ARG RUSTUP_VERSION=1.29.0
RUN ARCH=$([ "$TARGETARCH" = "arm64" ] && echo "aarch64-unknown-linux-gnu" || echo "x86_64-unknown-linux-gnu") \
    && BASE="https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${ARCH}" \
    && curl --proto '=https' --tlsv1.2 -sSf "${BASE}/rustup-init"        -o /tmp/rustup-init \
    && curl --proto '=https' --tlsv1.2 -sSf "${BASE}/rustup-init.sha256"  -o /tmp/rustup-init.sha256 \
    && cd /tmp && sha256sum -c rustup-init.sha256 \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --default-toolchain stable \
    && rm /tmp/rustup-init /tmp/rustup-init.sha256
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Cache dependency compilation by building a dummy binary first.
# benches/ stubs are required for all [[bench]] targets because Cargo validates source
# paths at manifest parse time, even though cargo build --release skips bench compilation.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs \
    && mkdir benches && touch benches/embeddings.rs benches/coreml.rs \
    && cargo build --release \
    && rm -rf src benches

COPY src ./src
COPY benches ./benches
RUN touch src/main.rs && cargo build --release

FROM ubuntu:24.04@sha256:c4a8d5503dfb2a3eb8ab5f807da5bc69a85730fb49b5cfca2330194ebcc41c7b
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libssl3t64 \
    && rm -rf /var/lib/apt/lists/*
# UID/GID must match EFS model-cache access point in Codekeeper CDK (10002).
RUN groupadd --gid 10002 bge \
    && useradd --uid 10002 --gid bge --no-create-home --shell /sbin/nologin bge

COPY --from=builder /app/target/release/bge-m3-embedding-server /usr/local/bin/
USER bge

HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD curl --fail --silent http://localhost:8081/health || exit 1

EXPOSE 8081
CMD ["bge-m3-embedding-server"]
