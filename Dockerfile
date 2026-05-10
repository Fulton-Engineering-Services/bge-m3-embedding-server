FROM ubuntu:24.04@sha256:c4a8d5503dfb2a3eb8ab5f807da5bc69a85730fb49b5cfca2330194ebcc41c7b AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates build-essential pkg-config libssl-dev python3 \
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

# Download the pinned ORT prebuilt static library, verify SHA-256, then extract.
# The archive is a raw-LZMA2-compressed tar produced by pyke.io; Python's lzma
# module handles FORMAT_RAW (dict_size 64 MiB) — no XZ container, no standard
# compression tool can decode it without FORMAT_RAW + filter specification.
# Checksums taken from ort-sys-2.0.0-rc.12/build/download/dist.txt (feature set "none").
ARG ORT_AMD64_URL=https://cdn.pyke.io/0/pyke:ort-rs/ms@1.24.2/x86_64-unknown-linux-gnu.tar.lzma2
ARG ORT_AMD64_SHA256=acc1cba79c337594ead1d88ca72516147aa60054c84217b53399a31caa5ba671
ARG ORT_ARM64_URL=https://cdn.pyke.io/0/pyke:ort-rs/ms@1.24.2/aarch64-unknown-linux-gnu.tar.lzma2
ARG ORT_ARM64_SHA256=7e4f5fec4494cbf578c4e28082b0229c42f735523f584259028dde96acf3b092
RUN case "$TARGETARCH" in \
        arm64) ORT_URL="$ORT_ARM64_URL" ORT_SHA256="$ORT_ARM64_SHA256" ;; \
        *)     ORT_URL="$ORT_AMD64_URL" ORT_SHA256="$ORT_AMD64_SHA256" ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -sSfL "$ORT_URL" -o /tmp/ort.tar.lzma2 \
    && echo "${ORT_SHA256}  /tmp/ort.tar.lzma2" | sha256sum -c - \
    && python3 -c "import lzma, tarfile, io, os; data = open('/tmp/ort.tar.lzma2', 'rb').read(); dec = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'dict_size': 1<<26}]); os.makedirs('/opt/ort', exist_ok=True); tf = tarfile.open(fileobj=io.BytesIO(dec)); tf.extractall('/opt/ort', filter='data'); tf.close()" \
    && rm /tmp/ort.tar.lzma2
ENV ORT_LIB_LOCATION=/opt/ort

WORKDIR /app

# Cache dependency compilation by building a dummy binary first.
# benches/ stubs are required for all [[bench]] targets because Cargo validates source
# paths at manifest parse time, even though cargo build --release skips bench compilation.
COPY Cargo.toml Cargo.lock build.rs ./
RUN mkdir src && echo "fn main(){}" > src/main.rs && touch src/lib.rs \
    && mkdir -p benches/coreml && touch benches/embeddings.rs benches/coreml/main.rs \
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
