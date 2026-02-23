FROM ubuntu:24.04@sha256:d1e2e92c075e5ca139d51a140fff46f84315c0fdce203eab2807c7e495eff4f9 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Cache dependency compilation by building a dummy binary first
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM ubuntu:24.04@sha256:d1e2e92c075e5ca139d51a140fff46f84315c0fdce203eab2807c7e495eff4f9
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3t64 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --no-create-home --shell /sbin/nologin bge

COPY --from=builder /app/target/release/bge-m3-axum-fastembed-rs /usr/local/bin/
USER bge

HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD /bin/bash -c 'echo > /dev/tcp/localhost/8081' 2>/dev/null || exit 1

EXPOSE 8081
CMD ["bge-m3-axum-fastembed-rs"]
