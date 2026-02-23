FROM rust:1.88-bookworm AS builder
WORKDIR /app

# Cache dependency compilation by building a dummy binary first
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/bge-m3-axum-fastembed-rs /usr/local/bin/

EXPOSE 8081
CMD ["bge-m3-axum-fastembed-rs"]
