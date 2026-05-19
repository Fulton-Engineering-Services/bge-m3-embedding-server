# TLS Feature Guide

Native TLS support for `bge-m3-embedding-server` — optional HTTPS via `axum-server` and
`tokio-rustls` (Rustls backend; no OpenSSL dependency).

## Contents

1. [Overview](#overview)
2. [Enabling TLS](#enabling-tls)
3. [Certificate Provisioning](#certificate-provisioning)
4. [TLS Behavior](#tls-behavior)
5. [Health Checks with TLS](#health-checks-with-tls)
6. [Service-to-Service Trust (bge-router)](#service-to-service-trust-bge-router)
7. [Environment Variable Reference](#environment-variable-reference)
8. [Build and CI](#build-and-ci)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The `tls` Cargo feature adds optional native TLS to the server. When activated, HTTPS is
served over the same bind address (`BGE_M3_BIND`, default `0.0.0.0:8081`) using:

- **`axum-server`** — Axum-native TLS listener with graceful shutdown support.
- **`tokio-rustls` / Rustls** — Pure-Rust TLS implementation; no system OpenSSL required.
- **`aws-lc-rs`** — Crypto backend pulled in by `axum-server/tls-rustls`; requires `cmake`
  and a C compiler at build time (see [Build-time requirement](#build-time-requirement)).

Two conditions must **both** be satisfied for HTTPS to activate:

| Condition | Default when absent |
|-----------|---------------------|
| Binary compiled with `--features tls` | Feature excluded → plain HTTP only |
| Both `BGE_M3_TLS_CERT_PATH` and `BGE_M3_TLS_KEY_PATH` set at runtime | Env vars absent → plain HTTP |

When either condition is absent the server falls back to plain HTTP automatically. Setting
**only one** of the two env vars is a hard startup error (see [Validation](#tls-behavior)).

---

## Enabling TLS

### Build-time requirement

Compile the binary with `--features tls`. This pulls in `axum-server` and the `aws-lc-sys`
C library crate, which requires `cmake` and a C compiler:

```bash
# Local / macOS (cmake ships with Xcode Command Line Tools)
cargo build --features tls

# Cloud / CI environment (download-ort required for ORT)
cargo build --features download-ort,tls
```

Without `--features tls` the TLS code path is physically absent from the binary, and the
`BGE_M3_TLS_*` env vars have no effect.

### Runtime requirements

Set **both** PEM-file env vars before starting the server:

```bash
BGE_M3_TLS_CERT_PATH=/tls/leaf.crt
BGE_M3_TLS_KEY_PATH=/tls/leaf.key
```

The files are read once at startup. The paths can point to any location the process has
read access to — container bind-mounts, Secrets Manager sidecar writes, etc.

Setting only one of the two variables causes a hard startup error:

```
TLS misconfiguration: BGE_M3_TLS_CERT_PATH and BGE_M3_TLS_KEY_PATH must both be set or both be absent
```

This fail-fast design prevents the server from silently ignoring a half-configured TLS setup
and binding plain HTTP when HTTPS was expected.

---

## Certificate Provisioning

### AWS ECS with shared internal CA (production)

The CDK entrypoint preamble for the ECS task definition generates a short-lived leaf
certificate from the shared internal CA stored in Secrets Manager as `LOCKBOX_TLS_CA_JSON`.
The preamble runs `openssl` at container startup and writes the cert and key to a shared
`tmpfs` volume:

```bash
# Run by the ECS entrypoint preamble before the server binary starts
openssl req -new -newkey rsa:2048 -nodes \
  -subj "/CN=${HOSTNAME}" \
  -keyout /tls/leaf.key \
  -out /tls/leaf.csr
openssl x509 -req -in /tls/leaf.csr \
  -CA /tls/ca.crt -CAkey /tls/ca.key -CAcreateserial \
  -days 1 -out /tls/leaf.crt
```

The CDK task definition injects `BGE_M3_TLS_CERT_PATH=/tls/leaf.crt` and
`BGE_M3_TLS_KEY_PATH=/tls/leaf.key` as environment variables. The `ca.crt` is written to
`/tls/ca.crt` for use by the router (see
[Service-to-Service Trust](#service-to-service-trust-bge-router)).

### Local development / testing

Generate a self-signed certificate with `openssl`:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout /tmp/leaf.key \
  -out /tmp/leaf.crt \
  -subj "/CN=localhost"
```

Then start the server:

```bash
BGE_M3_TLS_CERT_PATH=/tmp/leaf.crt \
BGE_M3_TLS_KEY_PATH=/tmp/leaf.key \
BGE_M3_CACHE_DIR=/tmp/bge-m3-cache \
BGE_M3_DISABLE_AUTO_BUDGET=1 \
cargo run --features download-ort,tls
```

---

## TLS Behavior

- **Rustls defaults**: TLS 1.2 minimum, TLS 1.3 preferred, AEAD-only cipher suites. No
  configuration knob required or available — Rustls enforces a secure profile out of the box.
- **Cert loaded once at startup**: cert and key files are read by `RustlsConfig::from_pem_file`
  before the listener binds. A mismatch between the cert and key (or a corrupt PEM file)
  produces a hard startup error:
  ```
  TLS config error: <rustls error description>
  ```
  Cert rotation requires a container restart — there is no live reload.
- **Graceful shutdown**: the server registers an `axum_server::Handle`. On `SIGINT` (Ctrl-C
  or ECS SIGTERM forwarded to the process), the handle triggers a 30-second drain window
  (`h.graceful_shutdown(Some(Duration::from_secs(30)))`). In-flight requests are allowed to
  complete; new connections are refused during the drain.
- **Startup log**: look for `mode = "tls"` in the `Listening` log line to confirm HTTPS is
  active. Plain HTTP emits `mode = "plain"`.

  ```json
  {"level":"INFO","message":"Listening","bind":"0.0.0.0:8081","mode":"tls"}
  ```

---

## Health Checks with TLS

ALB target groups and ECS container health checks must use HTTPS when TLS is active.

**TCP-level health check** — the existing bash `/dev/tcp` probe in the container definition
operates at the TCP layer and remains valid regardless of TLS:

```bash
# Works with both plain and TLS — tests that the port accepts connections
bash -c 'echo > /dev/tcp/127.0.0.1/8081'
```

**`curl`-based health check** — add `-k` to skip certificate verification for self-signed
or internal-CA-signed certs:

```bash
curl -sfk https://127.0.0.1:8081/health
```

For production ALB health checks against certs signed by a trusted CA, omit `-k` and ensure
the ALB's HTTPS target group is configured with the correct CA bundle or ACM certificate.

---

## Service-to-Service Trust (bge-router)

The `bge-router` connects to `bge-m3` upstreams over HTTP by default. When `bge-m3` is
running with TLS enabled and a leaf certificate signed by the shared internal CA, the router
must be told to use HTTPS and to trust that CA:

```bash
# On the router side
BGE_ROUTER_UPSTREAM_TLS=1
BGE_ROUTER_UPSTREAM_CA_BUNDLE=/tls/ca.crt
```

The CA bundle path must point to the same `ca.crt` used to sign the bge-m3 leaf cert. In
the ECS deployment pattern this is written to the same `/tls/` tmpfs volume on the router
task.

If `BGE_ROUTER_UPSTREAM_CA_BUNDLE` is set without `BGE_ROUTER_UPSTREAM_TLS=1`, the router
logs a WARN and ignores the bundle. See the
[bge-router CLAUDE.md](../../bge-router/CLAUDE.md) for the full router env var reference.

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BGE_M3_TLS_CERT_PATH` | unset | Path to TLS certificate PEM file. When set together with `BGE_M3_TLS_KEY_PATH` and the server is built with `--features tls`, the server binds HTTPS instead of HTTP. Both must be set or both must be absent — setting only one is a startup error. |
| `BGE_M3_TLS_KEY_PATH` | unset | Path to TLS private key PEM file. Must be set together with `BGE_M3_TLS_CERT_PATH`; see above. |

These are the only two new env vars introduced by this feature. All existing variables
(`BGE_M3_BIND`, `BGE_M3_WORKERS`, etc.) continue to work unchanged.

---

## Build and CI

The `tls` feature is **off by default**. Existing plain-HTTP builds and Docker images
(`Dockerfile`, `Dockerfile.cuda`) are completely unaffected.

The CI `test-tls` job (`.github/workflows/ci.yml`) compiles and tests the feature on every
push and pull request to `main`:

```yaml
test-tls:
  name: Test (tls feature)
  runs-on: ubuntu-latest
  steps:
    - name: Install cmake (required by aws-lc-sys)
      run: sudo apt-get install -y cmake

    - name: Clippy (tls)
      run: cargo clippy --all-targets --features "download-ort tls" -- -D warnings

    - name: Run tests (tls)
      run: cargo nextest run --features "download-ort tls" --no-tests=warn
```

`cmake` is installed explicitly because `aws-lc-sys` (the crypto backend) compiles a
bundled C library. On macOS `cmake` is provided by the Xcode Command Line Tools.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `TLS misconfiguration: BGE_M3_TLS_CERT_PATH and BGE_M3_TLS_KEY_PATH must both be set or both be absent` at startup | Only one of the two env vars is set | Set both vars, or unset both to revert to plain HTTP |
| `TLS config error: …` at startup | Invalid PEM file, corrupt cert/key, or cert and key are not a matched pair | Verify the files are valid PEM (`openssl verify -CAfile ca.crt leaf.crt`; `openssl rsa -check -in leaf.key`) and that the cert was signed with the corresponding key |
| Server starts in `mode = "plain"` despite env vars being set | Binary was not compiled with `--features tls` | Rebuild with `--features tls` (or `--features download-ort,tls` in the cloud environment) |
| `cmake: command not found` during build | `cmake` not installed | `sudo apt-get install -y cmake` (Linux) or `brew install cmake` (macOS) — required by `aws-lc-sys` |
| `curl: (60) SSL certificate problem` | Client does not trust the server's CA | Use `-k` for self-signed certs, or pass `--cacert /path/to/ca.crt` for internal-CA-signed certs |
| ALB health check returns unhealthy after enabling TLS | ALB target group is still configured for HTTP | Update the ALB target group protocol to HTTPS and configure the health check path accordingly |
