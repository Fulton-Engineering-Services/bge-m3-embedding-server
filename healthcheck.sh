#!/bin/sh
# Copyright (c) 2025-2026 Lockbox AI, Inc.
# All rights reserved.
#
# Dual-mode container HEALTHCHECK.
#
# The server runtime selects HTTPS vs HTTP based on whether both
# BGE_M3_TLS_CERT_PATH and BGE_M3_TLS_KEY_PATH are set (see Cargo.toml `tls`
# feature notes). This script makes the same decision so the same Dockerfile
# (and resulting image) works for both `EXTRA_FEATURES=""` and
# `EXTRA_FEATURES=tls` builds without needing two parallel Dockerfiles.
#
# When TLS is active the leaf cert is self-signed (issued from the internal
# root CA at task start by the CDK entrypoint preamble), so `-k` is required
# to skip CN/CA verification against the OS trust store.

set -eu

URL_PATH="/health/deep"
TIMEOUT=12
HOST="127.0.0.1:8081"

if [ -n "${BGE_M3_TLS_CERT_PATH:-}" ] && [ -n "${BGE_M3_TLS_KEY_PATH:-}" ]; then
    exec curl -kfsS --max-time "${TIMEOUT}" -o /dev/null "https://${HOST}${URL_PATH}"
else
    exec curl -fsS --max-time "${TIMEOUT}" -o /dev/null "http://${HOST}${URL_PATH}"
fi
