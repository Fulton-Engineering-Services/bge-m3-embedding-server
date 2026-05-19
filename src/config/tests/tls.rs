// Copyright (c) 2026 J. Patrick Fulton
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Tests for `BGE_M3_TLS_CERT_PATH` and `BGE_M3_TLS_KEY_PATH` config fields
//! and the half-config guard in [`Config::validate`].

use std::collections::HashMap;
use std::path::PathBuf;

use super::super::Config;
use super::helpers::lookup_from;

#[test]
fn both_tls_paths_set_parse_to_some_pathbuf() {
    let map = HashMap::from([
        ("BGE_M3_TLS_CERT_PATH", "/certs/server.crt"),
        ("BGE_M3_TLS_KEY_PATH", "/certs/server.key"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(
        cfg.tls_cert_path,
        Some(PathBuf::from("/certs/server.crt")),
        "cert path must be Some when env var is set"
    );
    assert_eq!(
        cfg.tls_key_path,
        Some(PathBuf::from("/certs/server.key")),
        "key path must be Some when env var is set"
    );
    cfg.validate()
        .expect("both paths set — validate must succeed");
}

#[test]
fn only_cert_path_set_validate_returns_err() {
    let map = HashMap::from([("BGE_M3_TLS_CERT_PATH", "/certs/server.crt")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let err = cfg
        .validate()
        .expect_err("half-config (cert only) must be rejected");
    assert!(
        err.to_string().contains("BGE_M3_TLS_CERT_PATH"),
        "error message must name the missing env var pair; got: {err}"
    );
}

#[test]
fn only_key_path_set_validate_returns_err() {
    let map = HashMap::from([("BGE_M3_TLS_KEY_PATH", "/certs/server.key")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let err = cfg
        .validate()
        .expect_err("half-config (key only) must be rejected");
    assert!(
        err.to_string().contains("BGE_M3_TLS_KEY_PATH"),
        "error message must name the missing env var pair; got: {err}"
    );
}

#[test]
fn neither_tls_path_set_both_none_no_error() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert!(
        cfg.tls_cert_path.is_none(),
        "cert path must be None when env var is absent"
    );
    assert!(
        cfg.tls_key_path.is_none(),
        "key path must be None when env var is absent"
    );
    cfg.validate()
        .expect("neither path set — validate must succeed");
}
