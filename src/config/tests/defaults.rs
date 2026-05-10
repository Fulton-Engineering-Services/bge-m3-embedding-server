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

use std::collections::HashMap;
use std::time::Duration;

use super::super::{Config, ModelVariant, MODEL_MAX_SEQ};
use super::helpers::lookup_from;

#[test]
fn defaults_without_env_vars() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));

    assert_eq!(cfg.cache_dir, "/cache");
    assert_eq!(cfg.bind_addr, "0.0.0.0:8081");
    assert_eq!(cfg.workers, 2);
    assert_eq!(cfg.intra_threads, 1);
    assert_eq!(cfg.max_batch, 256);
    assert_eq!(cfg.max_seq_length, MODEL_MAX_SEQ);
    assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(300)));
    assert_eq!(cfg.model_variant, ModelVariant::Fp16);
    assert!((cfg.memory_safety_factor - 0.7).abs() < 1e-9);
    assert!(
        cfg.cost_model_override.is_none(),
        "probe should run by default"
    );
    assert_eq!(cfg.heartbeat_secs, 60);
}

#[test]
fn workers_clamps_to_minimum_1() {
    let map = HashMap::from([("BGE_M3_WORKERS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.workers, 1);
}

#[test]
fn intra_threads_defaults_to_1() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 1);
}

#[test]
fn intra_threads_custom_value() {
    let map = HashMap::from([("BGE_M3_INTRA_THREADS", "4")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 4);
}

#[test]
fn intra_threads_clamps_to_minimum_1() {
    let map = HashMap::from([("BGE_M3_INTRA_THREADS", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 1);
}

#[test]
fn intra_threads_invalid_falls_back_to_default() {
    let map = HashMap::from([("BGE_M3_INTRA_THREADS", "not_a_number")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.intra_threads, 1);
}

#[test]
fn max_batch_clamps_to_minimum_1() {
    let map = HashMap::from([("BGE_M3_MAX_BATCH", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_batch, 1);
}

#[test]
fn max_seq_length_default_is_8192() {
    let map = HashMap::new();
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, 8192);
}

#[test]
fn max_seq_length_custom() {
    let map = HashMap::from([("BGE_M3_MAX_SEQ_LENGTH", "2048")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, 2048);
}

#[test]
fn max_seq_length_clamps_out_of_range() {
    // Over max
    let map = HashMap::from([("BGE_M3_MAX_SEQ_LENGTH", "99999")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, MODEL_MAX_SEQ);

    // Zero
    let map = HashMap::from([("BGE_M3_MAX_SEQ_LENGTH", "0")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    assert_eq!(cfg.max_seq_length, MODEL_MAX_SEQ);
}

#[test]
fn custom_values_are_applied() {
    let map = HashMap::from([
        ("BGE_M3_CACHE_DIR", "/tmp/models"),
        ("BGE_M3_BIND", "127.0.0.1:9090"),
        ("BGE_M3_WORKERS", "4"),
        ("BGE_M3_MAX_BATCH", "128"),
        ("BGE_M3_IDLE_TIMEOUT_SECS", "600"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));

    assert_eq!(cfg.cache_dir, "/tmp/models");
    assert_eq!(cfg.bind_addr, "127.0.0.1:9090");
    assert_eq!(cfg.workers, 4);
    assert_eq!(cfg.max_batch, 128);
    assert_eq!(cfg.idle_timeout, Some(Duration::from_secs(600)));
}
