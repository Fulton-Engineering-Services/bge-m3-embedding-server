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

use std::fs;
use tempfile::TempDir;

use super::super::prewarm_log::{
    log_engine_basenames_before_prewarm, log_engine_basenames_before_prewarm_for_sm,
};

#[test]
fn prewarm_log_empty_directory_sm_filtered() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();
    log_engine_basenames_before_prewarm_for_sm(&dir, Some("sm89"));
}

#[test]
fn prewarm_log_lists_matching_engines() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("a_sm89.engine"), b"").unwrap();
    fs::write(dir.join("b_sm120.engine"), b"").unwrap();
    log_engine_basenames_before_prewarm_for_sm(&dir, Some("sm89"));
}

#[test]
fn prewarm_log_truncates_large_lists() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();
    for i in 0..70usize {
        fs::write(dir.join(format!("plan_{i}_sm89.engine")), b"").unwrap();
    }
    log_engine_basenames_before_prewarm_for_sm(&dir, Some("sm89"));
}

#[test]
fn prewarn_log_unfiltered_wrapper_delegates() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();
    log_engine_basenames_before_prewarm(&dir);
}

#[test]
fn prewarm_log_missing_directory_does_not_panic() {
    let tmp = TempDir::new().expect("tempdir");
    let missing = tmp.path().join("missing-trt-engines");
    log_engine_basenames_before_prewarm_for_sm(&missing, None);
}
