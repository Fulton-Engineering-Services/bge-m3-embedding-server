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

use std::path::PathBuf;
use tempfile::TempDir;

use super::super::inspect::log_cache_state;
use super::super::paths::{TrtCacheInfo, engine_cache_path};

#[test]
fn log_cache_state_empty_emits_info() {
    let tmp = TempDir::new().expect("tempdir");
    let path = engine_cache_path(tmp.path());
    log_cache_state(&TrtCacheInfo {
        path: path.clone(),
        engine_count: 0,
        profile_count: 0,
    });
}

#[test]
fn log_cache_state_nonempty_emits_info() {
    let info = TrtCacheInfo {
        path: PathBuf::from("/cache/trt-engines"),
        engine_count: 3,
        profile_count: 2,
    };
    log_cache_state(&info);
}

#[test]
fn log_cache_state_create_dir_failure_path() {
    let info = TrtCacheInfo {
        path: PathBuf::from("/dev/null/impossible/trt-engines"),
        engine_count: 0,
        profile_count: 0,
    };
    log_cache_state(&info);
}
