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
use std::path::Path;
use tempfile::TempDir;

use super::super::fsync::fsync_cache_dir;

/// fsync sweep must succeed on a populated directory and leave file
/// contents intact. We can't assert the kernel actually flushed to disk
/// without root-level tooling, but exercising the syscall paths catches
/// permission / handle bugs that would silently degrade durability.
#[cfg(target_os = "linux")]
#[test]
fn fsync_cache_dir_walks_populated_directory() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("a.engine"), b"plan-a").unwrap();
    fs::write(dir.join("b.engine"), b"plan-b").unwrap();

    // No panic, no error log — and contents preserved afterwards.
    fsync_cache_dir(&dir);

    assert_eq!(fs::read(dir.join("a.engine")).unwrap(), b"plan-a");
    assert_eq!(fs::read(dir.join("b.engine")).unwrap(), b"plan-b");
}

/// On Linux, fsync of an empty directory must not panic. The TRT cache
/// directory is empty between `ensure_and_inspect` and the first
/// engine compile — the sweep must be safe to call in that window.
#[cfg(target_os = "linux")]
#[test]
fn fsync_cache_dir_handles_empty_directory() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("trt-engines");
    fs::create_dir_all(&dir).unwrap();

    fsync_cache_dir(&dir);

    // Directory still exists and is still empty.
    let entries: Vec<_> = fs::read_dir(&dir).unwrap().collect();
    assert_eq!(entries.len(), 0);
}

/// On Linux, fsync of a missing directory must not panic (logs a WARN).
#[cfg(target_os = "linux")]
#[test]
fn fsync_cache_dir_missing_directory_does_not_panic() {
    let tmp = TempDir::new().expect("tempdir");
    let missing = tmp.path().join("never-created");

    fsync_cache_dir(&missing);
}
