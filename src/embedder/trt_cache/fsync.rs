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

//! Post-compile cache durability via directory fsync.

use std::path::Path;

#[cfg(target_os = "linux")]
pub(crate) fn fsync_cache_dir(dir: &Path) {
    use std::fs::File;
    let read_dir = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                cache_path = %dir.display(),
                error = %e,
                "trt cache: could not enumerate directory for fsync; \
                 engine plan files may not be durable on the next OOM"
            );
            return;
        }
    };

    let mut synced = 0usize;
    for entry in read_dir.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        match File::open(&path).and_then(|f| f.sync_all()) {
            Ok(()) => synced += 1,
            Err(e) => {
                tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "trt cache: fsync(file) failed; engine plan may not be durable"
                );
            }
        }
    }

    match File::open(dir).and_then(|d| d.sync_all()) {
        Ok(()) => {
            tracing::debug!(
                cache_path = %dir.display(),
                files_synced = synced,
                "trt cache: directory fsynced"
            );
        }
        Err(e) => {
            tracing::warn!(
                cache_path = %dir.display(),
                error = %e,
                "trt cache: fsync(directory) failed; name → inode mapping \
                 may not be durable on the next OOM"
            );
        }
    }
}

/// Non-Linux no-op stub so callers do not need to gate every call site.
#[cfg(not(target_os = "linux"))]
pub(crate) fn fsync_cache_dir(_dir: &Path) {}
