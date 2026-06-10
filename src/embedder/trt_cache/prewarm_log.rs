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

//! Operator-visible prewarm basename logging.

use std::path::Path;

use super::enumerate::engine_basenames_for_sm;

pub(crate) fn log_engine_basenames_before_prewarm_for_sm(engine_dir: &Path, sm: Option<&str>) {
    const MAX_LIST: usize = 64;
    let basenames = match engine_basenames_for_sm(engine_dir, sm) {
        Ok(v) => v,
        Err(e) => {
            tracing::info!(
                cache_path = %engine_dir.display(),
                detected_sm = sm.unwrap_or("unfiltered"),
                error = %e,
                "trt prewarm: could not read engine cache directory for basename listing"
            );
            return;
        }
    };

    let matching_total = basenames.len();
    let mut listed = basenames;
    let truncated = matching_total > MAX_LIST;
    if truncated {
        listed.truncate(MAX_LIST);
    }
    // Also surface the unfiltered total so heterogeneous-SM situations are
    // obvious at a glance. When `sm` is `None` this equals `matching_total`.
    let unfiltered_total = if sm.is_some() {
        engine_basenames_for_sm(engine_dir, None).map_or(matching_total, |v| v.len())
    } else {
        matching_total
    };

    if matching_total == 0 {
        tracing::info!(
            cache_path = %engine_dir.display(),
            detected_sm = sm.unwrap_or("unfiltered"),
            engine_basename_count = 0,
            engine_basename_total_count = unfiltered_total,
            "trt prewarm: cache engine basenames (none for this SM)"
        );
    } else {
        tracing::info!(
            cache_path = %engine_dir.display(),
            detected_sm = sm.unwrap_or("unfiltered"),
            engine_basename_count = matching_total,
            engine_basename_total_count = unfiltered_total,
            truncated,
            engine_basenames = ?listed,
            "trt prewarm: cache engine basenames (for operator correlation; not shape-specific)"
        );
    }
}

/// Backwards-compatible wrapper for the unfiltered prewarm basename log.
///
/// Delegates to [`log_engine_basenames_before_prewarm_for_sm`] with
/// `sm = None` — equivalent to today's behaviour for callers that have not
/// yet been updated to pass the worker's detected SM.
pub(crate) fn log_engine_basenames_before_prewarm(engine_dir: &Path) {
    log_engine_basenames_before_prewarm_for_sm(engine_dir, None);
}
