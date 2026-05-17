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

//! **Destructive** stale-SM `TensorRT` engine plan garbage collection.
//!
//! # ⚠️ HAZARD — multi-SM ASG cache coexistence
//!
//! This module is gated behind the `cache-gc` Cargo feature for a reason:
//! ORT's `TensorRT` EP namespaces its engine plan filenames by compute
//! capability (`_sm75`, `_sm86`, `_sm89`, `_sm120`, …). Plans for the
//! "wrong" SM are simply never loaded by the runtime — there is no
//! correctness hazard from leaving them in place. Production fleets
//! deliberately rely on this property: an AWS ASG that mixes instance
//! families (T4 / A10G / L4 / L40S / Blackwell) shares a single EFS
//! engine cache so that any task can use any pre-compiled plan that
//! matches its hardware, and per-SM plans coexist by design.
//!
//! **A binary that runs `gc_stale_sm_plans` against a shared multi-SM
//! cache will delete plans that are still in active use by peer tasks
//! on different hardware.** The next time a peer needs a deleted plan
//! it pays a 30–170 s recompile and may hit autotuner OOM mid-build.
//!
//! For this reason the GC has **two independent gates**, both of which
//! must be on for the code to run:
//!
//! 1. **Compile gate** — the `cache-gc` Cargo feature. Production
//!    binaries built without the feature physically lack this module;
//!    `BGE_M3_TRT_CACHE_GC_ENABLED=1` is silently ignored.
//! 2. **Runtime gate** — `BGE_M3_TRT_CACHE_GC_ENABLED=1`. Defaults to
//!    `false` even when the feature is compiled in.
//!
//! Intended use: a dedicated maintenance / dev binary whose deployment
//! does not share a cache directory with any production task. A future
//! cache-maintenance tool with fleet-topology awareness will own the
//! shared-cache concern; this module is the minimum viable mechanism
//! for engineers cleaning up an obsolete cache on a workstation or a
//! lab host.
//!
//! ## What it does (when both gates are on)
//!
//! Scans the engine cache directory and deletes every `*.engine` file
//! whose `_smXX` suffix does not match `current_sm`, plus the aligned
//! `.engine.profile` / `.engine.timing_cache` sidecars. Files without
//! a recognizable `_smXX` suffix are conservatively preserved.
//!
//! The runtime log line always contains the substring
//! `"destructive cache GC ran"` so `CloudWatch` alerts can trip on it
//! without any further filter complexity.

use std::path::Path;

/// Sidecar suffixes ORT TRT writes alongside each `.engine` plan file.
///
/// Sidecar deletion is opportunistic: any of these whose basename
/// matches a deleted plan's prefix (up to `.engine`) is also deleted.
/// Unknown sidecars are left in place — they will be regenerated on
/// the next compile or are operator-managed.
pub(super) const ENGINE_SIDE_SUFFIXES: &[&str] = &[".profile", ".timing_cache"];

/// Counts and observations produced by a single GC sweep over the engine
/// cache directory. Logged at `WARN` by the caller so operators see
/// exactly how much disk was reclaimed and which SM tags were touched.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of `.engine` plan files deleted because their `_smXX` tag
    /// did not match the current device's compute capability.
    pub plans_deleted: usize,
    /// Total bytes freed (sum of deleted plan files + aligned sidecars).
    /// Best-effort — sidecars whose size cannot be stat'd are counted as
    /// zero rather than failing the sweep.
    pub bytes_freed: u64,
    /// Distinct SM tags observed on deleted plan files, in the order
    /// they were first encountered. Useful for operators correlating
    /// cache turnover with ASG instance-family changes.
    pub other_sms_observed: Vec<String>,
}

/// Scans `cache_path` for `.engine` plan files whose `_smXX` tag does
/// not match `current_sm` and deletes them along with their aligned
/// sidecars.
///
/// `current_sm` must be the literal cache-tag form (`"sm89"`, `"sm120"`).
/// Plans with no recognisable `_smXX` tag and plans matching `current_sm`
/// are preserved. Files that do not end in `.engine` (or in `.engine`
/// followed by a known sidecar suffix) are never touched. The function
/// does not recurse into subdirectories and is a silent no-op on a
/// missing directory.
///
/// # ⚠️ Hazard
///
/// Engine plan files are reproducible artifacts — the TRT EP will
/// recompile them on the next `session.run()`. **In a multi-SM ASG**,
/// however, those recompiles will fire on the peer tasks that owned
/// the deleted plans, not on the GC binary. Never call this against a
/// cache directory that is shared with production traffic. See the
/// module-level docs for the full hazard model.
#[allow(clippy::too_many_lines)]
pub fn gc_stale_sm_plans(cache_path: &Path, current_sm: &str) -> GcStats {
    let mut stats = GcStats::default();

    let read_dir = match std::fs::read_dir(cache_path) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::debug!(
                cache_path = %cache_path.display(),
                error = %e,
                "trt cache gc: cache directory missing or unreadable; skipping"
            );
            return stats;
        }
    };

    // Two-pass: collect engine plans first (so sidecar deletion can key
    // on the exact basename prefix), then walk sidecars in a second
    // `read_dir`. This keeps sidecar matching robust to enumeration
    // order.
    let mut deletions: Vec<DeletedPlan> = Vec::new();
    for entry in read_dir.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(stem) = name.strip_suffix(".engine") else {
            continue;
        };
        let Some(plan_sm) = extract_sm_tag(stem) else {
            // Plans without a recognisable `_smXX` tag are conservatively
            // preserved (see module docs).
            continue;
        };
        if plan_sm == current_sm {
            continue;
        }
        let path = entry.path();
        let plan_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
        match std::fs::remove_file(&path) {
            Ok(()) => {
                stats.plans_deleted += 1;
                stats.bytes_freed = stats.bytes_freed.saturating_add(plan_bytes);
                deletions.push(DeletedPlan {
                    basename_prefix: stem.to_string(),
                });
                if !stats.other_sms_observed.iter().any(|s| s == &plan_sm) {
                    stats.other_sms_observed.push(plan_sm);
                }
            }
            Err(e) => {
                tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "trt cache gc: failed to delete stale engine plan; \
                     subsequent runs will retry"
                );
            }
        }
    }

    if deletions.is_empty() {
        return stats;
    }

    // Second pass: delete aligned sidecars. We re-open the directory in
    // case the first sweep changed enumeration order.
    let Ok(read_dir) = std::fs::read_dir(cache_path) else {
        return stats;
    };
    for entry in read_dir.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some((prefix_with_engine, _suffix)) = ENGINE_SIDE_SUFFIXES
            .iter()
            .find_map(|s| name.strip_suffix(s).map(|p| (p, *s)))
        else {
            continue;
        };
        let Some(prefix) = prefix_with_engine.strip_suffix(".engine") else {
            continue;
        };
        let matches_deleted = deletions.iter().any(|d| d.basename_prefix == prefix);
        if !matches_deleted {
            continue;
        }
        let path = entry.path();
        let sidecar_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
        match std::fs::remove_file(&path) {
            Ok(()) => {
                stats.bytes_freed = stats.bytes_freed.saturating_add(sidecar_bytes);
            }
            Err(e) => {
                tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "trt cache gc: failed to delete aligned sidecar"
                );
            }
        }
    }

    stats
}

/// Cache-side bookkeeping carried between the plan pass and the sidecar pass.
struct DeletedPlan {
    basename_prefix: String,
}

/// Extracts the `smXX` tag immediately preceding `.engine` in a plan
/// basename's stem (the substring before the `.engine` extension).
///
/// Returns `None` for stems that do not end in `_sm<digits>` — those
/// filenames are conservatively preserved by [`gc_stale_sm_plans`] (see
/// module docs).
fn extract_sm_tag(stem: &str) -> Option<String> {
    let idx = stem.rfind("_sm")?;
    let after_underscore = &stem[idx + 1..]; // strip leading '_'
    if after_underscore.len() < 3 {
        return None;
    }
    // The tag is `sm` followed by purely numeric digits, ending at the
    // string end. Reject anything with trailing non-digit qualifiers
    // (e.g. `sm89-rev3`) so future TRT versions cannot trick us into
    // deleting current-SM plans.
    let digits = &after_underscore[2..];
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(after_underscore.to_string())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parse_tests {
    use super::extract_sm_tag;

    #[test]
    fn extract_sm_tag_recognises_standard_filenames() {
        assert_eq!(
            extract_sm_tag("TensorrtExecutionProvider_TRTKernel_graph_a_111_fp16_sm89"),
            Some("sm89".to_string())
        );
        assert_eq!(extract_sm_tag("eng_sm120"), Some("sm120".to_string()));
        assert_eq!(extract_sm_tag("eng_sm75"), Some("sm75".to_string()));
    }

    #[test]
    fn extract_sm_tag_rejects_qualified_or_malformed_tags() {
        assert_eq!(extract_sm_tag("eng_sm89-rev3"), None);
        assert_eq!(extract_sm_tag("eng_smABC"), None);
        assert_eq!(extract_sm_tag("eng_no_tag"), None);
        assert_eq!(extract_sm_tag("eng_sm"), None);
        assert_eq!(extract_sm_tag(""), None);
    }
}
