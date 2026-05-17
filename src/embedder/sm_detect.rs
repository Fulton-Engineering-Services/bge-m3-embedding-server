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

//! Per-device GPU compute-capability detection for SM-aware TRT cache filtering.
//!
//! ORT names every TRT engine plan with a `_smXX` suffix tied to the GPU's
//! compute capability (`sm75` = T4, `sm86` = A10G, `sm89` = L40S/L4, `sm120`
//! = Blackwell). Plans built for one SM cannot be loaded by another — the TRT
//! runtime silently refuses them and JIT-compiles instead. Filtering the
//! engine cache by the worker's own SM is the only way to produce a truthful
//! `cache_hit` signal on heterogeneous-SM fleets (or on fresh hosts where a
//! previous-SM cache survives on EFS).
//!
//! Detection mechanism: shell out to
//! `nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i <device_id>`.
//! The subprocess is cheap (single-digit ms) and avoids pulling in a CUDA
//! driver crate that the project does not otherwise need. The parser is a
//! pure function so the bulk of the surface area can be unit-tested without
//! a GPU.
//!
//! Failure modes (missing `nvidia-smi`, non-zero exit, parse error) return
//! `None` and the caller falls back to the legacy unfiltered behaviour so
//! operators rolling forward mid-deploy never see a hard regression.

/// Returns the GPU compute capability as a `smXY` string for the given CUDA
/// device, or `None` if detection fails for any reason.
///
/// The format is the same one ORT uses in its engine plan basenames (e.g.
/// `_sm120.engine`), so the return value can be plugged directly into the
/// SM-filtered enumerators in [`super::trt_cache`].
///
/// This function is a thin wrapper around `nvidia-smi`. It is intended to be
/// called once per worker at TRT prewarm time and the result cached; do not
/// call it on every request.
///
/// Failure-mode contract:
///
/// * `nvidia-smi` binary missing or unexecutable → `None`
/// * subprocess exits non-zero → `None`
/// * stdout cannot be parsed as `"X.Y"` → `None`
///
/// All failure cases are non-panicking. The caller is expected to log a
/// `WARN` and proceed with `None` semantics (no SM filter applied).
#[must_use]
pub(crate) fn detect_sm_for_device(device_id: u32) -> Option<String> {
    let device_arg = device_id.to_string();
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
            "-i",
            &device_arg,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = std::str::from_utf8(&output.stdout).ok()?;
    parse_compute_capability(stdout)
}

/// Parses `nvidia-smi --query-gpu=compute_cap` stdout (`"X.Y\n"` or
/// `"X.Y\nX.Y\n…"` for multi-device queries without `-i`) into an
/// ORT-compatible `smXY` string.
///
/// Defensive against extra whitespace, trailing newlines, and accidental
/// multi-line output (only the first non-empty line is consumed). Returns
/// `None` when:
///
/// * the input is empty after trimming;
/// * the first line is not exactly two dot-separated digit components
///   (`"X.Y"`, allowing 1+ digits each so `"12.0"` parses to `"sm120"`).
///
/// Pure: no I/O, no globals. The exhaustive correctness tests in the
/// sibling test module exercise every failure shape so the production
/// invariant ("strict `X.Y` only") cannot drift.
pub(crate) fn parse_compute_capability(stdout: &str) -> Option<String> {
    let first_line = stdout.lines().find(|l| !l.trim().is_empty())?.trim();
    let (major, minor) = first_line.split_once('.')?;
    if major.is_empty() || minor.is_empty() {
        return None;
    }
    if !major.chars().all(|c| c.is_ascii_digit()) || !minor.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("sm{major}{minor}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── parse_compute_capability ─────────────────────────────────────────

    #[test]
    fn parses_l40s_compute_cap() {
        assert_eq!(parse_compute_capability("8.9\n"), Some("sm89".to_string()));
    }

    #[test]
    fn parses_t4_compute_cap() {
        assert_eq!(parse_compute_capability("7.5\n"), Some("sm75".to_string()));
    }

    #[test]
    fn parses_a10g_compute_cap() {
        assert_eq!(parse_compute_capability("8.6\n"), Some("sm86".to_string()));
    }

    /// Blackwell sm120 = compute capability 12.0. The major component is
    /// two digits, the minor is one: `"sm120"` (not `"sm12_0"` or `"sm1200"`).
    #[test]
    fn parses_blackwell_compute_cap() {
        assert_eq!(
            parse_compute_capability("12.0\n"),
            Some("sm120".to_string())
        );
    }

    /// No trailing newline (some `nvidia-smi` versions omit it on `-i` query).
    #[test]
    fn parses_without_trailing_newline() {
        assert_eq!(parse_compute_capability("8.9"), Some("sm89".to_string()));
    }

    #[test]
    fn parses_with_surrounding_whitespace() {
        assert_eq!(
            parse_compute_capability("  8.9  \n"),
            Some("sm89".to_string())
        );
    }

    /// Multi-line stdout (operator forgot `-i`); we take only the first
    /// non-empty line so we never return data for the wrong device.
    #[test]
    fn parses_only_first_non_empty_line() {
        assert_eq!(
            parse_compute_capability("\n8.9\n12.0\n"),
            Some("sm89".to_string())
        );
    }

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(parse_compute_capability(""), None);
        assert_eq!(parse_compute_capability("\n\n\n"), None);
        assert_eq!(parse_compute_capability("   "), None);
    }

    #[test]
    fn input_without_dot_returns_none() {
        assert_eq!(parse_compute_capability("89\n"), None);
        assert_eq!(parse_compute_capability("garbage\n"), None);
    }

    #[test]
    fn input_with_non_digit_components_returns_none() {
        assert_eq!(parse_compute_capability("8.x\n"), None);
        assert_eq!(parse_compute_capability("x.9\n"), None);
        assert_eq!(parse_compute_capability("a.b\n"), None);
    }

    /// A trailing third component (`"8.9.0"`) would silently lose the third
    /// part if we split on the first dot only. The parser must reject it so
    /// we never construct a bogus SM string from malformed input.
    ///
    /// `split_once('.')` returns `("8", "9.0")`; the minor part fails the
    /// all-digits check, so the parser returns `None`.
    #[test]
    fn input_with_three_components_returns_none() {
        assert_eq!(parse_compute_capability("8.9.0\n"), None);
    }

    #[test]
    fn input_with_empty_component_returns_none() {
        assert_eq!(parse_compute_capability(".9\n"), None);
        assert_eq!(parse_compute_capability("8.\n"), None);
        assert_eq!(parse_compute_capability(".\n"), None);
    }

    // ─── detect_sm_for_device (subprocess wrapper) ────────────────────────

    /// On any host without `nvidia-smi` (macOS dev box, CPU-only CI runner,
    /// CPU EP build) the wrapper must return `None` rather than panicking
    /// or hanging. The whole degrade-safely path depends on this: the worker
    /// logs a WARN and proceeds with the unfiltered (legacy) behaviour.
    ///
    /// This test is intentionally NOT gated on platform — the function must
    /// be safe to call on every CI target.
    #[test]
    fn detect_returns_none_when_nvidia_smi_unavailable() {
        // On hosts WITH nvidia-smi (e.g. CI runners with a GPU) this test
        // would return `Some(...)`. That is also a correct outcome — the
        // contract is "either Some valid sm string, or None"; never panic.
        let result = detect_sm_for_device(0);
        match result {
            None => {}
            Some(s) => {
                assert!(s.starts_with("sm"), "got {s}; expected sm-prefixed");
                assert!(s.len() >= 4, "got {s}; expected at least 'sm75'");
            }
        }
    }
}
