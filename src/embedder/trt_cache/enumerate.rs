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

//! SM-aware engine plan enumeration and counting.

use std::path::{Path, PathBuf};

/// Returns `true` when `name` is a TRT engine plan basename whose `_smXX`
/// suffix matches the requested SM exactly.
///
/// ORT names every TRT engine plan with a `_smXX.engine` suffix tied to the
/// GPU compute capability that built it (`sm75` = T4, `sm86` = A10G, `sm89` =
/// L40S/L4, `sm120` = Blackwell). The match must be **strict** — `sm12` must
/// not match `sm120.engine`, or a B200 worker would happily believe a Hopper
/// plan is usable. We accomplish this by anchoring on the leading underscore:
/// the suffix tested is `"_{sm}.engine"`, so `_sm12.engine` and `_sm120.engine`
/// occupy disjoint string sets.
///
/// Pure, no I/O — easy to unit-test.
#[must_use]
pub(crate) fn matches_sm_suffix(name: &str, sm: &str) -> bool {
    let suffix = format!("_{sm}.engine");
    name.ends_with(&suffix)
}

/// Returns full paths of `.engine` files under `engine_dir` that match the
/// requested SM, or all `.engine` files when `sm` is `None`.
///
/// Single source of truth for engine enumeration: every other function in
/// this module that needs to enumerate engine plans (count, basenames,
/// operator log line) delegates here. Centralising the filter is the design
/// constraint behind the SM-aware refactor — a future caller that grew its
/// own `read_dir` loop would silently regress the heterogeneous-SM safety
/// invariant.
///
/// Failure modes (directory missing or unreadable) collapse to an empty
/// `Vec`, mirroring the legacy `count_engine_files` behaviour and the
/// "operator-visible, not load-bearing" stance of this module.
pub(crate) fn engine_files_for_sm(engine_dir: &Path, sm: Option<&str>) -> Vec<PathBuf> {
    let Ok(read_dir) = std::fs::read_dir(engine_dir) else {
        return Vec::new();
    };
    read_dir
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".engine") {
                return None;
            }
            match sm {
                Some(target) if !matches_sm_suffix(&name, target) => None,
                _ => Some(e.path()),
            }
        })
        .collect()
}

/// Counts `.engine` files in `dir` that match the requested SM.
///
/// Pass `Some("sm120")` to count only Blackwell plans; pass `None` to count
/// every `.engine` file regardless of suffix (legacy behaviour, also what
/// the wrapper [`count_engine_files`] does). Returns `0` when the directory
/// does not exist or cannot be read.
///
/// Crate-visible (not `pub(super)`) because the warmup-only postcondition
/// in `lib.rs` calls it directly to apply the same SM filter as the
/// per-worker prewarm path.
pub(crate) fn count_engine_files_for_sm(dir: &Path, sm: Option<&str>) -> usize {
    engine_files_for_sm(dir, sm).len()
}

/// Counts every `.engine` file in `dir` regardless of `_smXX` suffix.
///
/// Backwards-compatible wrapper around [`count_engine_files_for_sm`] with
/// `sm = None`. Retained because the operator-visible startup cache log
/// (`trt cache: found cached engines` in [`log_cache_state`]) reports the
/// total disk footprint, not the SM-filtered subset. Callers that drive
/// the prewarm postcondition use the SM-aware variant directly.
pub(crate) fn count_engine_files(dir: &Path) -> usize {
    count_engine_files_for_sm(dir, None)
}

/// Returns sorted basenames of `.engine` files under `engine_dir` that match
/// the requested SM, or all `.engine` basenames when `sm` is `None`.
///
/// Used for operator-visible logs before TRT prewarm; filenames are **not**
/// a reliable `(batch, seq)` key for dynamic models (see `CLAUDE.md`).
pub(crate) fn engine_basenames_for_sm(
    engine_dir: &Path,
    sm: Option<&str>,
) -> std::io::Result<Vec<String>> {
    // We re-implement the read instead of delegating to `engine_files_for_sm`
    // because the latter swallows I/O errors (returns empty on missing dir),
    // whereas this function needs to propagate them so the operator log can
    // surface "could not read engine cache directory for basename listing".
    let read_dir = std::fs::read_dir(engine_dir)?;
    let mut basenames: Vec<String> = read_dir
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".engine") {
                return None;
            }
            match sm {
                Some(target) if !matches_sm_suffix(&name, target) => None,
                _ => Some(name),
            }
        })
        .collect();
    basenames.sort();
    Ok(basenames)
}

/// Returns sorted basenames of every `.engine` file under `engine_dir`,
/// regardless of `_smXX` suffix.
///
/// Backwards-compatible wrapper around [`engine_basenames_for_sm`] with
/// `sm = None`. Retained for the unit tests below; production prewarm log
/// emission goes through [`log_engine_basenames_before_prewarm_for_sm`].
pub(crate) fn engine_basenames_in_dir_sorted(engine_dir: &Path) -> std::io::Result<Vec<String>> {
    engine_basenames_for_sm(engine_dir, None)
}
