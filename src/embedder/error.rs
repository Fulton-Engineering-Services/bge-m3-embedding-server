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

//! Error helpers for the embedder.

/// Converts an ort error to an anyhow error by formatting it as a string.
///
/// `ort::Error` became generic in rc.12 (`ort::Error<T>`) and its variants gained
/// `NonNull<>` pointer fields that are `!Send + !Sync`, preventing direct
/// `?`-propagation into `anyhow::Error` (which requires `Send + Sync + 'static`).
/// Accepting `impl Display` makes this helper work for all instantiations
/// (`ort::Error<()>`, `ort::Error<SessionBuilder>`, etc.) without naming the type.
pub(super) fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}
