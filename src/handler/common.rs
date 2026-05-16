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

//! Shared input validation, header utilities, and service-readiness helpers
//! used by all handlers.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::Ordering;

use axum::http::HeaderMap;
use serde::Serialize;

use crate::error::AppError;
use crate::state::AppState;

/// A sorted map of `X-*` HTTP request headers.
///
/// Keys are lowercase-normalized header names (e.g. `"x-request-id"`).
/// Values are UTF-8-decoded header values; headers with non-UTF-8 values
/// are silently skipped.
///
/// Serializes as a plain JSON object so it can be embedded as the
/// `x_headers` field in structured log events.
#[derive(Default, Serialize)]
#[serde(transparent)]
pub(super) struct XHeaders(pub(super) BTreeMap<String, String>);

impl XHeaders {
    /// Returns `true` when no `X-*` headers were present in the request.
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for XHeaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Compact JSON — suitable for both text and JSON log formats.
        match serde_json::to_string(&self.0) {
            Ok(s) => f.write_str(&s),
            Err(_) => f.write_str("{}"),
        }
    }
}

/// Collects all headers whose name starts with `x-` (case-insensitive) into
/// an [`XHeaders`] map.
///
/// Header names are stored in their lowercase-normalized form (axum's
/// [`HeaderMap`] already lowercases all names). Headers whose values are
/// not valid UTF-8 are silently skipped.
pub(super) fn collect_x_headers(headers: &HeaderMap) -> XHeaders {
    let mut map = BTreeMap::new();
    for (name, value) in headers {
        let name_str = name.as_str();
        if name_str.starts_with("x-") {
            if let Ok(val) = value.to_str() {
                map.insert(name_str.to_owned(), val.to_owned());
            }
        }
    }
    XHeaders(map)
}

/// Extracts the `x-codekeeper-project` header value if present and UTF-8.
pub(super) fn codekeeper_project(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-codekeeper-project")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

/// Maximum characters allowed per individual input string (SEC-3).
pub(super) const MAX_STRING_CHARS: usize = 32_768;

/// Validates a batch of input texts against size and length constraints.
///
/// Returns [`AppError::InvalidRequest`] if:
/// - `texts` is empty
/// - `texts.len() > max_batch`
/// - any individual text exceeds [`MAX_STRING_CHARS`] characters
pub(super) fn validate_input(texts: &[String], max_batch: usize) -> Result<(), AppError> {
    if texts.is_empty() {
        return Err(AppError::InvalidRequest(
            "input must not be empty".to_string(),
        ));
    }
    if texts.len() > max_batch {
        return Err(AppError::InvalidRequest(format!(
            "batch size {} exceeds maximum {}",
            texts.len(),
            max_batch
        )));
    }
    for (i, text) in texts.iter().enumerate() {
        let char_count = text.chars().count();
        if char_count > MAX_STRING_CHARS {
            return Err(AppError::InvalidRequest(format!(
                "input[{i}] length {char_count} exceeds maximum {MAX_STRING_CHARS} characters"
            )));
        }
    }
    Ok(())
}

/// Checks whether the service is ready to handle embedding requests.
pub(super) fn check_ready(state: &AppState) -> Result<(), AppError> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(AppError::ServiceUnavailable("model not ready".to_string()));
    }
    if state.pool.live_worker_count() == 0 {
        return Err(AppError::ServiceUnavailable(
            "no workers available".to_string(),
        ));
    }
    Ok(())
}
