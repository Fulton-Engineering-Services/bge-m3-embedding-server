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

use std::sync::atomic::Ordering;

use crate::error::AppError;
use crate::state::AppState;

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
