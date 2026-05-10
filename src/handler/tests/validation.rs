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

use super::super::common::{check_ready, validate_input, MAX_STRING_CHARS};
use super::helpers::make_state;
use crate::error::AppError;

// ── validate_input ────────────────────────────────────────────────────────

#[test]
fn validate_input_rejects_empty() {
    let result = validate_input(&[], 10);
    assert!(
        matches!(result, Err(AppError::InvalidRequest(msg)) if msg == "input must not be empty")
    );
}

#[test]
fn validate_input_rejects_over_batch() {
    let texts: Vec<String> = (0..5).map(|i| format!("text {i}")).collect();
    let result = validate_input(&texts, 3);
    assert!(
        matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains('5') && msg.contains('3'))
    );
}

#[test]
fn validate_input_accepts_at_limit() {
    let texts: Vec<String> = (0..3).map(|i| format!("text {i}")).collect();
    assert!(validate_input(&texts, 3).is_ok());
}

#[test]
fn validate_input_accepts_single() {
    let texts = vec!["hello".to_string()];
    assert!(validate_input(&texts, 256).is_ok());
}

#[test]
fn validate_input_rejects_oversized_string() {
    let long = "x".repeat(MAX_STRING_CHARS + 1);
    let texts = vec![long];
    let result = validate_input(&texts, 256);
    assert!(
        matches!(result, Err(AppError::InvalidRequest(msg)) if msg.contains("exceeds maximum"))
    );
}

#[test]
fn validate_input_accepts_at_char_limit() {
    let at_limit = "x".repeat(MAX_STRING_CHARS);
    let texts = vec![at_limit];
    assert!(
        validate_input(&texts, 256).is_ok(),
        "string exactly at MAX_STRING_CHARS should be accepted"
    );
}

// ── check_ready ───────────────────────────────────────────────────────────

#[test]
fn check_ready_returns_err_when_not_ready() {
    let state = make_state(false, 10);
    let result = check_ready(&state);
    assert!(matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "model not ready"));
}

#[test]
fn check_ready_returns_err_when_pool_dead() {
    let state = make_state(true, 10);
    let result = check_ready(&state);
    assert!(
        matches!(result, Err(AppError::ServiceUnavailable(msg)) if msg == "no workers available")
    );
}
