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

use axum::http::{header, HeaderValue};

use super::super::common::{check_ready, collect_x_headers, validate_input, MAX_STRING_CHARS};
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

// ── collect_x_headers ─────────────────────────────────────────────────────

#[test]
fn collect_x_headers_collects_x_prefix_and_skips_standard() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::HeaderName::from_static("x-foo"),
        HeaderValue::from_static("bar"),
    );
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );

    let result = collect_x_headers(&headers);
    assert!(!result.is_empty());
    // Key is normalized: hyphen → underscore
    assert_eq!(result.0.get("x_foo").map(String::as_str), Some("bar"));
    assert_eq!(result.0.len(), 1, "content-type must be excluded");
}

#[test]
fn collect_x_headers_empty_map_produces_empty() {
    let result = collect_x_headers(&axum::http::HeaderMap::new());
    assert!(result.is_empty());
}

#[test]
fn collect_x_headers_no_x_prefix_produces_empty() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer tok"),
    );
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from_static("42"));
    let result = collect_x_headers(&headers);
    assert!(result.is_empty());
}

#[test]
fn collect_x_headers_normalizes_hyphens_to_underscores() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::HeaderName::from_static("x-codekeeper-project"),
        HeaderValue::from_static("my-project"),
    );
    headers.insert(
        axum::http::HeaderName::from_static("x-request-id"),
        HeaderValue::from_static("abc123"),
    );
    let result = collect_x_headers(&headers);
    // Normalized keys — safe as JSON identifiers and log-path components
    assert_eq!(
        result.0.get("x_codekeeper_project").map(String::as_str),
        Some("my-project")
    );
    assert_eq!(
        result.0.get("x_request_id").map(String::as_str),
        Some("abc123")
    );
    // Original hyphenated keys must NOT be present
    assert!(!result.0.contains_key("x-codekeeper-project"));
}

#[test]
fn collect_x_headers_multiple_x_headers_sorted() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::HeaderName::from_static("x-request-id"),
        HeaderValue::from_static("abc"),
    );
    headers.insert(
        axum::http::HeaderName::from_static("x-project"),
        HeaderValue::from_static("my-proj"),
    );
    let result = collect_x_headers(&headers);
    let keys: Vec<&str> = result.0.keys().map(String::as_str).collect();
    // BTreeMap guarantees alphabetical order; keys are underscore-normalized
    assert_eq!(keys, vec!["x_project", "x_request_id"]);
}

#[test]
fn collect_x_headers_skips_non_utf8_values() {
    let mut headers = axum::http::HeaderMap::new();
    let non_utf8 = HeaderValue::from_bytes(&[0x66, 0x6f, 0x80, 0x6f]).expect("valid header bytes");
    headers.insert(
        axum::http::HeaderName::from_static("x-bad-encoding"),
        non_utf8,
    );
    // Non-UTF-8 header silently skipped — map is empty
    let result = collect_x_headers(&headers);
    assert!(result.is_empty());
}
