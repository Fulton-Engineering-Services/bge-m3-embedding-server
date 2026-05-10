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

//! Application-level error types that map to HTTP status codes.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tracing::error;

/// Application-level errors that map to HTTP status codes.
#[derive(Debug)]
pub enum AppError {
    /// The request was malformed or violates input constraints.
    /// Maps to HTTP 400 Bad Request.
    InvalidRequest(String),
    /// The service is not yet ready (model loading) or has no live workers.
    /// Maps to HTTP 503 Service Unavailable.
    ServiceUnavailable(String),
    /// An unexpected internal error occurred during embedding.
    /// Maps to HTTP 500 Internal Server Error.
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, code, message) = match self {
            AppError::InvalidRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                400u16,
                msg,
            ),
            AppError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                503u16,
                msg,
            ),
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                500u16,
                msg,
            ),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": code
            }
        });

        (status, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        error!(error = %err, "Internal error");
        AppError::Internal("internal server error".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::response::IntoResponse;

    async fn response_parts(err: AppError) -> (StatusCode, serde_json::Value) {
        let response = err.into_response();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read body");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("body is not valid JSON");
        (status, body)
    }

    #[tokio::test]
    async fn invalid_request_serializes_as_400() {
        let (status, body) =
            response_parts(AppError::InvalidRequest("bad input".to_string())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], 400);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "bad input");
    }

    #[tokio::test]
    async fn service_unavailable_serializes_as_503() {
        let (status, body) =
            response_parts(AppError::ServiceUnavailable("model not ready".to_string())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], 503);
        assert_eq!(body["error"]["type"], "service_unavailable");
        assert_eq!(body["error"]["message"], "model not ready");
    }

    #[tokio::test]
    async fn internal_error_serializes_as_500() {
        let (status, body) =
            response_parts(AppError::Internal("unexpected failure".to_string())).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["code"], 500);
        assert_eq!(body["error"]["type"], "internal_error");
        assert_eq!(body["error"]["message"], "unexpected failure");
    }

    #[tokio::test]
    async fn from_anyhow_error_produces_generic_message() {
        let err = anyhow::anyhow!("secret path /var/models/onnx failed to load");
        let app_err: AppError = err.into();
        let (status, body) = response_parts(app_err).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body["error"]["message"], "internal server error",
            "internal details must not leak to client"
        );
    }
}
