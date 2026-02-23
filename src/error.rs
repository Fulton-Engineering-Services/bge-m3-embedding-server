use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Application-level error type that maps to HTTP responses.
#[derive(Debug)]
pub enum AppError {
    InvalidRequest(String),
    ServiceUnavailable(String),
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
        AppError::Internal(err.to_string())
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
}
