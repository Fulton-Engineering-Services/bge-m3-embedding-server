//! Axum router construction with the request-id, tracing, and body-limit
//! layers attached.

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::HeaderValue;
use axum::{routing::get, routing::post, Router};
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::{DefaultOnFailure, DefaultOnResponse, MakeSpan, TraceLayer};
use tracing::Level;

use crate::handler;
use crate::state::AppState;

#[derive(Clone, Default)]
pub(super) struct UuidRequestId;

impl MakeRequestId for UuidRequestId {
    fn make_request_id<B>(&mut self, _request: &axum::http::Request<B>) -> Option<RequestId> {
        let id = uuid::Uuid::new_v4().to_string();
        HeaderValue::from_str(&id).ok().map(RequestId::new)
    }
}

/// Selects the tracing level for HTTP spans based on path.
///
/// `/health` and `/v1/models` are polled frequently by load balancers and the
/// Docker `HEALTHCHECK`. Logging them at DEBUG rather than INFO keeps
/// `CloudWatch` free of ~8,640 health-check records per container per day.
#[derive(Clone)]
pub(super) struct RouteAwareSpan;

impl<B> MakeSpan<B> for RouteAwareSpan {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> tracing::Span {
        let path = request.uri().path();
        let is_noisy = matches!(path, "/health" | "/v1/models");
        let method = request.method().as_str();
        if is_noisy {
            tracing::debug_span!(
                "http_request",
                method = method,
                uri = %request.uri(),
                version = ?request.version(),
            )
        } else {
            tracing::info_span!(
                "http_request",
                method = method,
                uri = %request.uri(),
                version = ?request.version(),
            )
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(handler::dense_embeddings))
        .route("/v1/sparse-embeddings", post(handler::sparse_embeddings))
        // The colon in `/v1/embeddings:both` is a valid `pchar` per RFC 3986
        // §3.3, but some HTTP clients (and URI builders) percent-encode it
        // anyway when it appears in a path segment. The router is built on
        // `matchit`, which matches the raw URI path byte-for-byte, so the
        // encoded forms are registered as alias routes pointing at the same
        // handler. RFC 3986 percent-encoding is case-insensitive, hence both
        // upper- and lowercase aliases.
        .route("/v1/embeddings:both", post(handler::both_embeddings))
        .route("/v1/embeddings%3Aboth", post(handler::both_embeddings))
        .route("/v1/embeddings%3aboth", post(handler::both_embeddings))
        .route("/v1/models", get(handler::models))
        .route("/health", get(handler::health))
        .layer(DefaultBodyLimit::max(2_097_152))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(RouteAwareSpan)
                .on_response(
                    DefaultOnResponse::new()
                        .level(Level::INFO)
                        .latency_unit(tower_http::LatencyUnit::Millis),
                )
                .on_failure(DefaultOnFailure::new().level(Level::ERROR)),
        )
        .layer(SetRequestIdLayer::x_request_id(UuidRequestId))
        .with_state(state)
}
