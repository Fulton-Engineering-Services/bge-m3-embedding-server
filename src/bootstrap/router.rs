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

//! Axum router construction with the request-id, tracing, and body-limit
//! layers attached.

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::HeaderValue;
use axum::{Router, routing::get, routing::post};
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::{DefaultOnFailure, DefaultOnResponse, MakeSpan, TraceLayer};
use tracing::Level;

use crate::handler;
use crate::state::AppState;

/// [`MakeRequestId`] implementation that assigns a random UUID v4 to every
/// incoming request, attached as the `x-request-id` header.
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

/// Builds the Axum [`Router`] with all embedding, health, and fleet-discovery
/// routes, a configurable body limit (default 32 MiB), request-id propagation,
/// and structured tracing.
pub fn build_router(state: Arc<AppState>, max_body_bytes: usize) -> Router {
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
        .layer(DefaultBodyLimit::max(max_body_bytes))
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
