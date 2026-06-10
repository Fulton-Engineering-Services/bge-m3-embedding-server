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

//! Tests for abandoned-request observability helpers.

use super::super::logging::log_if_abandoned_mid_flight;
use super::helpers::sample_embed_stats;

#[test]
fn abandoned_mid_flight_warns_when_receiver_dropped_on_success() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    drop(rx);
    let stats = sample_embed_stats(100, 256);
    let result: anyhow::Result<(Vec<Vec<f32>>, _)> = Ok((vec![], stats));
    log_if_abandoned_mid_flight(&tx, "dense", 2, &result, 100);
}

#[test]
fn abandoned_mid_flight_warns_when_receiver_dropped_on_error() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    drop(rx);
    let result: anyhow::Result<(Vec<Vec<f32>>, _)> = Err(anyhow::anyhow!("inference failed"));
    log_if_abandoned_mid_flight(&tx, "both", 1, &result, 50);
}

#[test]
fn abandoned_mid_flight_noop_when_receiver_alive() {
    let (tx, _rx) = tokio::sync::oneshot::channel();
    let stats = sample_embed_stats(10, 64);
    let result: anyhow::Result<(Vec<Vec<f32>>, _)> = Ok((vec![], stats));
    log_if_abandoned_mid_flight(&tx, "sparse", 0, &result, 10);
}
