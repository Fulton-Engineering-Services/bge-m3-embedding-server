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

use super::super::super::types::SparseEmbedding;
use super::super::EmbedPool;

// ── channel-closed error paths ────────────────────────────────────────

#[tokio::test]
async fn dense_returns_error_when_channel_closed() {
    let pool = EmbedPool::closed_for_test();
    let result = pool.dense(vec!["hello".into()]).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("channel closed"));
}

#[tokio::test]
async fn sparse_returns_error_when_channel_closed() {
    let pool = EmbedPool::closed_for_test();
    let result = pool.sparse(vec!["hello".into()]).await;
    let err = result.expect_err("expected an error");
    assert!(err.to_string().contains("channel closed"));
}

#[tokio::test]
async fn both_returns_error_when_channel_closed() {
    let pool = EmbedPool::closed_for_test();
    let result = pool.both(vec!["hello".into()]).await;
    let err = result.expect_err("expected an error");
    assert!(err.to_string().contains("channel closed"));
}

// ── fixture-pool round-trip ────────────────────────────────────────────

#[tokio::test]
async fn both_returns_paired_dense_and_sparse_from_fixture() {
    let dense_fixture = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let sparse_fixture = vec![
        SparseEmbedding {
            indices: vec![10],
            values: vec![0.7],
        },
        SparseEmbedding {
            indices: vec![20, 30],
            values: vec![0.8, 0.9],
        },
    ];
    let pool = EmbedPool::with_fixed_responses(dense_fixture.clone(), sparse_fixture.clone());
    let (result, _stats) = pool
        .both(vec!["a".into(), "b".into()])
        .await
        .expect("both should succeed against fixture pool");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].dense, dense_fixture[0]);
    assert_eq!(result[1].dense, dense_fixture[1]);
    assert_eq!(result[0].sparse.indices, sparse_fixture[0].indices);
    assert_eq!(result[1].sparse.indices, sparse_fixture[1].indices);
}
