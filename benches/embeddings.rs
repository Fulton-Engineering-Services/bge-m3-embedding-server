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

// Micro-benchmarks for pure-compute hot paths (E6 backlog item).
//
// This is a [[bin]] crate, so private types (TextInput, DenseRequest, etc.)
// are not accessible from this bench binary.  We benchmark the serde_json
// deserialization of the JSON shapes those types consume, which covers the
// same JSON-parsing overhead without needing access to internal modules.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

// ---------------------------------------------------------------------------
// bench_text_input_deser
// ---------------------------------------------------------------------------

fn bench_text_input_deser(c: &mut Criterion) {
    let single = r#""hello world this is a test sentence""#;
    let array_16 = serde_json::to_string(
        &(0..16)
            .map(|i| format!("sentence number {i}"))
            .collect::<Vec<_>>(),
    )
    .expect("array_16 should serialize");

    let mut group = c.benchmark_group("text_input_deser");

    group.bench_function("single_string", |b| {
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_str(black_box(single))
                .expect("single_string JSON should deserialize");
        });
    });

    group.bench_function("array_16", |b| {
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_str(black_box(&array_16))
                .expect("array_16 JSON should deserialize");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// bench_dense_request_deser
// ---------------------------------------------------------------------------

fn bench_dense_request_deser(c: &mut Criterion) {
    let single_req = r#"{"input": "embed this text please"}"#;
    let array_req = serde_json::to_string(&serde_json::json!({
        "input": (0..64).map(|i| format!("sentence {i}")).collect::<Vec<_>>()
    }))
    .expect("array_req should serialize");

    let mut group = c.benchmark_group("dense_request_deser");

    group.bench_function("single_input", |b| {
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_str(black_box(single_req))
                .expect("single_input JSON should deserialize");
        });
    });

    group.bench_with_input(BenchmarkId::new("array_input", 64), &array_req, |b, req| {
        b.iter(|| {
            let _: serde_json::Value =
                serde_json::from_str(black_box(req)).expect("array_input JSON should deserialize");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_text_input_deser, bench_dense_request_deser);
criterion_main!(benches);
