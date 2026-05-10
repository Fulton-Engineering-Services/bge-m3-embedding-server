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

use super::super::*;

#[test]
fn dense_request_model_is_optional() {
    let with_model = r#"{"input": "hello", "model": "bge-m3"}"#;
    let req: DenseRequest = serde_json::from_str(with_model).expect("deserialize with model");
    assert_eq!(req.model.as_deref(), Some("bge-m3"));
    assert_eq!(req.input, TextInput(vec!["hello".to_string()]));

    let without_model = r#"{"input": "hello"}"#;
    let req: DenseRequest = serde_json::from_str(without_model).expect("deserialize without model");
    assert!(req.model.is_none());
}

#[test]
fn dense_response_serializes_openai_format() {
    let response = DenseResponse {
        object: "list",
        model: "bge-m3",
        data: vec![DenseEmbeddingData {
            object: "embedding",
            index: 0,
            embedding: vec![0.1, 0.2, 0.3],
        }],
        usage: Usage {
            prompt_tokens: 3,
            total_tokens: 3,
        },
    };

    let json = serde_json::to_value(&response).expect("serialize dense response");
    assert_eq!(json["object"], "list");
    assert_eq!(json["model"], "bge-m3");
    assert_eq!(json["data"][0]["object"], "embedding");
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(json["data"][0]["embedding"][0], 0.1_f32);
    assert_eq!(json["usage"]["prompt_tokens"], 3);
    assert_eq!(json["usage"]["total_tokens"], 3);
}

#[test]
fn sparse_response_matches_consumer_format() {
    let response = SparseResponse {
        data: vec![SparseEmbeddingData {
            index: 0,
            sparse_values: SparseValues {
                indices: vec![42, 100],
                values: vec![0.5, 0.8],
            },
        }],
    };

    let json = serde_json::to_value(&response).expect("serialize sparse response");
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(json["data"][0]["sparse_values"]["indices"][0], 42);
    assert_eq!(json["data"][0]["sparse_values"]["indices"][1], 100);
    assert_eq!(json["data"][0]["sparse_values"]["values"][0], 0.5_f32);
    assert_eq!(json["data"][0]["sparse_values"]["values"][1], 0.8_f32);
}

#[test]
fn dual_response_serializes_with_paired_dense_and_sparse() {
    let response = DualResponse {
        object: "list",
        model: "bge-m3",
        data: vec![DualEmbeddingData {
            index: 0,
            embedding: vec![0.1, 0.2, 0.3],
            sparse_values: SparseValues {
                indices: vec![42, 100],
                values: vec![0.5, 0.8],
            },
        }],
        usage: Usage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };

    let json = serde_json::to_value(&response).expect("serialize dual response");
    assert_eq!(json["object"], "list");
    assert_eq!(json["model"], "bge-m3");
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(json["data"][0]["embedding"][0], 0.1_f32);
    assert_eq!(json["data"][0]["sparse_values"]["indices"][0], 42);
    assert_eq!(json["data"][0]["sparse_values"]["values"][1], 0.8_f32);
    assert_eq!(json["usage"]["prompt_tokens"], 5);
}

#[test]
fn dual_request_model_is_optional() {
    let with_model = r#"{"input": "hello", "model": "bge-m3"}"#;
    let req: DualRequest = serde_json::from_str(with_model).expect("deserialize with model");
    assert_eq!(req.model.as_deref(), Some("bge-m3"));
    assert_eq!(req.input, TextInput(vec!["hello".to_string()]));

    let without_model = r#"{"input": "hello"}"#;
    let req: DualRequest = serde_json::from_str(without_model).expect("deserialize without model");
    assert!(req.model.is_none());
}

#[test]
fn models_response_serializes_openai_format() {
    let resp = ModelsResponse {
        object: "list",
        data: vec![ModelEntry {
            id: "bge-m3",
            object: "model",
        }],
    };
    let json = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "bge-m3");
    assert_eq!(json["data"][0]["object"], "model");
}
