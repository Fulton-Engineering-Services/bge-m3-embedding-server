use serde::{Deserialize, Deserializer, Serialize};

// ---------------------------------------------------------------------------
// TextInput — accepts either a single string or an array of strings
// ---------------------------------------------------------------------------

/// Newtype wrapping a `Vec<String>` that deserializes from either
/// `"a single string"` or `["array", "of", "strings"]`.
#[derive(Debug, PartialEq)]
pub struct TextInput(pub Vec<String>);

impl<'de> Deserialize<'de> for TextInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrArray {
            Single(String),
            Multiple(Vec<String>),
        }

        match StringOrArray::deserialize(deserializer)? {
            StringOrArray::Single(s) => Ok(TextInput(vec![s])),
            StringOrArray::Multiple(v) => Ok(TextInput(v)),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense embedding types
// ---------------------------------------------------------------------------

/// Request body for the dense embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct DenseRequest {
    pub input: TextInput,
    pub model: Option<String>,
}

/// Top-level response for the dense embeddings endpoint (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct DenseResponse {
    pub object: &'static str,
    pub model: &'static str,
    pub data: Vec<DenseEmbeddingData>,
    pub usage: Usage,
}

/// Per-document dense embedding entry.
#[derive(Debug, Serialize)]
pub struct DenseEmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// Token usage counters.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Sparse embedding types
// ---------------------------------------------------------------------------

/// Request body for the sparse embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct SparseRequest {
    pub input: TextInput,
}

/// Top-level response for the sparse embeddings endpoint.
#[derive(Debug, Serialize)]
pub struct SparseResponse {
    pub data: Vec<SparseEmbeddingData>,
}

/// Per-document sparse embedding entry.
#[derive(Debug, Serialize)]
pub struct SparseEmbeddingData {
    pub index: usize,
    pub sparse_values: SparseValues,
}

/// Parallel arrays of token indices and their weights.
#[derive(Debug, Serialize)]
pub struct SparseValues {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn text_input_deserializes_single_string() {
        let json = r#""hello world""#;
        let input: TextInput = serde_json::from_str(json).expect("deserialize single string");
        assert_eq!(input, TextInput(vec!["hello world".to_string()]));
    }

    #[test]
    fn text_input_deserializes_array() {
        let json = r#"["foo", "bar", "baz"]"#;
        let input: TextInput = serde_json::from_str(json).expect("deserialize array");
        assert_eq!(
            input,
            TextInput(vec![
                "foo".to_string(),
                "bar".to_string(),
                "baz".to_string()
            ])
        );
    }

    #[test]
    fn dense_request_model_is_optional() {
        let with_model = r#"{"input": "hello", "model": "bge-m3"}"#;
        let req: DenseRequest = serde_json::from_str(with_model).expect("deserialize with model");
        assert_eq!(req.model.as_deref(), Some("bge-m3"));
        assert_eq!(req.input, TextInput(vec!["hello".to_string()]));

        let without_model = r#"{"input": "hello"}"#;
        let req: DenseRequest =
            serde_json::from_str(without_model).expect("deserialize without model");
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
    fn text_input_rejects_number() {
        let json = "42";
        let result = serde_json::from_str::<TextInput>(json);
        assert!(
            result.is_err(),
            "numbers should not deserialize as TextInput"
        );
    }

    #[test]
    fn text_input_rejects_object() {
        let json = r#"{"key": "value"}"#;
        let result = serde_json::from_str::<TextInput>(json);
        assert!(
            result.is_err(),
            "objects should not deserialize as TextInput"
        );
    }

    #[test]
    fn text_input_rejects_null() {
        let json = "null";
        let result = serde_json::from_str::<TextInput>(json);
        assert!(result.is_err(), "null should not deserialize as TextInput");
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

    proptest! {
        #[test]
        fn text_input_single_string_round_trips(s in "\\PC*") {
            // Any unicode string should deserialize as a single-element Vec
            let json = serde_json::to_string(&s).expect("string should serialize");
            let input: TextInput = serde_json::from_str(&json).expect("single string should deserialize");
            prop_assert_eq!(input.0.len(), 1);
            prop_assert_eq!(&input.0[0], &s);
        }
    }

    proptest! {
        #[test]
        fn text_input_array_round_trips(strings in prop::collection::vec("\\PC*", 1..=10)) {
            let json = serde_json::to_string(&strings).expect("array should serialize");
            let input: TextInput = serde_json::from_str(&json).expect("string array should deserialize");
            prop_assert_eq!(input.0, strings);
        }
    }

    proptest! {
        #[test]
        fn dense_request_deserializes_single_input(s in "\\PC*") {
            let json = serde_json::json!({ "input": s });
            let result: Result<DenseRequest, _> = serde_json::from_value(json);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap().input.0.len(), 1);
        }
    }

    proptest! {
        #[test]
        fn text_input_empty_array_deserializes_to_empty_vec(
            // Use a constant empty array — property just verifies this is stable
            _unused in 0..1_i32
        ) {
            let json = "[]";
            let input: TextInput = serde_json::from_str(json).expect("empty array should deserialize");
            prop_assert_eq!(input.0.len(), 0);
        }
    }
}
