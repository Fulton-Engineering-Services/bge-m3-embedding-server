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
