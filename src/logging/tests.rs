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

//! Tests for [`super::PrependModule`], [`super::PrependBuild`], and the
//! compile-time [`super::BUILD_VARIANT`] / [`super::BGE_MODULE`] constants.
//!
//! The formatter-chain tests wire `PrependModule(PrependBuild(JSON))` into a
//! subscriber, emit a single event into a captured `Vec<u8>` writer, and assert
//! the rendered line begins with `{"bge_module":"server","build":"…"`. We rely
//! on the raw-string assertion (not a key-order-preserving JSON parse) because
//! operators reading `CloudWatch` see the textual output, which is exactly what
//! we want to lock down.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use tracing::subscriber::with_default;
use tracing_subscriber::fmt::MakeWriter;

use super::{BGE_MODULE, BUILD_VARIANT, PrependBuild, PrependModule};

#[derive(Clone, Default)]
struct VecWriter(Arc<Mutex<Vec<u8>>>);

impl VecWriter {
    fn captured(&self) -> Vec<u8> {
        self.0.lock().expect("VecWriter mutex poisoned").clone()
    }
}

impl Write for VecWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("VecWriter mutex poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for VecWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn build_json_subscriber(buf: VecWriter) -> impl tracing::Subscriber + Send + Sync {
    let inner = tracing_subscriber::fmt::format()
        .json()
        .with_current_span(true);
    tracing_subscriber::fmt()
        .with_writer(buf)
        .event_format(PrependModule::new(PrependBuild::new(inner)))
        .fmt_fields(tracing_subscriber::fmt::format::JsonFields::new())
        .finish()
}

#[test]
fn json_log_line_starts_with_bge_module_then_build() {
    let buf = VecWriter::default();
    let subscriber = build_json_subscriber(buf.clone());

    with_default(subscriber, || {
        tracing::info!(answer = 42, "hello");
    });

    let bytes = buf.captured();
    let line = std::str::from_utf8(&bytes).expect("log output is valid UTF-8");
    let prefix = format!("{{\"bge_module\":\"{BGE_MODULE}\",\"build\":\"{BUILD_VARIANT}\",");
    assert!(
        line.starts_with(&prefix),
        "expected line to start with {prefix:?}, got: {line:?}"
    );
    assert!(
        line.contains("\"answer\":42"),
        "rewriter dropped a structured field: {line:?}"
    );
    assert!(
        line.contains("\"hello\""),
        "rewriter dropped the message: {line:?}"
    );
    assert!(
        line.ends_with('\n'),
        "rewriter dropped the trailing newline: {line:?}"
    );
}

#[test]
fn json_log_line_is_valid_json_object() {
    let buf = VecWriter::default();
    let subscriber = build_json_subscriber(buf.clone());

    with_default(subscriber, || {
        tracing::info!(route = "embeddings", batch_size = 4, "request complete");
    });

    let bytes = buf.captured();
    let line = std::str::from_utf8(&bytes).expect("log output is valid UTF-8");
    let parsed: serde_json::Value =
        serde_json::from_str(line.trim_end()).expect("each event must be a single JSON object");
    assert_eq!(
        parsed.get("bge_module").and_then(serde_json::Value::as_str),
        Some(BGE_MODULE),
        "bge_module attribute missing or wrong value: {parsed}"
    );
    assert_eq!(
        parsed.get("build").and_then(serde_json::Value::as_str),
        Some(BUILD_VARIANT),
        "build attribute missing or wrong value: {parsed}"
    );
}

#[test]
fn bge_module_is_always_server() {
    assert_eq!(BGE_MODULE, "server");
}

#[cfg(feature = "cuda")]
#[test]
fn build_variant_is_cuda_when_cuda_feature_enabled() {
    assert_eq!(BUILD_VARIANT, "cuda");
}

#[cfg(feature = "tensorrt")]
#[test]
fn build_variant_is_cuda_when_tensorrt_feature_enabled() {
    assert_eq!(BUILD_VARIANT, "cuda");
}

#[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
#[test]
fn build_variant_is_cpu_by_default() {
    assert_eq!(BUILD_VARIANT, "cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn json_log_line_starts_with_bge_module_then_cuda_build() {
    let buf = VecWriter::default();
    let subscriber = build_json_subscriber(buf.clone());

    with_default(subscriber, || {
        tracing::info!("cuda build check");
    });

    let bytes = buf.captured();
    let line = std::str::from_utf8(&bytes).expect("log output is valid UTF-8");
    assert!(
        line.starts_with("{\"bge_module\":\"server\",\"build\":\"cuda\","),
        "expected cuda prefix, got: {line:?}"
    );
}
