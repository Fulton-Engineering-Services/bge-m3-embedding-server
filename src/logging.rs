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

//! Tracing initialization with a build-variant tag on every JSON log line.
//!
//! In `CloudWatch` / non-TTY environments the server emits one JSON object per
//! log event. Operators reading logs from a mixed CPU/CUDA fleet need to know
//! at a glance which binary produced the line, so every JSON event begins
//! with a `"build"` attribute resolved at compile time from the active Cargo
//! features (`"cpu"` for the default MLAS image, `"cuda"` whenever the `cuda`
//! or `tensorrt` features are enabled — both are turned on by
//! `Dockerfile.cuda`).
//!
//! The human-readable `text` / `pretty` formats used during local dev are
//! left unchanged.

use std::fmt;

use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::registry::LookupSpan;

/// Compile-time identifier for the build of the server that is running.
///
/// Resolves to `"cuda"` when either the `cuda` or `tensorrt` Cargo feature is
/// enabled — the production CUDA image (`Dockerfile.cuda`) turns on both, and
/// operators care about "is this the GPU image?" not "which EP variant did it
/// register". Otherwise resolves to `"cpu"` (the default MLAS build).
pub const BUILD_VARIANT: &str = if cfg!(any(feature = "cuda", feature = "tensorrt")) {
    "cuda"
} else {
    "cpu"
};

/// Wraps a JSON [`FormatEvent`] so the rendered object always begins with a
/// `"build"` key.
///
/// The wrapper renders the inner formatter into a temporary `String`, then
/// rewrites the leading `{` to `{"build":"<variant>",`. All other keys, span
/// context, and trailing newline produced by the inner formatter are
/// preserved verbatim, so existing `CloudWatch` Insights queries keep working.
pub struct PrependBuild<F> {
    inner: F,
}

impl<F> PrependBuild<F> {
    /// Wrap `inner` so its rendered events are prefixed with the build key.
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<S, N, F> FormatEvent<S, N> for PrependBuild<F>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
    F: FormatEvent<S, N>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let mut buf = String::new();
        self.inner.format_event(ctx, Writer::new(&mut buf), event)?;

        let trailing_newline = buf.ends_with('\n');
        let body = if trailing_newline {
            &buf[..buf.len() - 1]
        } else {
            &buf[..]
        };

        if let Some(rest) = body.strip_prefix('{') {
            if let Some(rest) = rest.strip_prefix('}') {
                // Inner produced an empty object `{}` — emit `{"build":"…"}`
                // followed by whatever (if anything) trailed the close brace.
                write!(writer, "{{\"build\":\"{BUILD_VARIANT}\"}}{rest}")?;
            } else {
                write!(writer, "{{\"build\":\"{BUILD_VARIANT}\",{rest}")?;
            }
        } else {
            // Inner formatter did not produce a JSON object (e.g. someone
            // wrapped the wrong formatter). Pass the body through unchanged
            // rather than corrupting it.
            writer.write_str(body)?;
        }

        if trailing_newline {
            writer.write_char('\n')?;
        }
        Ok(())
    }
}

/// Initialize the global tracing subscriber.
///
/// Reads `RUST_LOG` for the filter directive (defaulting to `info`) and
/// `BGE_M3_LOG_FORMAT` for the format selection: `json`, `text`/`pretty`, or
/// auto-detect via the stdout TTY check. JSON events carry a leading
/// `"build"` attribute identifying the Cargo build variant; the human formats
/// are unchanged.
///
/// Calling this twice is harmless — the second call no-ops via `try_init`.
pub fn init() {
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let log_format = std::env::var("BGE_M3_LOG_FORMAT").ok();
    // JSON by default in non-TTY environments (Docker / Fargate / CloudWatch).
    // Force pretty with BGE_M3_LOG_FORMAT=text or BGE_M3_LOG_FORMAT=pretty.
    // Force JSON with BGE_M3_LOG_FORMAT=json.
    let want_json = match log_format.as_deref() {
        Some("text" | "pretty") => false,
        Some("json") => true,
        _ => !std::io::IsTerminal::is_terminal(&std::io::stdout()),
    };
    if want_json {
        let inner = tracing_subscriber::fmt::format()
            .json()
            .with_current_span(true);
        tracing_subscriber::fmt()
            .event_format(PrependBuild::new(inner))
            .fmt_fields(tracing_subscriber::fmt::format::JsonFields::new())
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    }
}

#[cfg(test)]
mod tests;
