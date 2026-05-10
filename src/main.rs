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

//! Thin entry point for the bge-m3 embedding server.
//!
//! The full orchestration logic lives in [`bge_m3_embedding_server::run`];
//! this binary only initialises tracing and hands off.

use bge_m3_embedding_server::run;

fn init_tracing() {
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    let log_format = std::env::var("BGE_M3_LOG_FORMAT").ok();
    // JSON by default in non-TTY environments (Docker/Fargate/CloudWatch).
    // Force pretty with BGE_M3_LOG_FORMAT=text or BGE_M3_LOG_FORMAT=pretty.
    // Force JSON with BGE_M3_LOG_FORMAT=json.
    let want_json = match log_format.as_deref() {
        Some("text" | "pretty") => false,
        Some("json") => true,
        _ => !std::io::IsTerminal::is_terminal(&std::io::stdout()),
    };
    if want_json {
        tracing_subscriber::fmt()
            .json()
            .with_current_span(true)
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    run().await
}
