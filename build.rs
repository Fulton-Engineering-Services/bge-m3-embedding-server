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

fn main() {
    if std::env::var_os("ORT_LIB_LOCATION").is_none() {
        println!(
            "cargo:warning=ORT_LIB_LOCATION is not set. \
             ort-sys may download a prebuilt ONNX Runtime binary from an external URL. \
             Set ORT_LIB_LOCATION to point to a locally built ONNX Runtime to avoid this."
        );
    }

    // Capture the short git SHA for the startup banner.
    // Falls back to "unknown" when git is unavailable (e.g. in Docker build contexts
    // where .git is excluded via .dockerignore).
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=BGE_M3_GIT_SHA={git_sha}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
}
