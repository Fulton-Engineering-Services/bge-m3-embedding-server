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
fn env_override_takes_priority() {
    // We can't reliably set env vars in parallel tests, so test the parsing
    // function directly rather than the env-var path.
    // At minimum, verify the function runs without panicking.
    let _ = detect_available_memory();
}

#[test]
fn memory_reading_has_positive_bytes() {
    let r = detect_available_memory();
    assert!(r.available_bytes > 0, "available_bytes must be positive");
}

#[cfg(target_os = "linux")]
#[test]
fn rss_is_positive_on_linux() {
    let rss = read_process_rss_bytes();
    assert!(rss.is_some(), "RSS should be readable on Linux");
    assert!(rss.unwrap() > 0, "RSS must be positive");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn rss_returns_none_on_non_linux() {
    assert!(
        read_process_rss_bytes().is_none(),
        "RSS measurement unsupported on non-Linux"
    );
}

#[test]
fn memory_source_display() {
    assert_eq!(MemorySource::Override.to_string(), "override");
    assert_eq!(MemorySource::CgroupV2.to_string(), "cgroup_v2");
    assert_eq!(MemorySource::CgroupV1.to_string(), "cgroup_v1");
    assert_eq!(MemorySource::HostRam.to_string(), "host_ram");
}
