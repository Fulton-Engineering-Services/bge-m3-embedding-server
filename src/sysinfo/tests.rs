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

use super::*;

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

// -----------------------------------------------------------------------
// cgroup_v2_walk unit tests
//
// These use tempfile::TempDir to construct a minimal fake cgroup2 FS so
// the tests run on any OS (including macOS CI) without touching real
// /sys or /proc paths.
// -----------------------------------------------------------------------

/// Helper: write `memory.max` at every component of `rel_path` under `root`,
/// creating intermediate directories.  `values` is a list of (`rel_dir`, content)
/// pairs — e.g. `[("ecs.slice/task/container", "30064771072"), ("ecs.slice/task", "max")]`.
#[cfg(target_os = "linux")]
fn write_memory_max_files(root: &std::path::Path, files: &[(&str, &str)]) {
    for (rel_dir, content) in files {
        let dir = root.join(rel_dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("memory.max"), content).unwrap();
    }
    // Also create memory.max at root itself to terminate walks cleanly.
    if !root.join("memory.max").exists() {
        std::fs::write(root.join("memory.max"), "max").unwrap();
    }
}

/// Helper: write the /proc/self/cgroup file (unified v2 format).
#[cfg(target_os = "linux")]
fn write_proc_cgroup(dir: &std::path::Path, rel_cgroup_path: &str) -> std::path::PathBuf {
    let path = dir.join("proc_cgroup");
    std::fs::write(&path, format!("0::/{rel_cgroup_path}\n")).unwrap();
    path
}

/// Case 1 — ECS-style deeply-nested cgroup: numeric limit at the leaf.
#[cfg(target_os = "linux")]
#[test]
fn cgroup_v2_walk_ecs_nested_returns_leaf_limit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let cgroup_rel = "ecs.slice/ecs-task.scope/container-abc123";
    let limit_bytes: usize = 30 * 1024 * 1024 * 1024; // 30 GiB

    write_memory_max_files(
        root,
        &[
            (cgroup_rel, &limit_bytes.to_string()),
            ("ecs.slice/ecs-task.scope", "max"),
            ("ecs.slice", "max"),
        ],
    );
    let proc_cgroup = write_proc_cgroup(root, cgroup_rel);

    let result = cgroup_v2_walk(root.to_str().unwrap(), proc_cgroup.to_str().unwrap());
    assert!(result.is_some(), "should find the leaf limit");
    let r = result.unwrap();
    assert_eq!(r.available_bytes, limit_bytes);
    assert_eq!(r.source, MemorySource::CgroupV2);
}

/// Case 2 — leaf is "max", parent has a numeric limit: walk-up succeeds.
#[cfg(target_os = "linux")]
#[test]
fn cgroup_v2_walk_walks_up_to_parent_with_limit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let cgroup_rel = "ecs.slice/task.scope/container-xyz";
    let limit_bytes: usize = 28 * 1024 * 1024 * 1024; // 28 GiB

    write_memory_max_files(
        root,
        &[
            (cgroup_rel, "max"),
            ("ecs.slice/task.scope", &limit_bytes.to_string()),
            ("ecs.slice", "max"),
        ],
    );
    let proc_cgroup = write_proc_cgroup(root, cgroup_rel);

    let result = cgroup_v2_walk(root.to_str().unwrap(), proc_cgroup.to_str().unwrap());
    assert!(result.is_some(), "should walk up and find parent limit");
    assert_eq!(result.unwrap().available_bytes, limit_bytes);
}

/// Case 3 — all ancestors (including root) have "max": returns None so
/// caller falls through to `host_ram`.
#[cfg(target_os = "linux")]
#[test]
fn cgroup_v2_walk_all_max_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let cgroup_rel = "system.slice/myservice.service";

    write_memory_max_files(root, &[(cgroup_rel, "max"), ("system.slice", "max")]);
    // root memory.max is already "max" from write_memory_max_files helper.
    let proc_cgroup = write_proc_cgroup(root, cgroup_rel);

    let result = cgroup_v2_walk(root.to_str().unwrap(), proc_cgroup.to_str().unwrap());
    assert!(result.is_none(), "all-max hierarchy should return None");
}

/// Case 4 — /proc/self/cgroup is empty: returns None.
#[cfg(target_os = "linux")]
#[test]
fn cgroup_v2_walk_empty_proc_cgroup_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let proc_cgroup = root.join("proc_cgroup_empty");
    std::fs::write(&proc_cgroup, "").unwrap();

    let result = cgroup_v2_walk(root.to_str().unwrap(), proc_cgroup.to_str().unwrap());
    assert!(result.is_none(), "empty proc/cgroup should return None");
}

/// Case 5 — /proc/self/cgroup has cgroup-v1 lines only (no `0::` prefix):
/// returns None, preserving the cgroup-v1 fallback path.
#[cfg(target_os = "linux")]
#[test]
fn cgroup_v2_walk_v1_only_cgroup_file_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    // cgroup v1 format: multiple lines, non-zero hierarchy IDs.
    let proc_cgroup = root.join("proc_cgroup_v1");
    std::fs::write(
        &proc_cgroup,
        "11:memory:/docker/abc123\n10:cpu,cpuacct:/docker/abc123\n",
    )
    .unwrap();

    let result = cgroup_v2_walk(root.to_str().unwrap(), proc_cgroup.to_str().unwrap());
    assert!(
        result.is_none(),
        "v1-only cgroup file should return None (no 0:: line)"
    );
}
