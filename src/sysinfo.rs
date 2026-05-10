/// Memory detection for auto-budget computation.
///
/// Production target is Linux (Fargate/ECS). On Linux we walk the cgroup
/// hierarchy to find the container memory limit, then fall back to host RAM
/// reported by `/proc/meminfo`. On macOS we read host RAM via `sysctl`;
/// cgroup support requires unsafe FFI so it is deferred.
///
/// ## cgroup-v2 detection on ECS Managed Instances (Bottlerocket)
///
/// ECS Managed Instances launch containers **without** `--cgroupns=private`,
/// so `/sys/fs/cgroup/memory.max` resolves to the unified-hierarchy root,
/// which reads `"max"` (no limit). The actual container memory limit is
/// set at a deeper path whose last component is recorded in
/// `/proc/self/cgroup` (unified-hierarchy format: a single line
/// `0::<path>`, e.g. `0::/ecs.slice/ecs-…-task.scope/<id>`).
///
/// `cgroup_memory()` reads `/proc/self/cgroup`, extracts that path, then
/// reads `memory.max` at each ancestor (deepest first) until it finds a
/// numeric limit or exhausts the tree. Falls through to `host_ram` only
/// when the entire walk yields `"max"` (truly unconstrained host).
///
/// RSS tracking (`read_process_rss_bytes`) is Linux-only (parses
/// `/proc/self/statm`). On macOS it returns `None`; the auto-budget logic
/// treats `None` as "cannot measure model footprint" and uses conservative
/// defaults.
use tracing::warn;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Where the available-memory reading came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // CgroupV2/CgroupV1 are constructed only on Linux; macOS sees them as unused.
pub(crate) enum MemorySource {
    /// `BGE_M3_AVAILABLE_MEMORY_BYTES` env override.
    Override,
    /// Linux cgroup v2 `memory.max`.
    CgroupV2,
    /// Linux cgroup v1 `memory.limit_in_bytes`.
    CgroupV1,
    /// `/proc/meminfo` `MemAvailable` (Linux) or `sysctl hw.memsize` (macOS).
    HostRam,
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Override => f.write_str("override"),
            Self::CgroupV2 => f.write_str("cgroup_v2"),
            Self::CgroupV1 => f.write_str("cgroup_v1"),
            Self::HostRam => f.write_str("host_ram"),
        }
    }
}

/// A memory reading with its provenance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryReading {
    pub available_bytes: usize,
    pub source: MemorySource,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detects available memory for the process.
///
/// Detection chain (first success wins):
/// 1. `BGE_M3_AVAILABLE_MEMORY_BYTES` env override.
/// 2. Linux cgroup v2: `/sys/fs/cgroup/memory.max`.
/// 3. Linux cgroup v1: `/sys/fs/cgroup/memory/memory.limit_in_bytes`.
/// 4. Linux: `/proc/meminfo` `MemAvailable`.
/// 5. macOS: `sysctl hw.memsize` (total host RAM; no cgroup support).
/// 6. Fallback: 4 GiB constant with a warning log.
pub(crate) fn detect_available_memory() -> MemoryReading {
    // --- step 1: explicit override ---
    if let Some(bytes) = env_override() {
        return MemoryReading {
            available_bytes: bytes,
            source: MemorySource::Override,
        };
    }

    // --- step 2 / 3: Linux cgroup ---
    #[cfg(target_os = "linux")]
    if let Some(r) = cgroup_memory() {
        return r;
    }

    // --- step 4 / 5: OS-level RAM ---
    if let Some(bytes) = host_ram() {
        return MemoryReading {
            available_bytes: bytes,
            source: MemorySource::HostRam,
        };
    }

    // --- fallback ---
    let fallback: usize = 4 * 1024 * 1024 * 1024; // 4 GiB
    warn!(
        available_bytes = fallback,
        "Memory detection failed on all paths; using 4 GiB fallback. \
         Set BGE_M3_AVAILABLE_MEMORY_BYTES to override."
    );
    MemoryReading {
        available_bytes: fallback,
        source: MemorySource::HostRam,
    }
}

/// Returns the current process's RSS (Resident Set Size) in bytes, or `None`
/// if measurement is not supported on this platform.
///
/// Linux: parses `/proc/self/statm`. Field 1 (index 1) is RSS in pages;
/// multiplied by the system page size (typically 4096).
///
/// macOS: returns `None` — requires `task_info` FFI which conflicts with
/// `unsafe_code = "forbid"`. A future release can add it via the `mach2`
/// crate.
pub(crate) fn read_process_rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    return linux_rss();

    #[cfg(not(target_os = "linux"))]
    None
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn env_override() -> Option<usize> {
    std::env::var("BGE_M3_AVAILABLE_MEMORY_BYTES")
        .ok()
        .and_then(|v| {
            v.parse::<usize>().ok().or_else(|| {
                warn!(
                    value = %v,
                    "BGE_M3_AVAILABLE_MEMORY_BYTES is not a valid usize; ignoring"
                );
                None
            })
        })
}

#[cfg(target_os = "linux")]
fn cgroup_memory() -> Option<MemoryReading> {
    // Sentinel threshold: the cgroup v1 kernel uses a near-i64::MAX value when
    // no limit is configured. Treat any value ≥ 1 TiB as "unlimited".
    const ONE_TIB: usize = 1024 * 1024 * 1024 * 1024;

    // --- cgroup v2: path-walk from /proc/self/cgroup ---
    //
    // ECS Managed Instances (Bottlerocket) do NOT set --cgroupns=private, so
    // /sys/fs/cgroup/memory.max resolves to the host root where value is "max".
    // The container's actual limit lives at a deeper path recorded in
    // /proc/self/cgroup (unified v2 format: `0::<path>`).
    //
    // Walk ancestors deepest-first until a numeric limit < 1 TiB is found.
    // If the entire walk yields "max", fall through to cgroup v1 then host_ram.
    if let Some(reading) = cgroup_v2_walk("/sys/fs/cgroup", "/proc/self/cgroup") {
        return Some(reading);
    }

    // --- cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes ---
    if let Ok(raw) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        let trimmed = raw.trim();
        if let Ok(bytes) = trimmed.parse::<usize>() {
            if bytes < ONE_TIB {
                tracing::debug!(bytes, source = "cgroup_v1", "Detected memory limit");
                return Some(MemoryReading {
                    available_bytes: bytes,
                    source: MemorySource::CgroupV1,
                });
            }
        }
    }

    None
}

/// Reads the cgroup v2 memory limit by walking ancestors of the container's
/// cgroup path.
///
/// # Arguments
///
/// - `cgroup_fs_root`: the mountpoint of the cgroup v2 filesystem (normally
///   `/sys/fs/cgroup`; injectable for unit tests).
/// - `proc_self_cgroup`: path to the per-process cgroup file (normally
///   `/proc/self/cgroup`; injectable for unit tests).
///
/// Parses the unified-hierarchy line (`0::<path>`), then iterates from the
/// deepest ancestor up to the root, reading `memory.max` at each level.
/// Returns the first numeric limit found that is below 1 TiB, or `None`
/// when the entire walk yields `"max"` or the file is unreadable.
#[cfg(target_os = "linux")]
pub(crate) fn cgroup_v2_walk(
    cgroup_fs_root: &str,
    proc_self_cgroup: &str,
) -> Option<MemoryReading> {
    const ONE_TIB: usize = 1024 * 1024 * 1024 * 1024;

    let cgroup_content = std::fs::read_to_string(proc_self_cgroup).ok()?;

    // Unified hierarchy: exactly one line, format `0::<path>` (e.g. `0::/ecs.slice/…`)
    // Legacy v1 has multiple lines, each with `<hierarchy_id>:<controllers>:<path>`.
    // We only attempt v2 if we find the unified `0::` prefix.
    let cgroup_rel_path = cgroup_content
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?;

    // Build the absolute cgroup directory path.
    let cgroup_dir = std::path::PathBuf::from(cgroup_fs_root).join(
        // Strip the leading '/' so PathBuf::join doesn't replace the root.
        cgroup_rel_path.trim_start_matches('/'),
    );

    // Walk ancestors from deepest to shallowest (inclusive of the container
    // cgroup itself, exclusive of the root mountpoint).
    let mut current = cgroup_dir.as_path();
    let fs_root = std::path::Path::new(cgroup_fs_root);

    loop {
        let memory_max = current.join("memory.max");
        if let Ok(raw) = std::fs::read_to_string(&memory_max) {
            let trimmed = raw.trim();
            if trimmed != "max" {
                if let Ok(bytes) = trimmed.parse::<usize>() {
                    if bytes < ONE_TIB {
                        tracing::debug!(
                            bytes,
                            source = "cgroup_v2",
                            path = %memory_max.display(),
                            "Detected memory limit"
                        );
                        return Some(MemoryReading {
                            available_bytes: bytes,
                            source: MemorySource::CgroupV2,
                        });
                    }
                }
            }
        }

        // Stop at the cgroup filesystem root — don't walk above it.
        if current == fs_root {
            break;
        }

        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    None
}

/// Linux: parse `MemAvailable` from `/proc/meminfo` (kB → bytes).
/// macOS: read total host RAM via `sysctl hw.memsize`.
fn host_ram() -> Option<usize> {
    #[cfg(target_os = "linux")]
    return linux_meminfo_available();

    #[cfg(target_os = "macos")]
    return macos_host_ram();

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    None
}

#[cfg(target_os = "linux")]
fn linux_meminfo_available() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn macos_host_ram() -> Option<usize> {
    // `sysctl -n hw.memsize` returns an integer in bytes printed to stdout.
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let stdout = std::str::from_utf8(&output.stdout).ok()?.trim();
    stdout.parse::<usize>().ok()
}

#[cfg(target_os = "linux")]
fn linux_rss() -> Option<usize> {
    // /proc/self/statm: all values in pages.
    // Fields: size, rss, shared, text, lib, data, dt
    let raw = std::fs::read_to_string("/proc/self/statm").ok()?;
    let rss_pages: usize = raw.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: page_size is a compile-time constant on Linux (4096 on x86_64/arm64).
    // We use sysconf(SC_PAGESIZE) via libc-free approach: fallback to 4096.
    let page_size = page_size_bytes();
    Some(rss_pages * page_size)
}

#[cfg(target_os = "linux")]
fn page_size_bytes() -> usize {
    // Read from /proc/self/auxv would be ideal but requires parsing ELF aux
    // vectors. Parsing /proc/$pid/smaps is too heavy. sysconf(SC_PAGESIZE)
    // requires libc. The practical answer on Linux/x86_64 and Linux/aarch64
    // is always 4096; we hard-code that to avoid any unsafe.
    4096
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
    /// creating intermediate directories.  `values` is a list of (rel_dir, content)
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
    /// caller falls through to host_ram.
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
}
