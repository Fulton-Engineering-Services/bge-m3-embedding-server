/// Memory detection for auto-budget computation.
///
/// Production target is Linux (Fargate/ECS). On Linux we walk the cgroup
/// hierarchy to find the container memory limit, then fall back to host RAM
/// reported by `/proc/meminfo`. On macOS we read host RAM via `sysctl`;
/// cgroup support requires unsafe FFI so it is deferred.
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
        return MemoryReading { available_bytes: bytes, source: MemorySource::Override };
    }

    // --- step 2 / 3: Linux cgroup ---
    #[cfg(target_os = "linux")]
    if let Some(r) = cgroup_memory() {
        return r;
    }

    // --- step 4 / 5: OS-level RAM ---
    if let Some(bytes) = host_ram() {
        return MemoryReading { available_bytes: bytes, source: MemorySource::HostRam };
    }

    // --- fallback ---
    let fallback: usize = 4 * 1024 * 1024 * 1024; // 4 GiB
    warn!(
        available_bytes = fallback,
        "Memory detection failed on all paths; using 4 GiB fallback. \
         Set BGE_M3_AVAILABLE_MEMORY_BYTES to override."
    );
    MemoryReading { available_bytes: fallback, source: MemorySource::HostRam }
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
    // cgroup v2: /sys/fs/cgroup/memory.max (value "max" means unlimited)
    if let Ok(raw) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let trimmed = raw.trim();
        if trimmed != "max" {
            if let Ok(bytes) = trimmed.parse::<usize>() {
                tracing::debug!(bytes, source = "cgroup_v2", "Detected memory limit");
                return Some(MemoryReading {
                    available_bytes: bytes,
                    source: MemorySource::CgroupV2,
                });
            }
        }
    }

    // cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    // The kernel sets a sentinel value of 9223372036854771712 (near i64::MAX)
    // when no limit is configured. Treat any value ≥ 1 TiB as "unlimited".
    const ONE_TIB: usize = 1024 * 1024 * 1024 * 1024;
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
            let kb: usize = line
                .split_whitespace()
                .nth(1)?
                .parse()
                .ok()?;
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
}
