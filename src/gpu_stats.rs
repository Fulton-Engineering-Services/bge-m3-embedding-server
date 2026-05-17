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

//! Per-device GPU VRAM, utilization, and temperature heartbeat logging.
//!
//! On GPU builds (`cuda` or `tensorrt` feature), this module initialises an
//! NVML handle at startup and emits one structured `INFO` log event per CUDA
//! device on each heartbeat tick.  On CPU builds the module compiles to a
//! zero-cost stub so the rest of the codebase can call it unconditionally
//! without any `#[cfg]` noise at the call site.

// ---------------------------------------------------------------------------
// GPU build — real NVML implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "nvml")]
mod inner {
    use nvml_wrapper::Nvml;
    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
    use tracing::{debug, info, warn};

    enum NvmlState {
        Ready { nvml: Box<Nvml>, gpu_count: usize },
        Unavailable,
    }

    /// Collects per-device VRAM and GPU utilization stats via NVML and emits
    /// them as structured log events.
    ///
    /// Instantiate once with [`GpuStatsCollector::init`], then call
    /// [`GpuStatsCollector::emit_heartbeat`] on each heartbeat tick.
    pub struct GpuStatsCollector {
        state: NvmlState,
    }

    impl GpuStatsCollector {
        /// Attempts to initialise NVML.
        ///
        /// If NVML is unavailable (driver not present, permission denied, etc.)
        /// a single `WARN` is logged and the collector enters a no-op state for
        /// the remainder of the process lifetime.
        pub fn init(gpu_count: usize) -> Self {
            match Nvml::init() {
                Ok(nvml) => {
                    info!(gpu_count, "NVML initialised; GPU heartbeat stats enabled");
                    Self {
                        state: NvmlState::Ready {
                            nvml: Box::new(nvml),
                            gpu_count,
                        },
                    }
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "NVML unavailable — GPU heartbeat stats disabled for this process"
                    );
                    Self {
                        state: NvmlState::Unavailable,
                    }
                }
            }
        }

        /// Emits one `INFO` log event per CUDA device with VRAM, utilization,
        /// and temperature statistics.
        ///
        /// Per-device errors are logged at `DEBUG` and skipped; the loop
        /// continues for remaining devices.  This method never panics.
        pub fn emit_heartbeat(&self) {
            let (nvml, gpu_count) = match &self.state {
                NvmlState::Ready { nvml, gpu_count } => (nvml, *gpu_count),
                NvmlState::Unavailable => return,
            };

            for device_idx in 0..gpu_count {
                #[allow(clippy::cast_possible_truncation)]
                let device_idx_u32 = device_idx as u32;

                let device = match nvml.device_by_index(device_idx_u32) {
                    Ok(d) => d,
                    Err(e) => {
                        debug!(
                            gpu_device = device_idx_u32,
                            error = %e,
                            "NVML: could not open device"
                        );
                        continue;
                    }
                };

                let mem = match device.memory_info() {
                    Ok(m) => m,
                    Err(e) => {
                        debug!(
                            gpu_device = device_idx_u32,
                            error = %e,
                            "NVML: could not read memory info"
                        );
                        continue;
                    }
                };

                let utilization = match device.utilization_rates() {
                    Ok(u) => u,
                    Err(e) => {
                        debug!(
                            gpu_device = device_idx_u32,
                            error = %e,
                            "NVML: could not read utilization rates"
                        );
                        continue;
                    }
                };

                let gpu_temp_c = match device.temperature(TemperatureSensor::Gpu) {
                    Ok(t) => t,
                    Err(e) => {
                        debug!(
                            gpu_device = device_idx_u32,
                            error = %e,
                            "NVML: could not read GPU temperature"
                        );
                        continue;
                    }
                };

                let vram_used_mb = mem.used / (1024 * 1024);
                let vram_total_mb = mem.total / (1024 * 1024);
                #[allow(clippy::cast_precision_loss)]
                let vram_utilization_pct = if mem.total > 0 {
                    mem.used as f32 / mem.total as f32 * 100.0
                } else {
                    0.0
                };
                let gpu_utilization_pct = utilization.gpu;
                let gpu_temp_f = gpu_temp_c * 9 / 5 + 32;

                info!(
                    gpu_device = device_idx_u32,
                    vram_used_mb,
                    vram_total_mb,
                    vram_utilization_pct,
                    gpu_utilization_pct,
                    gpu_temp_c,
                    gpu_temp_f,
                    "gpu heartbeat"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU build — zero-cost stub
// ---------------------------------------------------------------------------

#[cfg(not(feature = "nvml"))]
mod inner {
    /// No-op GPU stats collector for CPU builds.
    ///
    /// All methods compile away completely; no NVML dependency is pulled in.
    pub struct GpuStatsCollector;

    impl GpuStatsCollector {
        /// Returns a no-op collector.  The `gpu_count` argument is accepted for
        /// API compatibility with GPU builds but is otherwise ignored.
        pub fn init(_gpu_count: usize) -> Self {
            Self
        }

        /// No-op on CPU builds.
        #[allow(clippy::unused_self)]
        pub fn emit_heartbeat(&self) {}
    }
}

pub(crate) use inner::GpuStatsCollector;
