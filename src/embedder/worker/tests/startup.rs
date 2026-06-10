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

//! Tests for startup helper predicates.

use super::super::startup::detected_sm_for_ep;
use crate::config::EpSelection;

#[test]
fn detected_sm_for_ep_none_on_cpu() {
    assert!(detected_sm_for_ep(EpSelection::Cpu, 0).is_none());
}

#[test]
fn detected_sm_for_ep_none_on_cuda() {
    assert!(detected_sm_for_ep(EpSelection::Cuda, 0).is_none());
}

#[test]
fn detected_sm_for_ep_trt_without_gpu_returns_none_or_sm() {
    // On CI/macOS without nvidia-smi this is typically None; on GPU hosts it
    // may return Some. Either outcome must not panic.
    let _ = detected_sm_for_ep(EpSelection::TensorRt, 0);
}
