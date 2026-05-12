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

use crate::sysinfo::detect_gpu_count;

#[test]
fn env_override_is_respected() {
    assert_eq!(detect_gpu_count(Some(4)), 4);
    assert_eq!(detect_gpu_count(Some(8)), 8);
    assert_eq!(detect_gpu_count(Some(1)), 1);
}

#[test]
fn env_override_clamps_zero_to_one() {
    assert_eq!(detect_gpu_count(Some(0)), 1);
}

#[test]
fn no_override_returns_at_least_one() {
    // On any platform (including Linux without an NVIDIA driver) the function
    // must return ≥ 1.  We can't assert the exact value on a non-GPU CI host,
    // but the contract "never 0" must always hold.
    assert!(detect_gpu_count(None) >= 1);
}
