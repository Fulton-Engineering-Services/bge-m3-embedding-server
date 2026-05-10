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

use super::super::CostModel;

pub fn model(a: f64, b: f64, max_bytes: usize) -> CostModel {
    CostModel {
        a,
        b,
        max_workspace_bytes: max_bytes,
    }
}

// Simple model with no quadratic term; maps budget/count = max_tokens_per_chunk.
pub fn linear_model(bytes_per_token: f64, max_bytes: usize) -> CostModel {
    model(bytes_per_token, 0.0, max_bytes)
}
