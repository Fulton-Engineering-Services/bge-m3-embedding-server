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

use super::super::fit::DataPoint;

/// Builds a `DataPoint` from `(batch, seq, a, b)` using the model formula
/// `rss = a * (batch * seq) + b * (batch * seq²)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn make_dp(batch: usize, seq: usize, a: f64, b: f64) -> DataPoint {
    let rss_delta = (a * (batch * seq) as f64 + b * (batch * seq * seq) as f64) as usize;
    DataPoint {
        batch,
        seq,
        rss_delta,
    }
}
