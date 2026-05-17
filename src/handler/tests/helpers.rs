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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8};

use arc_swap::ArcSwap;
use tokio::sync::Semaphore;

use crate::binpack::CostModel;
use crate::embedder::EmbedPool;
use crate::state::{AppState, ProbeStatus};

pub fn make_state(ready: bool, max_batch: usize) -> Arc<AppState> {
    Arc::new(AppState {
        pool: EmbedPool::closed_for_test(),
        ready: AtomicBool::new(ready),
        max_batch,
        total_workers: 2,
        max_seq_length: 8192,
        tuning: std::sync::OnceLock::new(),
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        probe_status: AtomicU8::new(ProbeStatus::Disabled as u8),
        request_permits: Arc::new(Semaphore::new(usize::MAX >> 3)),
    })
}
