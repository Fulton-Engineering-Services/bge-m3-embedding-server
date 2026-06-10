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
use std::sync::atomic::AtomicUsize;

use arc_swap::ArcSwap;

use super::super::config::WorkerConfig;
use crate::binpack::CostModel;
use crate::config::{EpSelection, ModelVariant};

pub fn test_worker_config(ep: EpSelection, ceiling: usize) -> WorkerConfig {
    WorkerConfig {
        cost_model: Arc::new(ArcSwap::from_pointee(CostModel::conservative(
            CostModel::DEFAULT_MAX_WORKSPACE,
        ))),
        idle_timeout: None,
        model_variant: ModelVariant::Fp32,
        max_seq_length: 512,
        intra_threads: 1,
        ep,
        trt_warmup_shapes: vec![],
        device_id: 0,
        gpu_count: 1,
        trt_max_workspace_bytes: None,
        gpu_mem_limit_bytes: None,
        jit_suspect_tx: None,
        engine_propagation_tx: None,
        prewarm_strict: true,
        circuit_breaker_threshold: 5,
        trt_inband_jit_guard_enabled: true,
        trt_inband_jit_guard_seq: 4096,
        warmed_seq_ceiling: Arc::new(AtomicUsize::new(ceiling)),
        #[cfg(feature = "cache-gc")]
        trt_cache_gc_enabled: false,
    }
}

pub fn sample_embed_stats(
    inference_ms: u64,
    max_chunk_seq: usize,
) -> crate::embedder::types::EmbedStats {
    crate::embedder::types::EmbedStats {
        chunks: 1,
        max_chunk_seq,
        total_token_positions: max_chunk_seq,
        tokenize_ms: 1,
        inference_ms,
        seq_len_min: max_chunk_seq,
        seq_len_max: max_chunk_seq,
        seq_len_mean: max_chunk_seq,
        seq_len_p95: max_chunk_seq,
    }
}
