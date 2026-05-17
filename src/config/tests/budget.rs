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

//! Tests for cost-model env-var overrides: `BGE_M3_DISABLE_AUTO_BUDGET`,
//! `BGE_M3_TOKEN_BUDGET`, and `BGE_M3_COST_MODEL_A/B`.

use std::collections::HashMap;

use super::super::Config;
use super::helpers::lookup_from;
use crate::binpack::CostModel;

#[test]
fn disable_auto_budget_yields_conservative_model() {
    let map = HashMap::from([("BGE_M3_DISABLE_AUTO_BUDGET", "1")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg.cost_model_override.expect("override must be set");
    assert!((cm.a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON);
    assert!((cm.b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON);
    assert_eq!(cm.max_workspace_bytes, CostModel::DEFAULT_MAX_WORKSPACE);
}

#[test]
fn token_budget_translates_to_cost_model() {
    // With token_budget=8192 and conservative coefficients the workspace
    // must be a positive number.
    let map = HashMap::from([("BGE_M3_TOKEN_BUDGET", "8192")]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg.cost_model_override.expect("override must be set");
    assert!(cm.max_workspace_bytes > 0);
    assert!((cm.a - CostModel::CONSERVATIVE_A).abs() < f64::EPSILON);
    assert!((cm.b - CostModel::CONSERVATIVE_B).abs() < f64::EPSILON);
}

#[test]
fn explicit_cost_model_override() {
    let map = HashMap::from([
        ("BGE_M3_COST_MODEL_A", "20000.0"),
        ("BGE_M3_COST_MODEL_B", "5.0"),
        ("BGE_M3_AVAILABLE_MEMORY_BYTES", "1073741824"),
    ]);
    let cfg = Config::from_lookup(lookup_from(&map));
    let cm = cfg.cost_model_override.expect("override must be set");
    assert!((cm.a - 20_000.0).abs() < 1e-9);
    assert!((cm.b - 5.0).abs() < 1e-9);
    assert_eq!(cm.max_workspace_bytes, 1_073_741_824);
}
