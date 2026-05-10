//! Startup memory probe and cost-model coefficient fitter.
//!
//! Submodules:
//! - `runner`: drives the `(batch, seq)` shape sweep on the leader worker
//!   (`run_probe`, `PROBE_SHAPES`, the absolute-RSS guard, the arena warm-up).
//! - `cache`: persistent EFS-backed cache for fitted probe coefficients
//!   (`try_load_probe_cache`, `save_probe_cache`, `ProbeCache`).
//! - `fit`: ordinary least-squares fitter for the quadratic cost model
//!   (`fit_cost_model`, `DataPoint`).
//! - `corpus`: helpers that synthesize probe texts from the curated corpus.
//! - `validate`: tokenizer + ndarray shape check at the configured `max_seq`
//!   (no `session.run()`).

mod cache;
mod corpus;
mod fit;
mod runner;
mod validate;

pub(crate) use cache::{save_probe_cache, try_load_probe_cache};
pub(crate) use runner::run_probe;

#[cfg(test)]
mod tests;
