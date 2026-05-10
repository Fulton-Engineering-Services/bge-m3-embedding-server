//! Sparse embedding equivalence cases.
//!
//! Reserved for future sparse-only equivalence tests. The current
//! `equivalence_all_seq_lengths` (in `dense.rs`) only checks dense cosine
//! similarity; a parallel sparse check would compare BGE-M3 SPLADE-style
//! outputs against the reference `reference_sparse_seq_*.json` fixtures.
//!
//! Until such tests are added, this module is intentionally empty so the
//! `tests/equivalence/main.rs` harness can declare `mod sparse;` and the
//! file layout matches the source-layout-refactor plan.
