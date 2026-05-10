//! Error helpers for the embedder.

/// Converts an ort error to an anyhow error by formatting it as a string.
///
/// `ort::Error` became generic in rc.12 (`ort::Error<T>`) and its variants gained
/// `NonNull<>` pointer fields that are `!Send + !Sync`, preventing direct
/// `?`-propagation into `anyhow::Error` (which requires `Send + Sync + 'static`).
/// Accepting `impl Display` makes this helper work for all instantiations
/// (`ort::Error<()>`, `ort::Error<SessionBuilder>`, etc.) without naming the type.
pub(super) fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}
