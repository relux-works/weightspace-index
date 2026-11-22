//! Typed errors for the patch operations (`replacement`, `apply`).
//!
//! The legacy `check`/`generate` paths keep aggregating `Vec<String>`
//! problems in the binary; only the new operations return [`ToolError`].
//! Every malformed input is one of these refusals, never a panic.

use std::path::PathBuf;

/// One operation failed or was refused. A refusal before publication
/// publishes no output; a late I/O error after the publishing hard
/// link or rename may leave a complete visible output (see `files`).
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// A filesystem operation failed.
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The `safetensors` crate rejected data we asked it to serialize.
    #[error("{path}: {source}")]
    Safetensors {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },
    /// A safetensors header or its tensor table is malformed.
    #[error("{path}: malformed safetensors: {detail}")]
    Malformed { path: PathBuf, detail: String },
    /// A tensor dtype whose elements are not whole stored bytes cannot be
    /// addressed element by element, so the operation refuses it.
    #[error("{path}: tensor {tensor}: unsupported dtype {dtype} (elements are not whole bytes)")]
    UnsupportedDtype {
        path: PathBuf,
        tensor: String,
        dtype: safetensors::Dtype,
    },
    /// The base does not hold the bytes the patch expects it to replace.
    #[error("tensor {tensor}: wrong base: {detail}")]
    WrongBase { tensor: String, detail: String },
    /// A full-form patch tensor has no `--expect-tensor-sha256` digest.
    #[error(
        "tensor {tensor}: full-form patch tensor needs an expected base digest \
         (--expect-tensor-sha256 {tensor}=HEX); the patch file does not record one"
    )]
    MissingExpectedDigest { tensor: String },
    /// An input file changed size while its private snapshot was
    /// being copied, so the snapshot would not be one consistent
    /// byte sequence.
    #[error("{path}: input changed while it was being snapshotted")]
    InputChanged { path: PathBuf },
    /// An input path does not resolve to a regular file.
    #[error("{path}: input is not a regular file")]
    NotRegularFile { path: PathBuf },
    /// The authors' manifest is malformed or does not match the checkpoints.
    #[error("manifest {path}: {detail}")]
    Manifest { path: PathBuf, detail: String },
    /// The output path already exists; outputs are never clobbered.
    #[error("output already exists: {path}")]
    OutputExists { path: PathBuf },
    /// The output path has no usable final name, or the name is one
    /// the platform reserves (see `output::OutputTarget::check`).
    #[error("invalid output name: {path}")]
    InvalidOutputName { path: PathBuf },
    /// The output path's parent exists but is not a directory.
    #[error("{path}: output parent is not a directory")]
    OutputParentNotDir { path: PathBuf },
    /// A symlink where the custody rules refuse one.
    #[error("symlink refused at {path}: {detail}")]
    SymlinkRefused { path: PathBuf, detail: String },
    /// Slice 1 supports a single root `model.safetensors` only.
    #[error("{path}: sharded or indexed model layouts are not supported: {detail}")]
    ShardedModel { path: PathBuf, detail: String },
    /// A CLI/operation argument is invalid.
    #[error("invalid argument: {detail}")]
    InvalidArgument { detail: String },
    /// Base and edited checkpoints are identical; there is no
    /// replacement to write. A normal user outcome, not a failure
    /// of either input.
    #[error("base and edited are identical; no replacement to write")]
    NoChanges,
    /// Base and edited checkpoints do not describe the same tensors
    /// (name set, shape or dtype), so no diff is meaningful.
    #[error("tensor mismatch: {detail}")]
    TensorSetMismatch { detail: String },
    /// The output would live inside the model directory being copied.
    #[error("{output}: output must not be inside the model directory {model}")]
    OutputInsideInput { output: PathBuf, model: PathBuf },
    /// An input path that must be a directory is not one.
    #[error("{path}: not a directory")]
    NotDirectory { path: PathBuf },
    /// A digest string is not 64 hex characters (see
    /// `crate::files::Sha256Hex::parse` and its own error type).
    #[error("invalid SHA-256 digest: {detail}")]
    ParseDigest { detail: String },
    /// An internal invariant failed (for example generated output
    /// failing its own post-write gate, or checked arithmetic in
    /// the representation planner overflowing). Distinct from every
    /// user-input refusal above so frontends can report it as a bug.
    #[error("internal error: {detail}")]
    Internal { detail: String },
    /// The cancellation token was set; nothing was published.
    #[error("cancelled")]
    Cancelled,
    /// Two or more stacked replacement patches modify the same coordinate
    /// while `--on-conflict=error` is active.
    #[error("tensor {tensor}: patch conflict on {overlapping_entries} entries: {detail}")]
    PatchConflict {
        tensor: String,
        overlapping_entries: u64,
        detail: String,
    },
    /// Backup or restore of an in-place patched model failed.
    #[error("backup {path}: {detail}")]
    Backup { path: PathBuf, detail: String },
}

impl ToolError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn malformed(path: impl Into<PathBuf>, detail: impl Into<String>) -> Self {
        Self::Malformed {
            path: path.into(),
            detail: detail.into(),
        }
    }

    pub(crate) fn internal(detail: impl Into<String>) -> Self {
        Self::Internal {
            detail: detail.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::Sha256Hex;

    // Refusal-variant table: every formerly-`Refused` outcome is a
    // distinct variant a frontend can branch on without text matching.
    #[test]
    fn refusal_variants_are_distinct_and_stable() {
        let cases: Vec<ToolError> = vec![
            ToolError::NoChanges,
            ToolError::TensorSetMismatch {
                detail: "tensor name sets differ".into(),
            },
            ToolError::InputChanged {
                path: PathBuf::from("base.safetensors"),
            },
            ToolError::OutputInsideInput {
                output: PathBuf::from("model/out"),
                model: PathBuf::from("model"),
            },
            ToolError::NotRegularFile {
                path: PathBuf::from("fifo"),
            },
            ToolError::NotDirectory {
                path: PathBuf::from("file"),
            },
            ToolError::Internal {
                detail: "generated replacement failed its post-write gate".into(),
            },
        ];
        let rendered: Vec<String> = cases.iter().map(ToString::to_string).collect();
        for (i, a) in rendered.iter().enumerate() {
            for b in &rendered[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert!(rendered[0].contains("identical"), "{}", rendered[0]);
        assert!(rendered[3].contains("inside the model"), "{}", rendered[3]);
    }

    #[test]
    fn parse_digest_has_own_error_type() {
        let err = Sha256Hex::parse("xyz").unwrap_err();
        assert!(err.to_string().contains("64 hex"), "{err}");
        let tool: ToolError = err.into();
        assert!(matches!(tool, ToolError::ParseDigest { .. }), "{tool}");
        assert_eq!(
            Sha256Hex::parse(&"AB".repeat(32)).unwrap().as_str(),
            "ab".repeat(32).as_str()
        );
    }
}
