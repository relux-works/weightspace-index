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
    /// The authors' manifest is malformed or does not match the checkpoints.
    #[error("manifest {path}: {detail}")]
    Manifest { path: PathBuf, detail: String },
    /// The output path already exists; outputs are never clobbered.
    #[error("output already exists: {path}")]
    OutputExists { path: PathBuf },
    /// A symlink where the custody rules refuse one.
    #[error("symlink refused at {path}: {detail}")]
    SymlinkRefused { path: PathBuf, detail: String },
    /// Slice 1 supports a single root `model.safetensors` only.
    #[error("{path}: sharded or indexed model layouts are not supported: {detail}")]
    ShardedModel { path: PathBuf, detail: String },
    /// A CLI/operation argument is invalid.
    #[error("invalid argument: {detail}")]
    InvalidArgument { detail: String },
    /// Any other refusal with a stated reason.
    #[error("refused: {detail}")]
    Refused { detail: String },
    /// The cancellation token was set; nothing was published.
    #[error("cancelled")]
    Cancelled,
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

    pub(crate) fn refused(detail: impl Into<String>) -> Self {
        Self::Refused {
            detail: detail.into(),
        }
    }
}
