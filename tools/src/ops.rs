//! Application operations shared by the CLI and the later TUI: build a
//! replacement file from two checkpoints, and apply a replacement file
//! to a copy of a model directory. Frontends own parsing and rendering;
//! all patch, hash and filesystem semantics live here.
//!
//! `apply` never recomposes a checkpoint: the model directory is copied
//! to a sibling temporary directory, the copied `model.safetensors` is
//! patched in place by ranged writes of only the changed bytes, and the
//! directory is renamed into place only after every write and check
//! has completed. A refusal before that rename publishes nothing and
//! leaves every input byte-unchanged; the rename itself is a local
//! check-then-rename rather than atomic no-clobber against a
//! concurrent writer, and a late post-rename I/O error may leave the
//! complete output visible (see `files::TempDir::publish`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::error::ToolError;
use crate::files::{Sha256Hex, check_cancelled};

mod apply;
mod build;

pub use apply::apply_replacement;
pub use build::build_replacement;

/// Progress reported between tensors; frontends adapt it as they like.
#[non_exhaustive]
pub struct Progress {
    pub operation: &'static str,
    pub tensor: Option<String>,
    pub tensors_done: u64,
    pub tensors_total: u64,
}

/// Cooperation context: a cancellation token checked between tensors
/// and inside chunked loops, plus a progress sink. Synchronous by
/// design; a TUI runs operations on a worker thread with the same `Cx`.
pub struct Cx<'a> {
    pub(crate) cancelled: &'a AtomicBool,
    progress: &'a mut (dyn FnMut(Progress) + 'a),
}

impl<'a> Cx<'a> {
    pub fn new(cancelled: &'a AtomicBool, progress: &'a mut (dyn FnMut(Progress) + 'a)) -> Self {
        Self {
            cancelled,
            progress,
        }
    }

    pub(crate) fn check(&self) -> Result<(), ToolError> {
        check_cancelled(self.cancelled)
    }

    fn emit(&mut self, operation: &'static str, tensor: Option<String>, done: u64, total: u64) {
        (self.progress)(Progress {
            operation,
            tensor,
            tensors_done: done,
            tensors_total: total,
        });
    }
}

/// `weightspace replacement`: diff two checkpoints into one file.
pub struct BuildReplacement {
    pub base: PathBuf,
    pub edited: PathBuf,
    pub manifest: Option<PathBuf>,
    pub out: PathBuf,
}

/// `weightspace apply`: bake a replacement into a copy of a model
/// directory holding a single root `model.safetensors`.
pub struct ApplyReplacement {
    pub patch: PathBuf,
    pub model: PathBuf,
    pub out: PathBuf,
    /// Expected base SHA-256 per full-form tensor, from
    /// `--expect-tensor-sha256 NAME=HEX`. The patch file itself does
    /// not record these digests.
    pub expect_tensor_sha256: BTreeMap<String, Sha256Hex>,
}

/// What an operation did. For `apply`, `out_sha256`/`out_bytes` describe
/// the patched `model.safetensors`; `entries_changed` counts sparse
/// indices plus the element counts of full tensors.
pub struct Report {
    pub tensors_changed: u64,
    pub entries_changed: u64,
    pub out_sha256: Sha256Hex,
    pub out_bytes: u64,
}
