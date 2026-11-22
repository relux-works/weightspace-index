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
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use crate::error::ToolError;
use crate::files::{Sha256Hex, check_cancelled};

mod apply;
mod build;
pub mod stack;

pub use apply::apply_replacement;
pub(crate) use apply::verify_patch_against_base;
pub use build::build_replacement;
pub use stack::{
    AppliedPatchRecord, ApplyStack, ConflictPolicy, PatchSpec, PatchedState, RestoreReport,
    apply_stack, read_state, restore_backup,
};

/// The long-running phase a [`Progress`] report belongs to. Byte
/// counters are per-phase: `bytes_done` restarts when the phase
/// changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Snapshot,
    Verify,
    Copy,
    Patch,
    Hash,
    Plan,
    Serialize,
    Publish,
}

/// Progress reported between tensors and at chunk granularity inside
/// the long byte-moving phases; frontends adapt it as they like.
/// Byte fields are 0/0 on per-tensor reports that carry no byte
/// counter.
#[non_exhaustive]
pub struct Progress {
    pub operation: &'static str,
    pub phase: Phase,
    pub tensor: Option<String>,
    pub tensors_done: u64,
    pub tensors_total: u64,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

/// Cooperation context: a cancellation token checked between tensors
/// and inside chunked loops, plus a progress sink. Synchronous by
/// design; a TUI runs operations on a worker thread with the same `Cx`.
pub struct Cx<'a> {
    pub(crate) cancelled: &'a AtomicBool,
    progress: &'a mut (dyn FnMut(Progress) + 'a),
    temp_dir: PathBuf,
    last_phase: Option<Phase>,
    last_bytes_done: u64,
}

impl<'a> Cx<'a> {
    /// A context whose private input snapshots are made in the
    /// system temporary directory (`std::env::temp_dir()`).
    pub fn new(cancelled: &'a AtomicBool, progress: &'a mut (dyn FnMut(Progress) + 'a)) -> Self {
        Self::new_with_temp_dir(cancelled, progress, std::env::temp_dir())
    }

    /// Like [`Self::new`], but private input snapshots are made in
    /// `temp_dir`, which needs room for a copy of each input a
    /// operation holds at once (on Linux the system default is
    /// often tmpfs, i.e. RAM).
    pub fn new_with_temp_dir(
        cancelled: &'a AtomicBool,
        progress: &'a mut (dyn FnMut(Progress) + 'a),
        temp_dir: PathBuf,
    ) -> Self {
        Self {
            cancelled,
            progress,
            temp_dir,
            last_phase: None,
            last_bytes_done: 0,
        }
    }

    /// The directory private input snapshots are made in.
    pub(crate) fn temp_dir(&self) -> &Path {
        &self.temp_dir
    }

    pub(crate) fn check(&self) -> Result<(), ToolError> {
        check_cancelled(self.cancelled)
    }

    fn emit(
        &mut self,
        operation: &'static str,
        phase: Phase,
        tensor: Option<String>,
        done: u64,
        total: u64,
    ) {
        self.last_phase = Some(phase);
        (self.progress)(Progress {
            operation,
            phase,
            tensor,
            tensors_done: done,
            tensors_total: total,
            bytes_done: 0,
            bytes_total: 0,
        });
    }

    /// Chunk-granular byte progress, throttled: at most one report
    /// per MiB per phase, plus the phase's first and final reports.
    pub(crate) fn emit_bytes(
        &mut self,
        operation: &'static str,
        phase: Phase,
        bytes_done: u64,
        bytes_total: u64,
    ) {
        const THROTTLE_BYTES: u64 = 1024 * 1024;
        let phase_changed = self.last_phase != Some(phase);
        let due = phase_changed
            || bytes_done == 0
            || bytes_done >= bytes_total
            || bytes_done.saturating_sub(self.last_bytes_done) >= THROTTLE_BYTES;
        if !due {
            return;
        }
        self.last_phase = Some(phase);
        self.last_bytes_done = bytes_done;
        (self.progress)(Progress {
            operation,
            phase,
            tensor: None,
            tensors_done: 0,
            tensors_total: 0,
            bytes_done,
            bytes_total,
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
