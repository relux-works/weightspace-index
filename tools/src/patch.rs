//! Patch planning: replacement-file parsing and validation, the sparse
//! index stream, the base/edited tensor diff, the full-vs-sparse stored
//! byte comparison, and the authors' manifest shape. No float
//! arithmetic anywhere: comparison and assignment are raw stored bytes.

mod diff;
mod entry;
mod file;
mod index;
mod manifest;
mod plan;

pub use diff::{TensorDiff, diff_tensor};
pub use entry::{EntryKind, parse_entry_name};
pub(crate) use file::validate_replacement_streaming;
pub use file::{
    PatchFile, PatchTensor, check_replacement_file, parse_patch, parse_patch_in,
    parse_patch_in_with_progress,
};
pub use index::for_each_index_block;
pub use manifest::{Manifest, ManifestTensor, read_manifest};
pub use plan::{ChangedTensor, Representation, choose_representation, choose_representations};
