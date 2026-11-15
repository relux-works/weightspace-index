use std::collections::BTreeMap;
use std::path::Path;

use safetensors::Dtype;

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;

use super::entry::{EntryKind, parse_entry_name};

/// One patched tensor in a parsed replacement file.
#[derive(Clone, Debug)]
pub enum PatchTensor {
    Full {
        meta: TensorMeta,
    },
    Sparse {
        idx: TensorMeta,
        new: TensorMeta,
        old: TensorMeta,
        /// Number of sparse entries (`#idx` length).
        count: u64,
    },
}

/// A replacement file, parsed and cross-validated (sparse index values
/// themselves are validated while streaming, see [`for_each_index_block`]).
pub struct PatchFile {
    pub checkpoint: Checkpoint,
    pub tensors: BTreeMap<String, PatchTensor>,
}

pub fn parse_patch(path: &Path) -> Result<PatchFile, ToolError> {
    // Patch files take the same snapshot-backed official reader
    // path as checkpoints; there is no separate patch parser.
    let checkpoint = Checkpoint::open(path)?;
    let malformed = |detail: String| ToolError::malformed(path, detail);
    let mut full: BTreeMap<String, TensorMeta> = BTreeMap::new();
    let mut sparse: BTreeMap<String, [Option<TensorMeta>; 3]> = BTreeMap::new();
    for (name, meta) in &checkpoint.tensors {
        let (base, kind) = parse_entry_name(path, name)?;
        match kind {
            EntryKind::Full => drop(full.insert(base.to_owned(), meta.clone())),
            EntryKind::Idx => sparse.entry(base.to_owned()).or_default()[0] = Some(meta.clone()),
            EntryKind::New => sparse.entry(base.to_owned()).or_default()[1] = Some(meta.clone()),
            EntryKind::Old => sparse.entry(base.to_owned()).or_default()[2] = Some(meta.clone()),
        }
    }
    if full.is_empty() && sparse.is_empty() {
        return Err(malformed("no tensors".into()));
    }
    let mut tensors: BTreeMap<String, PatchTensor> = BTreeMap::new();
    for (base, meta) in full {
        tensors.insert(base, PatchTensor::Full { meta });
    }
    for (base, trio) in sparse {
        if tensors.contains_key(&base) {
            return Err(malformed(format!("{base}: both full and sparse")));
        }
        let [Some(idx), Some(new), Some(old)] = trio else {
            return Err(malformed(format!(
                "{base}: sparse tensors need #idx, #new and #old"
            )));
        };
        let count = match idx.shape.as_slice() {
            [n] if *n >= 1 && idx.dtype == Dtype::I64 && idx.byte_len() == 8 * *n => *n,
            _ => {
                return Err(malformed(format!(
                    "{base}#idx: must be a non-empty 1-D I64 tensor"
                )));
            }
        };
        // No legacy `#old` shape alias: the official reader has
        // already refused any entry whose stored bytes do not match
        // its declared shape, and here `#old` must declare exactly
        // one value per sparse index, like `#new`.
        if new.dtype != old.dtype || new.shape != [count] || old.shape != [count] {
            return Err(malformed(format!(
                "{base}: #new and #old must match #idx in shape and each other in dtype"
            )));
        }
        tensors.insert(
            base,
            PatchTensor::Sparse {
                idx,
                new,
                old,
                count,
            },
        );
    }
    Ok(PatchFile {
        checkpoint,
        tensors,
    })
}

/// Rejects a descriptor no checkpoint in this crate can produce —
/// inverted byte offsets or a sub-byte dtype whose
/// `elem_bytes()` is 0 — with a typed error, so the public diff
/// and index-stream entry points below never panic (for example
/// dividing by a zero element size) on a caller-constructed
/// [`TensorMeta`]. File-derived descriptors always pass: the
/// official reader enforces `start <= end` and `Checkpoint::open`
/// refuses sub-byte dtypes.
pub(super) fn check_meta(path: &Path, tensor: &str, meta: &TensorMeta) -> Result<(), ToolError> {
    if meta.offsets_inverted() {
        return Err(ToolError::malformed(
            path,
            format!(
                "{tensor}: inverted data offsets ({}, {})",
                meta.data_offsets.0, meta.data_offsets.1
            ),
        ));
    }
    if meta.elem_bytes() == 0 {
        return Err(ToolError::UnsupportedDtype {
            path: path.to_path_buf(),
            tensor: tensor.to_owned(),
            dtype: meta.dtype,
        });
    }
    Ok(())
}
