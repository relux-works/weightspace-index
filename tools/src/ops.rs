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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use safetensors::tensor::TensorView;
use sha2::{Digest, Sha256};

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::{
    Sha256Hex, TempDir, TempFile, check_cancelled, check_input_dir, check_output_path, copy_range,
    fsync_dir, hash_file, open_input, parent_dir, write_at,
};
use crate::patch::{
    ChangedTensor, Manifest, PatchTensor, Representation, choose_representations, diff_tensor,
    for_each_index_block, parse_patch, read_manifest,
};

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
    cancelled: &'a AtomicBool,
    progress: &'a mut (dyn FnMut(Progress) + 'a),
}

impl<'a> Cx<'a> {
    pub fn new(cancelled: &'a AtomicBool, progress: &'a mut (dyn FnMut(Progress) + 'a)) -> Self {
        Self {
            cancelled,
            progress,
        }
    }

    fn check(&self) -> Result<(), ToolError> {
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

struct OwnedTensor {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn shape_usize(path: &Path, name: &str, shape: &[u64]) -> Result<Vec<usize>, ToolError> {
    shape
        .iter()
        .map(|&d| {
            usize::try_from(d)
                .map_err(|_| ToolError::malformed(path, format!("{name}: shape too large")))
        })
        .collect()
}

pub fn build_replacement(req: &BuildReplacement, cx: &mut Cx) -> Result<Report, ToolError> {
    check_output_path(&req.out)?;
    let base = Checkpoint::open(&req.base)?;
    let edited = Checkpoint::open(&req.edited)?;
    check_same_tensor_set(&base, &edited)?;

    let manifest = req.manifest.as_deref().map(read_manifest).transpose()?;
    if let Some(manifest) = &manifest {
        verify_manifest_hashes(manifest, &base, &edited, cx)?;
    }

    let total = base.tensors.len() as u64;
    let mut done = 0u64;
    let mut changed_names: BTreeSet<String> = BTreeSet::new();
    // (name, numel, changed) per changed tensor, in base order. The
    // count scan materializes nothing; representations are decided
    // for the whole file at once, because the serializer-bound size
    // of one tensor's entries depends on the others' layout.
    let mut plan: Vec<(String, u64, u64)> = Vec::new();

    for (name, base_meta) in &base.tensors {
        cx.check()?;
        let edited_meta = edited.tensor(name)?;
        let numel = base_meta.numel(&base.path, name)?;
        let count = diff_tensor(
            name,
            &base,
            base_meta,
            &edited,
            edited_meta,
            cx.cancelled,
            false,
        )?
        .changed;
        done += 1;
        cx.emit("replacement", Some(name.clone()), done, total);
        if count == 0 {
            continue;
        }
        changed_names.insert(name.clone());
        plan.push((name.clone(), numel, count));
    }

    if changed_names.is_empty() {
        return Err(ToolError::refused(
            "base and edited are identical; no replacement to write",
        ));
    }
    if let Some(manifest) = &manifest {
        let listed: BTreeSet<&String> = manifest.tensors.keys().collect();
        let changed: BTreeSet<&String> = changed_names.iter().collect();
        if listed != changed {
            let missing: Vec<_> = changed.difference(&listed).collect();
            let extra: Vec<_> = listed.difference(&changed).collect();
            return Err(ToolError::Manifest {
                path: manifest.path.clone(),
                detail: format!(
                    "manifest tensor set differs from the changed tensors \
                     (missing {missing:?}, extra {extra:?})"
                ),
            });
        }
    }

    let mut candidates: Vec<ChangedTensor<'_>> = Vec::with_capacity(plan.len());
    for (name, numel, count) in &plan {
        let meta = base.tensor(name)?;
        candidates.push(ChangedTensor {
            name,
            dtype: meta.dtype,
            shape: &meta.shape,
            numel: *numel,
            changed: *count,
        });
    }
    let choices = choose_representations(&candidates)?;

    let mut entries: BTreeMap<String, OwnedTensor> = BTreeMap::new();
    let mut entries_changed = 0u64;
    for ((name, numel, count), choice) in plan.iter().zip(&choices) {
        cx.check()?;
        let (numel, count) = (*numel, *count);
        let base_meta = base.tensor(name)?;
        let edited_meta = edited.tensor(name)?;
        // Only a sparse winner pays for a second, collecting scan.
        match choice {
            Representation::Sparse => {
                entries_changed += count;
                let diff = diff_tensor(
                    name,
                    &base,
                    base_meta,
                    &edited,
                    edited_meta,
                    cx.cancelled,
                    true,
                )?;
                let (idx, new, old) = diff.sparse.ok_or_else(|| {
                    ToolError::refused(format!("{name}: tensor changed while diffing"))
                })?;
                if diff.changed != count {
                    return Err(ToolError::refused(format!(
                        "{name}: tensor changed while diffing"
                    )));
                }
                let n = usize::try_from(count)
                    .map_err(|_| ToolError::refused(format!("{name}: too many changes")))?;
                entries.insert(
                    format!("{name}#idx"),
                    OwnedTensor {
                        dtype: safetensors::Dtype::I64,
                        shape: vec![n],
                        data: idx,
                    },
                );
                entries.insert(
                    format!("{name}#new"),
                    OwnedTensor {
                        dtype: base_meta.dtype,
                        shape: vec![n],
                        data: new,
                    },
                );
                entries.insert(
                    format!("{name}#old"),
                    OwnedTensor {
                        dtype: base_meta.dtype,
                        shape: vec![n],
                        data: old,
                    },
                );
            }
            Representation::Full => {
                entries_changed += numel;
                // The replacement file stores the whole tensor, so the
                // edited range must be materialized; read it in chunks.
                let mut data = Vec::new();
                copy_tensor_bytes(&edited, edited_meta, &mut data, cx)?;
                entries.insert(
                    name.clone(),
                    OwnedTensor {
                        dtype: edited_meta.dtype,
                        shape: shape_usize(&edited.path, name, &edited_meta.shape)?,
                        data,
                    },
                );
            }
        }
    }

    // Serialize the (small) replacement file itself with the official
    // writer. Determinism: the crate sorts tensors by dtype alignment
    // then name and writes one metadata key, so identical inputs give
    // identical bytes; a multi-key metadata map would serialize in hash
    // order, so edited-file metadata is not copied across. The single
    // key matches the example replacement file's `__metadata__`.
    let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(entries.len());
    for (name, tensor) in &entries {
        let view =
            TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).map_err(|e| {
                ToolError::Safetensors {
                    path: req.out.clone(),
                    source: e,
                }
            })?;
        views.push((name.clone(), view));
    }
    let mut file_metadata: HashMap<String, String> = HashMap::new();
    file_metadata.insert("format".into(), "pt".into());
    let bytes = safetensors::serialize(
        views.iter().map(|(name, view)| (name.as_str(), view)),
        Some(file_metadata),
    )
    .map_err(|e| ToolError::Safetensors {
        path: req.out.clone(),
        source: e,
    })?;
    let out_sha256 = Sha256Hex::from_digest(&Sha256::digest(&bytes));
    let out_bytes = bytes.len() as u64;

    let mut temp = TempFile::create_in(&parent_dir(&req.out))?;
    temp.file_mut()
        .write_all(&bytes)
        .map_err(|e| ToolError::io(temp.path(), e))?;
    temp.sync()?;
    // The generated replacement must satisfy the index's own format gate
    // before anything is published under the final name.
    crate::layout::check_replacement_file(temp.path()).map_err(|e| {
        ToolError::refused(format!(
            "generated replacement failed check_replacement_file: {e}"
        ))
    })?;
    cx.check()?;
    temp.publish(&req.out)?;
    Ok(Report {
        tensors_changed: changed_names.len() as u64,
        entries_changed,
        out_sha256,
        out_bytes,
    })
}

/// Reads one whole tensor into memory, one bounded chunk at a time
/// over the snapshot mapping. Used only where the replacement
/// format itself stores the whole tensor (the full representation
/// in the output file).
fn copy_tensor_bytes(
    checkpoint: &Checkpoint,
    meta: &TensorMeta,
    out: &mut Vec<u8>,
    cx: &Cx,
) -> Result<(), ToolError> {
    let mut left = meta.byte_len();
    let mut offset = 0u64;
    let mut buf = vec![0u8; crate::files::CHUNK_BYTES];
    while left > 0 {
        cx.check()?;
        let n =
            usize::try_from(left.min(crate::files::CHUNK_BYTES as u64)).expect("chunk fits usize");
        checkpoint.read_tensor_at(meta, offset, &mut buf[..n])?;
        out.extend_from_slice(&buf[..n]);
        offset += n as u64;
        left -= n as u64;
    }
    Ok(())
}

fn check_same_tensor_set(base: &Checkpoint, edited: &Checkpoint) -> Result<(), ToolError> {
    for name in base.tensors.keys() {
        if name.contains('#') {
            return Err(ToolError::malformed(
                &base.path,
                format!("{name}: '#' is reserved for sparse patch entries"),
            ));
        }
    }
    let base_names: BTreeSet<&String> = base.tensors.keys().collect();
    let edited_names: BTreeSet<&String> = edited.tensors.keys().collect();
    if base_names != edited_names {
        let only_base: Vec<_> = base_names.difference(&edited_names).collect();
        let only_edited: Vec<_> = edited_names.difference(&base_names).collect();
        return Err(ToolError::refused(format!(
            "tensor name sets differ (only in base {only_base:?}, only in edited {only_edited:?})"
        )));
    }
    for (name, b) in &base.tensors {
        let e = edited.tensor(name)?;
        if b.shape != e.shape || b.dtype != e.dtype {
            return Err(ToolError::refused(format!(
                "tensor {name}: shape/dtype differ between base ({:?} {:?}) \
                 and edited ({:?} {:?})",
                b.shape, b.dtype, e.shape, e.dtype
            )));
        }
    }
    Ok(())
}

fn verify_manifest_hashes(
    manifest: &Manifest,
    base: &Checkpoint,
    edited: &Checkpoint,
    cx: &Cx,
) -> Result<(), ToolError> {
    let fail = |detail: String| ToolError::Manifest {
        path: manifest.path.clone(),
        detail,
    };
    for (name, listed) in &manifest.tensors {
        cx.check()?;
        let (b, e) = match (base.tensors.get(name), edited.tensors.get(name)) {
            (Some(b), Some(e)) => (b, e),
            _ => return Err(fail(format!("{name}: tensor missing from a checkpoint"))),
        };
        if b.shape != listed.shape || e.shape != listed.shape {
            return Err(fail(format!("{name}: shape differs from the manifest")));
        }
        if b.dtype != listed.dtype || e.dtype != listed.dtype {
            return Err(fail(format!("{name}: dtype differs from the manifest")));
        }
        let base_hash = base.tensor_hash(b, cx.cancelled)?;
        if base_hash != listed.base_sha256 {
            return Err(fail(format!(
                "{name}: base_sha256 mismatch (manifest {}, computed {base_hash})",
                listed.base_sha256
            )));
        }
        let edited_hash = edited.tensor_hash(e, cx.cancelled)?;
        if edited_hash != listed.edited_sha256 {
            return Err(fail(format!(
                "{name}: edited_sha256 mismatch (manifest {}, computed {edited_hash})",
                listed.edited_sha256
            )));
        }
    }
    Ok(())
}

pub fn apply_replacement(req: &ApplyReplacement, cx: &mut Cx) -> Result<Report, ToolError> {
    check_output_path(&req.out)?;
    check_input_dir(&req.model)?;
    // The output (and its sibling temp directory) must not live inside
    // the model being copied, or the copy would try to copy itself.
    let model_canonical = fs::canonicalize(&req.model).map_err(|e| ToolError::io(&req.model, e))?;
    let out_parent = parent_dir(&req.out);
    let out_parent_canonical =
        fs::canonicalize(&out_parent).map_err(|e| ToolError::io(&out_parent, e))?;
    if out_parent_canonical.starts_with(&model_canonical) {
        return Err(ToolError::refused(format!(
            "{}: output must not be inside the model directory {}",
            req.out.display(),
            req.model.display()
        )));
    }
    let base_file_relative = single_weights_file(&req.model)?;
    let base_path = req.model.join(&base_file_relative);

    let patch = parse_patch(&req.patch)?;
    let base = Checkpoint::open(&base_path)?;
    validate_patch_against_base(&patch, &base, req)?;

    // Wrong-base verification against the base snapshot bytes — the
    // exact bytes parsed — before anything is copied or published.
    // The original file is hash-checked against that snapshot
    // below, before the copy is made.
    let total = patch.tensors.len() as u64;
    let mut done = 0u64;
    let mut entries_changed = 0u64;
    for (name, tensor) in &patch.tensors {
        cx.check()?;
        let base_meta = base.tensor(name)?;
        match tensor {
            PatchTensor::Full { .. } => {
                let expected = req.expect_tensor_sha256.get(name).ok_or_else(|| {
                    ToolError::MissingExpectedDigest {
                        tensor: name.clone(),
                    }
                })?;
                let actual = base.tensor_hash(base_meta, cx.cancelled)?;
                if actual != *expected {
                    return Err(ToolError::WrongBase {
                        tensor: name.clone(),
                        detail: format!("expected base sha256 {expected}, computed {actual}"),
                    });
                }
                entries_changed += base_meta.numel(&base.path, name)?;
            }
            PatchTensor::Sparse {
                idx, old, count, ..
            } => {
                verify_sparse_old(&patch.checkpoint, name, idx, old, &base, base_meta, cx)?;
                entries_changed += *count;
            }
        }
        done += 1;
        cx.emit("apply", Some(name.clone()), done, total);
    }

    // Copy the whole model directory (byte-for-byte, symlink-resolved,
    // materialized regular files) into a sibling temporary directory,
    // patch the copy's weights in place by ranged writes, then publish
    // by rename. Verification above ran on the base snapshot bytes;
    // the original file must still be those exact bytes, and the copy
    // is hash-checked against them before any ranged write.
    let source_hash = hash_file(&base_path, cx.cancelled)?;
    if source_hash != *base.snapshot_sha256() {
        return Err(ToolError::refused(format!(
            "{}: base file changed after it was snapshot-verified; nothing published",
            base_path.display()
        )));
    }
    let temp = TempDir::create_in(&parent_dir(&req.out))?;
    let mut copied_dirs = Vec::new();
    copy_model_dir(&req.model, temp.path(), cx, &mut copied_dirs)?;
    let target_path = temp.path().join(&base_file_relative);
    if hash_file(&target_path, cx.cancelled)? != source_hash {
        return Err(ToolError::refused(format!(
            "{}: copy differs from the base file; nothing published",
            target_path.display()
        )));
    }

    let mut target = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&target_path)
        .map_err(|e| ToolError::io(&target_path, e))?;
    for (name, tensor) in &patch.tensors {
        cx.check()?;
        let base_meta = base.tensor(name)?;
        let target_start = base.data_start() + base_meta.data_offsets.0;
        match tensor {
            PatchTensor::Full { meta } => {
                // Stream the full tensor from the patch snapshot
                // mapping into the copied file, one bounded chunk
                // at a time at the tensor's target offset.
                target
                    .seek(SeekFrom::Start(target_start))
                    .map_err(|e| ToolError::io(&target_path, e))?;
                let mut written = 0u64;
                let mut buf = vec![0u8; crate::files::CHUNK_BYTES];
                while written < meta.byte_len() {
                    cx.check()?;
                    let n = usize::try_from(
                        (meta.byte_len() - written).min(crate::files::CHUNK_BYTES as u64),
                    )
                    .expect("chunk fits usize");
                    patch
                        .checkpoint
                        .read_tensor_at(meta, written, &mut buf[..n])?;
                    target
                        .write_all(&buf[..n])
                        .map_err(|e| ToolError::io(&target_path, e))?;
                    written += n as u64;
                }
            }
            PatchTensor::Sparse { idx, new, .. } => {
                let elem = base_meta.elem_bytes();
                let patch_ck = &patch.checkpoint;
                let mut values: Vec<u8> = Vec::new();
                for_each_index_block(patch_ck, name, idx, cx.cancelled, |block, ordinal| {
                    check_index_block(name, block, base_meta, &base.path)?;
                    let block_bytes = block.len() as u64 * elem;
                    values.clear();
                    values.resize(usize::try_from(block_bytes).expect("block fits usize"), 0);
                    patch_ck.read_tensor_at(new, ordinal * elem, &mut values)?;
                    let elem_usize = usize::try_from(elem).expect("element fits usize");
                    for (i, &index) in block.iter().enumerate() {
                        let at = target_start + (index as u64) * elem;
                        write_at(
                            &mut target,
                            &target_path,
                            at,
                            &values[i * elem_usize..(i + 1) * elem_usize],
                        )?;
                    }
                    Ok(())
                })?;
            }
        }
    }
    target.flush().map_err(|e| ToolError::io(&target_path, e))?;
    target
        .sync_all()
        .map_err(|e| ToolError::io(&target_path, e))?;
    let out_bytes = target
        .metadata()
        .map_err(|e| ToolError::io(&target_path, e))?
        .len();
    drop(target);
    for dir in copied_dirs.iter().rev() {
        fsync_dir(dir)?;
    }
    let out_sha256 = hash_file(&target_path, cx.cancelled)?;
    cx.check()?;
    temp.publish(&req.out)?;
    Ok(Report {
        tensors_changed: patch.tensors.len() as u64,
        entries_changed,
        out_sha256,
        out_bytes,
    })
}

/// Slice 1 supports exactly one weights file at the model root:
/// `model.safetensors`. An index file or several safetensors files are
/// a sharded layout and are refused, never guessed at.
fn single_weights_file(model: &Path) -> Result<PathBuf, ToolError> {
    let sharded = |detail: String| ToolError::ShardedModel {
        path: model.to_path_buf(),
        detail,
    };
    if model.join("model.safetensors.index.json").exists() {
        return Err(sharded(
            "model.safetensors.index.json present (sharded checkpoint)".into(),
        ));
    }
    let mut weights = Vec::new();
    for entry in read_sorted_dir(model)? {
        if entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            weights.push(PathBuf::from(entry.file_name()));
        }
    }
    match weights.as_slice() {
        [only] if only.as_os_str() == "model.safetensors" => Ok(only.clone()),
        [] => Err(sharded("no root model.safetensors".into())),
        [only] => Err(sharded(format!(
            "root weights file is {}; slice 1 patches only model.safetensors",
            only.display()
        ))),
        many => Err(sharded(format!(
            "{} root .safetensors files (sharded checkpoint)",
            many.len()
        ))),
    }
}

fn validate_patch_against_base(
    patch: &crate::patch::PatchFile,
    base: &Checkpoint,
    req: &ApplyReplacement,
) -> Result<(), ToolError> {
    let mut full_names: BTreeSet<&String> = BTreeSet::new();
    for (name, tensor) in &patch.tensors {
        let base_meta = base.tensors.get(name).ok_or_else(|| ToolError::WrongBase {
            tensor: name.clone(),
            detail: "tensor not present in the base checkpoint".into(),
        })?;
        match tensor {
            PatchTensor::Full { meta } => {
                full_names.insert(name);
                if meta.shape != base_meta.shape || meta.dtype != base_meta.dtype {
                    return Err(ToolError::WrongBase {
                        tensor: name.clone(),
                        detail: format!(
                            "patch shape/dtype {:?} {:?} differ from base {:?} {:?}",
                            meta.shape, meta.dtype, base_meta.shape, base_meta.dtype
                        ),
                    });
                }
            }
            PatchTensor::Sparse { new, .. } => {
                if new.dtype != base_meta.dtype {
                    return Err(ToolError::WrongBase {
                        tensor: name.clone(),
                        detail: format!(
                            "sparse values dtype {:?} differ from base dtype {:?}",
                            new.dtype, base_meta.dtype
                        ),
                    });
                }
            }
        }
    }
    for name in &full_names {
        if !req.expect_tensor_sha256.contains_key(*name) {
            return Err(ToolError::MissingExpectedDigest {
                tensor: (*name).clone(),
            });
        }
    }
    for name in req.expect_tensor_sha256.keys() {
        if !full_names.contains(name) {
            return Err(ToolError::InvalidArgument {
                detail: format!(
                    "--expect-tensor-sha256 names {name}, which is not a full-form tensor in the patch"
                ),
            });
        }
    }
    Ok(())
}

fn check_index_block(
    tensor: &str,
    block: &[i64],
    base_meta: &TensorMeta,
    base_path: &Path,
) -> Result<(), ToolError> {
    let numel = base_meta.numel(base_path, tensor)?;
    if let Some(&index) = block.iter().find(|&&i| i as u64 >= numel) {
        return Err(ToolError::WrongBase {
            tensor: tensor.to_owned(),
            detail: format!("sparse index {index} is outside the base tensor ({numel} elements)"),
        });
    }
    Ok(())
}

/// Verifies every `#old` value against the base snapshot bytes at
/// its index, block by block, before anything is written.
fn verify_sparse_old(
    patch: &Checkpoint,
    name: &str,
    idx: &TensorMeta,
    old: &TensorMeta,
    base: &Checkpoint,
    base_meta: &TensorMeta,
    cx: &Cx,
) -> Result<(), ToolError> {
    let elem = base_meta.elem_bytes();
    let elem_usize = usize::try_from(elem).expect("element fits usize");
    let mut base_elem = vec![0u8; elem_usize];
    let mut old_values: Vec<u8> = Vec::new();
    for_each_index_block(patch, name, idx, cx.cancelled, |block, ordinal| {
        check_index_block(name, block, base_meta, &base.path)?;
        let block_bytes = block.len() as u64 * elem;
        old_values.clear();
        old_values.resize(usize::try_from(block_bytes).expect("block fits usize"), 0);
        patch.read_tensor_at(old, ordinal * elem, &mut old_values)?;
        for (i, &index) in block.iter().enumerate() {
            base.read_tensor_at(base_meta, (index as u64) * elem, &mut base_elem)?;
            let expected = &old_values[i * elem_usize..(i + 1) * elem_usize];
            if base_elem[..] != *expected {
                return Err(ToolError::WrongBase {
                    tensor: name.to_owned(),
                    detail: format!("#old value at flat index {index} does not match the base"),
                });
            }
        }
        Ok(())
    })
}

fn read_sorted_dir(dir: &Path) -> Result<Vec<fs::DirEntry>, ToolError> {
    let mut entries: Vec<fs::DirEntry> = fs::read_dir(dir)
        .map_err(|e| ToolError::io(dir, e))?
        .collect::<Result<_, _>>()
        .map_err(|e| ToolError::io(dir, e))?;
    entries.sort_by_key(|e| e.file_name());
    Ok(entries)
}

/// Copies a model directory recursively, byte-for-byte. File symlinks
/// are copied as the regular file they resolve to; directory symlinks
/// and special files are refused; nothing is written through a link.
fn copy_model_dir(
    src: &Path,
    dst: &Path,
    cx: &Cx,
    copied_dirs: &mut Vec<PathBuf>,
) -> Result<(), ToolError> {
    let mut stack: Vec<(PathBuf, PathBuf)> = vec![(src.to_path_buf(), dst.to_path_buf())];
    copied_dirs.push(dst.to_path_buf());
    while let Some((from, to)) = stack.pop() {
        for entry in read_sorted_dir(&from)? {
            cx.check()?;
            let entry_path = entry.path();
            let target = to.join(entry.file_name());
            let meta =
                fs::symlink_metadata(&entry_path).map_err(|e| ToolError::io(&entry_path, e))?;
            let kind = meta.file_type();
            if kind.is_symlink() {
                let (resolved, mut src_file) = open_input(&entry_path).map_err(|e| {
                    // open_input refuses non-regular resolved files; a
                    // symlinked directory inside the model gets the
                    // dedicated symlink refusal instead.
                    if fs::metadata(&entry_path).is_ok_and(|m| m.is_dir()) {
                        ToolError::SymlinkRefused {
                            path: entry_path.clone(),
                            detail: "input directory symlinks are refused".into(),
                        }
                    } else {
                        e
                    }
                })?;
                copy_one_file(&resolved, &mut src_file, &target, cx)?;
            } else if kind.is_dir() {
                fs::create_dir(&target).map_err(|e| ToolError::io(&target, e))?;
                copied_dirs.push(target.clone());
                stack.push((entry_path, target));
            } else if kind.is_file() {
                let mut src_file =
                    File::open(&entry_path).map_err(|e| ToolError::io(&entry_path, e))?;
                copy_one_file(&entry_path, &mut src_file, &target, cx)?;
            } else {
                return Err(ToolError::refused(format!(
                    "{}: special files are refused",
                    entry_path.display()
                )));
            }
        }
    }
    Ok(())
}

fn copy_one_file(src_path: &Path, src: &mut File, dst: &Path, cx: &Cx) -> Result<(), ToolError> {
    let len = src
        .metadata()
        .map_err(|e| ToolError::io(src_path, e))?
        .len();
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dst)
        .map_err(|e| ToolError::io(dst, e))?;
    copy_range(src, src_path, 0, len, &mut out, dst, cx.cancelled)?;
    out.flush().map_err(|e| ToolError::io(dst, e))?;
    out.sync_all().map_err(|e| ToolError::io(dst, e))
}
