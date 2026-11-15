use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::{
    OutputTarget, TempDir, check_input_dir, hash_file, is_within, sync_dir, write_at,
};
use crate::patch::{PatchTensor, for_each_index_block, parse_patch};

use super::{ApplyReplacement, Cx, Report};
use crate::model_dir::{copy_model_dir, single_weights_file};

pub fn apply_replacement(req: &ApplyReplacement, cx: &mut Cx) -> Result<Report, ToolError> {
    let out = OutputTarget::check(&req.out)?;
    check_input_dir(&req.model)?;
    // The output (and its sibling temp directory) must not live inside
    // the model being copied, or the copy would try to copy itself.
    // `out.parent()` is already canonical; containment compares
    // components ASCII-case-insensitively on every OS.
    let model_canonical = fs::canonicalize(&req.model).map_err(|e| ToolError::io(&req.model, e))?;
    if is_within(out.parent(), &model_canonical) {
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
    let temp = TempDir::create_in(out.parent())?;
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
        sync_dir(dir)?;
    }
    let out_sha256 = hash_file(&target_path, cx.cancelled)?;
    cx.check()?;
    temp.publish(&out)?;
    Ok(Report {
        tensors_changed: patch.tensors.len() as u64,
        entries_changed,
        out_sha256,
        out_bytes,
    })
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
