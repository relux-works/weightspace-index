use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use sha2::{Digest, Sha256};

use crate::files::{
    OutputTarget, Sha256Hex, TempDir, check_input_dir, hash_file_with_progress, is_within,
    sync_dir, write_at,
};
use crate::patch::{PatchTensor, for_each_index_block, parse_patch_in_with_progress};

use super::{ApplyReplacement, Cx, Phase, Report};
use crate::model_dir::{copy_model_dir, single_weights_file};

pub fn apply_replacement(req: &ApplyReplacement, cx: &mut Cx) -> Result<Report, ToolError> {
    let out = check_output(req)?;
    let (base_file_relative, base_path) = locate_weights(req)?;
    let (patch, base, entries_changed) = verify_patch_against_base(req, &base_path, cx)?;
    let (temp, target_path, copied_dirs) = stage_copy(req, &out, &base, &base_file_relative, cx)?;
    let out_bytes = write_changes(&patch, &base, &target_path, cx)?;
    let out_sha256 = publish(temp, &out, &target_path, &copied_dirs, cx)?;
    Ok(Report {
        tensors_changed: patch.tensors.len() as u64,
        entries_changed,
        out_sha256,
        out_bytes,
    })
}

fn check_output(req: &ApplyReplacement) -> Result<OutputTarget, ToolError> {
    let out = OutputTarget::check(&req.out)?;
    check_input_dir(&req.model)?;
    // The output (and its sibling temp directory) must not live inside
    // the model being copied, or the copy would try to copy itself.
    // `out.parent()` is already canonical; containment compares
    // components ASCII-case-insensitively on every OS.
    let model_canonical = fs::canonicalize(&req.model).map_err(|e| ToolError::io(&req.model, e))?;
    if is_within(out.parent(), &model_canonical) {
        return Err(ToolError::OutputInsideInput {
            output: req.out.clone(),
            model: req.model.clone(),
        });
    }
    Ok(out)
}

fn locate_weights(req: &ApplyReplacement) -> Result<(PathBuf, PathBuf), ToolError> {
    let base_file_relative = single_weights_file(&req.model)?;
    let base_path = req.model.join(&base_file_relative);
    Ok((base_file_relative, base_path))
}

fn verify_patch_against_base(
    req: &ApplyReplacement,
    base_path: &Path,
    cx: &mut Cx,
) -> Result<(crate::patch::PatchFile, Checkpoint, u64), ToolError> {
    let patch_len = std::fs::metadata(&req.patch).map(|m| m.len()).unwrap_or(0);
    cx.emit_bytes("apply", Phase::Snapshot, 0, patch_len);
    let temp_dir = cx.temp_dir().to_path_buf();
    let patch =
        parse_patch_in_with_progress(&req.patch, &temp_dir, cx.cancelled, &mut |done, total| {
            cx.emit_bytes("apply", Phase::Snapshot, done, total);
        })?;
    cx.emit_bytes("apply", Phase::Snapshot, patch_len, patch_len);
    let base_len = std::fs::metadata(base_path).map(|m| m.len()).unwrap_or(0);
    cx.emit_bytes("apply", Phase::Snapshot, 0, base_len);
    let base = Checkpoint::open_in_with_progress(
        base_path,
        &temp_dir,
        cx.cancelled,
        &mut |done, total| {
            cx.emit_bytes("apply", Phase::Snapshot, done, total);
        },
    )?;
    cx.emit_bytes("apply", Phase::Snapshot, base_len, base_len);
    validate_patch_against_base(&patch, &base, req)?;

    // Wrong-base verification against the base snapshot bytes — the
    // exact bytes parsed — before anything is copied or published.
    // The staged copy is later written from this same snapshot
    // mapping (see `stage_copy`), so no re-read of the original
    // file needs to be compared against it.
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
                // Byte progress inside the (potentially long)
                // wrong-base hash, under the Verify phase whose
                // per-tensor counts this loop already reports.
                let actual =
                    base.tensor_hash_with_progress(base_meta, cx.cancelled, &mut |done, total| {
                        cx.emit_bytes("apply", Phase::Verify, done, total);
                    })?;
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
        cx.emit("apply", Phase::Verify, Some(name.clone()), done, total);
    }
    Ok((patch, base, entries_changed))
}

fn stage_copy(
    req: &ApplyReplacement,
    out: &OutputTarget,
    base: &Checkpoint,
    base_file_relative: &Path,
    cx: &mut Cx,
) -> Result<(TempDir, PathBuf, Vec<PathBuf>), ToolError> {
    // Copy the model directory (byte-for-byte, symlink-resolved,
    // materialized regular files) into a sibling temporary
    // directory, except the weights file: the staged weights are
    // written from the verified snapshot mapping below, hashing
    // while writing. The original weights file is never re-read
    // after verification, and no separate hash pass over either
    // the original or the staged copy is needed: the bytes written
    // are the snapshot bytes by construction, and the
    // hash-while-writing comparison against `snapshot_sha256`
    // catches any divergence between the two.
    let temp = TempDir::create_in(out.parent())?;
    let target_path = temp.path().join(base_file_relative);
    let mut copied_dirs = Vec::new();
    cx.emit_bytes("apply", Phase::Copy, 0, 0);
    copy_model_dir(
        &req.model,
        temp.path(),
        cx,
        &mut copied_dirs,
        Some(&target_path),
    )?;

    let snapshot = base.snapshot_bytes();
    let total = snapshot.len() as u64;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target_path)
        .map_err(|e| ToolError::io(&target_path, e))?;
    let mut hasher = Sha256::new();
    let mut done = 0u64;
    cx.emit_bytes("apply", Phase::Copy, 0, total);
    for chunk in snapshot.chunks(crate::files::CHUNK_BYTES) {
        cx.check()?;
        file.write_all(chunk)
            .map_err(|e| ToolError::io(&target_path, e))?;
        hasher.update(chunk);
        done += chunk.len() as u64;
        cx.emit_bytes("apply", Phase::Copy, done, total);
    }
    file.flush().map_err(|e| ToolError::io(&target_path, e))?;
    file.sync_all()
        .map_err(|e| ToolError::io(&target_path, e))?;
    drop(file);
    if Sha256Hex::from_digest(&hasher.finalize()) != *base.snapshot_sha256() {
        return Err(ToolError::internal(format!(
            "{}: staged copy hash differs from the verified snapshot; nothing published",
            target_path.display()
        )));
    }
    Ok((temp, target_path, copied_dirs))
}

fn write_changes(
    patch: &crate::patch::PatchFile,
    base: &Checkpoint,
    target_path: &Path,
    cx: &mut Cx,
) -> Result<u64, ToolError> {
    let mut target = OpenOptions::new()
        .read(true)
        .write(true)
        .open(target_path)
        .map_err(|e| ToolError::io(target_path, e))?;
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
                    .map_err(|e| ToolError::io(target_path, e))?;
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
                        .map_err(|e| ToolError::io(target_path, e))?;
                    written += n as u64;
                    cx.emit_bytes("apply", Phase::Patch, written, meta.byte_len());
                }
            }
            PatchTensor::Sparse {
                idx, new, count, ..
            } => {
                let elem = base_meta.elem_bytes();
                let patch_ck = &patch.checkpoint;
                let mut values: Vec<u8> = Vec::new();
                let total_bytes = *count * elem;
                cx.emit_bytes("apply", Phase::Patch, 0, total_bytes);
                // Cancellation is checked per index block (bounded
                // at `INDEX_BLOCK` entries) both by the stream and
                // here, and byte progress is reported per block;
                // the per-element writes inside one block are the
                // bounded remainder.
                for_each_index_block(patch_ck, name, idx, cx.cancelled, |block, ordinal| {
                    cx.check()?;
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
                            target_path,
                            at,
                            &values[i * elem_usize..(i + 1) * elem_usize],
                        )?;
                    }
                    cx.emit_bytes(
                        "apply",
                        Phase::Patch,
                        (ordinal + block.len() as u64) * elem,
                        total_bytes,
                    );
                    Ok(())
                })?;
            }
        }
    }
    target.flush().map_err(|e| ToolError::io(target_path, e))?;
    target
        .sync_all()
        .map_err(|e| ToolError::io(target_path, e))?;
    let out_bytes = target
        .metadata()
        .map_err(|e| ToolError::io(target_path, e))?
        .len();
    drop(target);
    Ok(out_bytes)
}

fn publish(
    temp: TempDir,
    out: &OutputTarget,
    target_path: &Path,
    copied_dirs: &[PathBuf],
    cx: &mut Cx,
) -> Result<Sha256Hex, ToolError> {
    for dir in copied_dirs.iter().rev() {
        sync_dir(dir)?;
    }
    let len = std::fs::metadata(target_path).map(|m| m.len()).unwrap_or(0);
    cx.emit_bytes("apply", Phase::Hash, 0, len);
    let out_sha256 = hash_file_with_progress(target_path, cx.cancelled, &mut |done, total| {
        cx.emit_bytes("apply", Phase::Hash, done, total);
    })?;
    cx.emit_bytes("apply", Phase::Hash, len, len);
    cx.check()?;
    cx.emit_bytes("apply", Phase::Publish, 0, 0);
    temp.publish(out)?;
    Ok(out_sha256)
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
