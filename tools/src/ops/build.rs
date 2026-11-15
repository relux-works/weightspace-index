use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::Path;

use safetensors::tensor::TensorView;
use sha2::{Digest, Sha256};

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::{OutputTarget, Sha256Hex, TempFile};
use crate::patch::{
    ChangedTensor, Manifest, Representation, choose_representations, diff_tensor, read_manifest,
};

use super::{BuildReplacement, Cx, Report};

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
    let out = OutputTarget::check(&req.out)?;
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

    let mut temp = TempFile::create_in(out.parent())?;
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
    temp.publish(&out)?;
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
