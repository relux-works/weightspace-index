use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::Path;

use safetensors::tensor::{Metadata, TensorInfo, TensorView};
use sha2::{Digest, Sha256};

use crate::checkpoint::Checkpoint;
use crate::error::ToolError;
use crate::files::{OutputTarget, Sha256Hex, TempFile};
use crate::patch::{
    ChangedTensor, Manifest, Representation, choose_representations, diff_tensor, read_manifest,
    validate_replacement_streaming,
};

use super::{BuildReplacement, Cx, Phase, Report};

type ChangePlan = Vec<(String, u64, u64)>;

/// Stored bytes of one replacement entry. Full-form entries borrow
/// the edited checkpoint's snapshot mapping (zero-copy; the tensor
/// is never materialized into a heap buffer), while sparse entries
/// own the index/value arrays the format itself materializes.
enum EntryData<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl AsRef<[u8]> for EntryData<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }
}

struct EntryTensor<'a> {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: EntryData<'a>,
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
    let out = check_output(req)?;
    let (base, edited, manifest) = open_inputs(req, cx)?;
    let (plan, changed_names) = plan_changes(&base, &edited, manifest.as_ref(), cx)?;
    let (entries, entries_changed) = collect_entries(&base, &edited, &plan, cx)?;
    let (out_sha256, out_bytes) = serialize_and_gate(req, &out, &entries, cx)?;
    Ok(Report {
        tensors_changed: changed_names.len() as u64,
        entries_changed,
        out_sha256,
        out_bytes,
    })
}

fn check_output(req: &BuildReplacement) -> Result<OutputTarget, ToolError> {
    OutputTarget::check(&req.out)
}

fn open_inputs(
    req: &BuildReplacement,
    cx: &mut Cx,
) -> Result<(Checkpoint, Checkpoint, Option<Manifest>), ToolError> {
    let base_len = std::fs::metadata(&req.base).map(|m| m.len()).unwrap_or(0);
    cx.emit_bytes("replacement", Phase::Snapshot, 0, base_len);
    let temp_dir = cx.temp_dir().to_path_buf();
    let base = Checkpoint::open_in_with_progress(
        &req.base,
        &temp_dir,
        cx.cancelled,
        &mut |done, total| {
            cx.emit_bytes("replacement", Phase::Snapshot, done, total);
        },
    )?;
    cx.emit_bytes("replacement", Phase::Snapshot, base_len, base_len);
    let edited_len = std::fs::metadata(&req.edited).map(|m| m.len()).unwrap_or(0);
    cx.emit_bytes("replacement", Phase::Snapshot, 0, edited_len);
    let edited = Checkpoint::open_in_with_progress(
        &req.edited,
        &temp_dir,
        cx.cancelled,
        &mut |done, total| {
            cx.emit_bytes("replacement", Phase::Snapshot, done, total);
        },
    )?;
    cx.emit_bytes("replacement", Phase::Snapshot, edited_len, edited_len);
    check_same_tensor_set(&base, &edited)?;

    let manifest = req.manifest.as_deref().map(read_manifest).transpose()?;
    if let Some(manifest) = &manifest {
        verify_manifest_hashes(manifest, &base, &edited, cx)?;
    }
    Ok((base, edited, manifest))
}

fn plan_changes(
    base: &Checkpoint,
    edited: &Checkpoint,
    manifest: Option<&Manifest>,
    cx: &mut Cx,
) -> Result<(ChangePlan, BTreeSet<String>), ToolError> {
    let total = base.tensors.len() as u64;
    let mut done = 0u64;
    let mut changed_names: BTreeSet<String> = BTreeSet::new();
    // (name, numel, changed) per changed tensor, in base order. The
    // count scan materializes nothing; representations are decided
    // for the whole file at once, because the serializer-bound size
    // of one tensor's entries depends on the others' layout.
    let mut plan: ChangePlan = Vec::new();

    for (name, base_meta) in &base.tensors {
        cx.check()?;
        let edited_meta = edited.tensor(name)?;
        let numel = base_meta.numel(&base.path, name)?;
        let count = diff_tensor(
            name,
            base,
            base_meta,
            edited,
            edited_meta,
            cx.cancelled,
            false,
        )?
        .changed;
        done += 1;
        cx.emit("replacement", Phase::Plan, Some(name.clone()), done, total);
        if count == 0 {
            continue;
        }
        changed_names.insert(name.clone());
        plan.push((name.clone(), numel, count));
    }

    if changed_names.is_empty() {
        return Err(ToolError::NoChanges);
    }
    if let Some(manifest) = manifest {
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
    Ok((plan, changed_names))
}

fn collect_entries<'e>(
    base: &Checkpoint,
    edited: &'e Checkpoint,
    plan: &ChangePlan,
    cx: &mut Cx,
) -> Result<(BTreeMap<String, EntryTensor<'e>>, u64), ToolError> {
    let mut candidates: Vec<ChangedTensor<'_>> = Vec::with_capacity(plan.len());
    for (name, numel, count) in plan {
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

    let mut entries: BTreeMap<String, EntryTensor<'e>> = BTreeMap::new();
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
                    base,
                    base_meta,
                    edited,
                    edited_meta,
                    cx.cancelled,
                    true,
                )?;
                let (idx, new, old) = diff.sparse.ok_or_else(|| ToolError::InputChanged {
                    path: base.path.clone(),
                })?;
                if diff.changed != count {
                    return Err(ToolError::InputChanged {
                        path: base.path.clone(),
                    });
                }
                let n = usize::try_from(count)
                    .map_err(|_| ToolError::internal(format!("{name}: too many changes")))?;
                entries.insert(
                    format!("{name}#idx"),
                    EntryTensor {
                        dtype: safetensors::Dtype::I64,
                        shape: vec![n],
                        data: EntryData::Owned(idx),
                    },
                );
                entries.insert(
                    format!("{name}#new"),
                    EntryTensor {
                        dtype: base_meta.dtype,
                        shape: vec![n],
                        data: EntryData::Owned(new),
                    },
                );
                entries.insert(
                    format!("{name}#old"),
                    EntryTensor {
                        dtype: base_meta.dtype,
                        shape: vec![n],
                        data: EntryData::Owned(old),
                    },
                );
            }
            Representation::Full => {
                entries_changed += numel;
                // Zero-copy: the entry borrows the edited tensor's
                // bytes from its snapshot mapping; the serializer
                // streams them from there and no heap copy of the
                // tensor is ever made.
                entries.insert(
                    name.clone(),
                    EntryTensor {
                        dtype: edited_meta.dtype,
                        shape: shape_usize(&edited.path, name, &edited_meta.shape)?,
                        data: EntryData::Borrowed(edited.tensor_bytes(edited_meta)?),
                    },
                );
            }
        }
    }
    Ok((entries, entries_changed))
}

/// The `serialize` and `gate_and_publish` phases, fused only in
/// ownership: the streamed writer needs the temp file handle in
/// hand while serializing, so the temp file is created before the
/// `Serialize` reports and published inside the same call. Phase
/// names and order are still reported separately (`Serialize` while
/// streaming into the temp handle, then `Publish` after the
/// post-write gate), and cancellation is checked per written chunk
/// and again before publication.
fn serialize_and_gate(
    req: &BuildReplacement,
    out: &OutputTarget,
    entries: &BTreeMap<String, EntryTensor<'_>>,
    cx: &mut Cx,
) -> Result<(Sha256Hex, u64), ToolError> {
    let (temp, out_sha256, out_bytes) = serialize_to_temp(req, out, entries, cx)?;
    gate_temp_and_publish(temp, out, cx)?;
    Ok((out_sha256, out_bytes))
}

/// Streams the replacement file into a fresh temp file, hashing
/// while writing; no whole-file buffer is materialized. The byte
/// layout reproduces `safetensors::serialize` (0.8.0) exactly,
/// using only its public types: entries are sorted by descending
/// dtype alignment then name (the crate's `prepare` order), data
/// offsets are assigned contiguously in that order, the header is
/// `serde_json::to_string` of the official `Metadata` padded with
/// spaces to a multiple of 8, preceded by its little-endian length,
/// and tensor bytes follow in the same order. `serialize` itself
/// cannot be used here because it returns the whole file as one
/// `Vec`, and `serialize_to_file` cannot be used because it
/// persists through its own sibling tempfile onto the destination
/// path (clobbering it), bypassing this crate's no-clobber
/// `TempFile` publish. Byte-identity with `serialize` is asserted
/// by test (streaming writer vs `safetensors::serialize`) and by
/// the fixture hashes; see the R2 pending validations.
/// Determinism: one `__metadata__` key (`format: pt`), matching
/// the previous writer and the example replacement file; edited-file metadata is not copied across, as before.
fn serialize_to_temp(
    req: &BuildReplacement,
    out: &OutputTarget,
    entries: &BTreeMap<String, EntryTensor<'_>>,
    cx: &mut Cx,
) -> Result<(TempFile, Sha256Hex, u64), ToolError> {
    // `TensorView`s over the entry bytes: borrowed full-form views
    // point straight into the edited snapshot mapping.
    let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(entries.len());
    for (name, tensor) in entries {
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.data.as_ref())
            .map_err(|e| ToolError::Safetensors {
                path: req.out.clone(),
                source: e,
            })?;
        views.push((name.clone(), view));
    }
    // The official writer's layout order: descending dtype
    // alignment (the `Dtype` order), then name.
    let mut ordered: Vec<&(String, TensorView<'_>)> = views.iter().collect();
    ordered.sort_by(|a, b| b.1.dtype().cmp(&a.1.dtype()).then_with(|| a.0.cmp(&b.0)));
    let mut offset = 0usize;
    let mut header_tensors: Vec<(String, TensorInfo)> = Vec::with_capacity(ordered.len());
    for (name, view) in &ordered {
        let n = view.data().len();
        header_tensors.push((
            (*name).clone(),
            TensorInfo {
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                data_offsets: (offset, offset + n),
            },
        ));
        offset += n;
    }
    let mut file_metadata: HashMap<String, String> = HashMap::new();
    file_metadata.insert("format".into(), "pt".into());
    let metadata =
        Metadata::new(Some(file_metadata), header_tensors).map_err(|e| ToolError::Safetensors {
            path: req.out.clone(),
            source: e,
        })?;
    let mut header_bytes = serde_json::to_string(&metadata)
        .map_err(|e| {
            ToolError::internal(format!("generated replacement header does not render: {e}"))
        })?
        .into_bytes();
    let aligned_len = header_bytes.len().next_multiple_of(8);
    header_bytes.resize(aligned_len, b' ');
    // The official writer refuses headers above its (private)
    // 100 MB cap; mirror that refusal before writing anything.
    if header_bytes.len() > 100_000_000 {
        return Err(ToolError::internal(
            "generated replacement header exceeds the safetensors header size cap",
        ));
    }
    let total = 8 + header_bytes.len() as u64 + offset as u64;

    let mut temp = TempFile::create_in(out.parent())?;
    let mut hasher = Sha256::new();
    let mut written = 0u64;
    cx.emit_bytes("replacement", Phase::Serialize, 0, total);
    // Header (length prefix + padded JSON) and then each tensor's
    // bytes, all through one chunked, hashing, cancelling writer.
    let prefix = (header_bytes.len() as u64).to_le_bytes();
    for part in [&prefix[..], &header_bytes[..]] {
        write_streaming(&mut temp, part, &mut hasher, &mut written, total, cx)?;
    }
    for (_, view) in &ordered {
        write_streaming(&mut temp, view.data(), &mut hasher, &mut written, total, cx)?;
    }
    temp.sync()?;
    cx.emit_bytes("replacement", Phase::Serialize, written, total);
    Ok((temp, Sha256Hex::from_digest(&hasher.finalize()), written))
}

/// Writes `bytes` to the temp file in bounded chunks, hashing
/// while writing and checking cancellation per chunk.
fn write_streaming(
    temp: &mut TempFile,
    bytes: &[u8],
    hasher: &mut Sha256,
    written: &mut u64,
    total: u64,
    cx: &mut Cx,
) -> Result<(), ToolError> {
    for chunk in bytes.chunks(crate::files::CHUNK_BYTES) {
        cx.check()?;
        temp.file_mut()
            .write_all(chunk)
            .map_err(|e| ToolError::io(temp.path(), e))?;
        hasher.update(chunk);
        *written += chunk.len() as u64;
        cx.emit_bytes("replacement", Phase::Serialize, *written, total);
    }
    Ok(())
}

/// Post-write gate and publication: the generated replacement must
/// pass the streaming gate (official-reader parse of the temp file
/// plus the `#idx` stream checks, no whole-file buffer) before
/// anything is published under the final name.
fn gate_temp_and_publish(temp: TempFile, out: &OutputTarget, cx: &mut Cx) -> Result<(), ToolError> {
    let temp_dir = cx.temp_dir().to_path_buf();
    match validate_replacement_streaming(temp.path(), &temp_dir, cx.cancelled) {
        Ok(()) => {}
        Err(ToolError::Cancelled) => return Err(ToolError::Cancelled),
        Err(e) => {
            return Err(ToolError::internal(format!(
                "generated replacement failed its streaming post-write gate: {e}"
            )));
        }
    }
    cx.check()?;
    cx.emit_bytes("replacement", Phase::Publish, 0, 0);
    temp.publish(out)
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
        return Err(ToolError::TensorSetMismatch {
            detail: format!(
                "tensor name sets differ (only in base {only_base:?}, only in edited {only_edited:?})"
            ),
        });
    }
    for (name, b) in &base.tensors {
        let e = edited.tensor(name)?;
        if b.shape != e.shape || b.dtype != e.dtype {
            return Err(ToolError::TensorSetMismatch {
                detail: format!(
                    "tensor {name}: shape/dtype differ between base ({:?} {:?}) \
                     and edited ({:?} {:?})",
                    b.shape, b.dtype, e.shape, e.dtype
                ),
            });
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
