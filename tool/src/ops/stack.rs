//! Multi-patch replacement stacking with conflict merge policies
//! (`error`, `add`, `average`, `weighted`, `last-wins`, `first-wins`)
//! plus `--in-place` verified backups and `restore`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::{
    OutputTarget, Sha256Hex, TempDir, check_input_dir, hash_file_with_progress, is_within,
    sync_dir, write_at,
};
use crate::model_dir::{copy_model_dir, single_weights_file};
use crate::patch::{PatchFile, PatchTensor, for_each_index_block, parse_patch_in_with_progress};

use super::{ApplyReplacement, Cx, Phase, Report, apply_replacement};

pub const STATE_FILE: &str = ".weightspace-state.json";
pub const BACKUP_DIR: &str = ".weightspace-backup";

/// Policy for coordinates modified by more than one replacement patch in a stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConflictPolicy {
    /// Refuse when any coordinate is modified by more than one patch.
    #[default]
    Error,
    /// Additive delta merge: `W0 + sum(W_k - W0)`, identical to LoRA merge.
    Add,
    /// Mean of overlapping patched values: `sum(W_k) / m = W0 + sum(delta_k) / m`.
    Average,
    /// Weighted delta merge: `W0 + sum(alpha_k * (W_k - W0))`.
    Weighted,
    /// Bit-exact value from the last patch in stack order.
    LastWins,
    /// Bit-exact value from the first patch in stack order.
    FirstWins,
}

impl ConflictPolicy {
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        match s {
            "error" => Ok(Self::Error),
            "add" => Ok(Self::Add),
            "average" | "avg" => Ok(Self::Average),
            "weighted" => Ok(Self::Weighted),
            "last-wins" | "last" => Ok(Self::LastWins),
            "first-wins" | "first" => Ok(Self::FirstWins),
            _ => Err(ToolError::InvalidArgument {
                detail: format!(
                    "unknown --on-conflict policy {s:?}; expected error, add, average, weighted, last-wins, or first-wins"
                ),
            }),
        }
    }
}

/// One patch input in a stack, with an optional merge weight (default `1.0`).
#[derive(Clone, Debug)]
pub struct PatchSpec {
    pub path: PathBuf,
    pub weight: f64,
}

impl PatchSpec {
    pub fn parse(raw: &str) -> Result<Self, ToolError> {
        // Accept `PATH:WEIGHT` only when suffix after the last `:` parses as a finite f64
        // and the full string is not an existing path (preserves Windows drive letters like `C:\...`).
        let full_path = PathBuf::from(raw);
        if full_path.exists() {
            return Ok(Self {
                path: full_path,
                weight: 1.0,
            });
        }
        if let Some((head, tail)) = raw.rsplit_once(':')
            && !head.is_empty()
            && head.len() > 1
            && let Ok(weight) = tail.parse::<f64>()
        {
            if !weight.is_finite() {
                return Err(ToolError::InvalidArgument {
                    detail: format!("patch weight must be finite, got {tail:?}"),
                });
            }
            return Ok(Self {
                path: PathBuf::from(head),
                weight,
            });
        }
        Ok(Self {
            path: full_path,
            weight: 1.0,
        })
    }
}

/// Request to apply one or more replacement patches (copy to `--out` or `--in-place` with backup).
pub struct ApplyStack {
    pub patches: Vec<PatchSpec>,
    pub model: PathBuf,
    pub out: Option<PathBuf>,
    pub in_place: bool,
    pub on_conflict: ConflictPolicy,
    pub expect_tensor_sha256: BTreeMap<String, Sha256Hex>,
}

/// Persisted state inside an in-place or stacked patched model directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchedState {
    pub contract_version: u32,
    pub weights_file: String,
    pub base_sha256: String,
    pub backup_rel_path: Option<String>,
    pub on_conflict: ConflictPolicy,
    pub applied_patches: Vec<AppliedPatchRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppliedPatchRecord {
    pub path: String,
    pub sha256: String,
    pub weight: f64,
}

/// Result of restoring an in-place patched model from its verified backup.
#[derive(Debug)]
pub struct RestoreReport {
    pub weights_path: PathBuf,
    pub restored_sha256: Sha256Hex,
    pub restored_bytes: u64,
}

/// Applies one or more replacement patches, either into a new `--out` directory
/// or `--in-place` with an automatic verified backup.
pub fn apply_stack(req: &ApplyStack, cx: &mut Cx) -> Result<Report, ToolError> {
    if req.patches.is_empty() {
        return Err(ToolError::InvalidArgument {
            detail: "at least one --patch is required".into(),
        });
    }
    if req.in_place && req.out.is_some() {
        return Err(ToolError::InvalidArgument {
            detail: "--in-place and --out cannot be used together".into(),
        });
    }
    if !req.in_place && req.out.is_none() {
        return Err(ToolError::InvalidArgument {
            detail: "either --out <DIR> or --in-place is required".into(),
        });
    }

    check_input_dir(&req.model)?;
    let existing_state = read_state(&req.model)?;

    if req.on_conflict != ConflictPolicy::Weighted
        && let Some(bad) = req
            .patches
            .iter()
            .find(|p| (p.weight - 1.0).abs() > f64::EPSILON)
    {
        return Err(ToolError::InvalidArgument {
            detail: format!(
                "patch weight {} on {} requires --on-conflict=weighted",
                bad.weight,
                bad.path.display()
            ),
        });
    }

    // Fast path: single non-in-place patch on an unpatched model delegates directly
    // to the audited single-patch apply.
    if !req.in_place
        && existing_state.is_none()
        && req.patches.len() == 1
        && (req.patches[0].weight - 1.0).abs() < f64::EPSILON
    {
        return apply_replacement(
            &ApplyReplacement {
                patch: req.patches[0].path.clone(),
                model: req.model.clone(),
                out: req.out.clone().expect("checked above"),
                expect_tensor_sha256: req.expect_tensor_sha256.clone(),
            },
            cx,
        );
    }

    let model_canonical = fs::canonicalize(&req.model).map_err(|e| ToolError::io(&req.model, e))?;
    let weights_rel = single_weights_file(&req.model)?;

    // Determine effective base weights path & complete patch list when stacking on an already-patched model.
    let (base_weights_path, effective_specs) = match &existing_state {
        Some(state) => {
            let backup_rel = state.backup_rel_path.as_deref().ok_or_else(|| {
                ToolError::Backup {
                    path: req.model.join(STATE_FILE),
                    detail: "model has a patch state without a local base backup; apply the full patch stack against the unpatched base directory".into(),
                }
            })?;
            let backup_path = req.model.join(backup_rel);
            if !backup_path.is_file() {
                return Err(ToolError::Backup {
                    path: backup_path,
                    detail: "recorded base backup file is missing".into(),
                });
            }
            let backup_sha = hash_file_with_progress(&backup_path, cx.cancelled, &mut |_, _| {})?;
            if backup_sha.as_str() != state.base_sha256 {
                return Err(ToolError::Backup {
                    path: backup_path,
                    detail: format!(
                        "backup digest mismatch: expected {}, got {}",
                        state.base_sha256, backup_sha
                    ),
                });
            }
            let mut combined = Vec::new();
            for prev in &state.applied_patches {
                let p = PathBuf::from(&prev.path);
                if !p.is_file() {
                    return Err(ToolError::Backup {
                        path: p,
                        detail: "previously applied patch file is no longer present; restore base first or pass all patches explicitly".into(),
                    });
                }
                combined.push(PatchSpec {
                    path: p,
                    weight: prev.weight,
                });
            }
            combined.extend(req.patches.iter().cloned());
            (backup_path, combined)
        }
        None => (req.model.join(&weights_rel), req.patches.clone()),
    };

    let temp_dir = cx.temp_dir().to_path_buf();
    let base = Checkpoint::open_in_with_progress(
        &base_weights_path,
        &temp_dir,
        cx.cancelled,
        &mut |done, total| cx.emit_bytes("apply", Phase::Snapshot, done, total),
    )?;

    // Parse and verify every patch in the stack against the clean unpatched base.
    let mut parsed_patches: Vec<(PatchSpec, PatchFile)> = Vec::with_capacity(effective_specs.len());
    let mut all_full_names: BTreeSet<String> = BTreeSet::new();
    for spec in &effective_specs {
        cx.check()?;
        let patch = parse_patch_in_with_progress(
            &spec.path,
            &temp_dir,
            cx.cancelled,
            &mut |done, total| cx.emit_bytes("apply", Phase::Snapshot, done, total),
        )?;
        for (name, tensor) in &patch.tensors {
            let base_meta = base.tensors.get(name).ok_or_else(|| ToolError::WrongBase {
                tensor: name.clone(),
                detail: "tensor not present in the base checkpoint".into(),
            })?;
            match tensor {
                PatchTensor::Full { meta } => {
                    all_full_names.insert(name.clone());
                    if meta.shape != base_meta.shape || meta.dtype != base_meta.dtype {
                        return Err(ToolError::WrongBase {
                            tensor: name.clone(),
                            detail: format!(
                                "patch shape/dtype {:?} {:?} differ from base {:?} {:?}",
                                meta.shape, meta.dtype, base_meta.shape, base_meta.dtype
                            ),
                        });
                    }
                    let expected = req.expect_tensor_sha256.get(name).ok_or_else(|| {
                        ToolError::MissingExpectedDigest {
                            tensor: name.clone(),
                        }
                    })?;
                    let actual = base.tensor_hash_with_progress(
                        base_meta,
                        cx.cancelled,
                        &mut |done, total| cx.emit_bytes("apply", Phase::Verify, done, total),
                    )?;
                    if actual != *expected {
                        return Err(ToolError::WrongBase {
                            tensor: name.clone(),
                            detail: format!("expected base sha256 {expected}, computed {actual}"),
                        });
                    }
                }
                PatchTensor::Sparse { idx, new, old, .. } => {
                    if new.dtype != base_meta.dtype {
                        return Err(ToolError::WrongBase {
                            tensor: name.clone(),
                            detail: format!(
                                "sparse values dtype {:?} differ from base dtype {:?}",
                                new.dtype, base_meta.dtype
                            ),
                        });
                    }
                    verify_sparse_old(&patch.checkpoint, name, idx, old, &base, base_meta, cx)?;
                }
            }
        }
        parsed_patches.push((spec.clone(), patch));
    }

    for name in req.expect_tensor_sha256.keys() {
        if !all_full_names.contains(name) {
            return Err(ToolError::InvalidArgument {
                detail: format!(
                    "--expect-tensor-sha256 names {name}, which is not a full-form tensor in any patch"
                ),
            });
        }
    }

    // Collect every touched tensor across the stack.
    let mut touched_tensors: BTreeSet<String> = BTreeSet::new();
    for (_, p) in &parsed_patches {
        for name in p.tensors.keys() {
            touched_tensors.insert(name.clone());
        }
    }

    // Compute composed writes per tensor before touching destination files.
    let mut tensor_updates: BTreeMap<String, Vec<(u64, Vec<u8>)>> = BTreeMap::new();
    let mut total_entries = 0u64;
    for name in &touched_tensors {
        cx.check()?;
        let base_meta = base.tensor(name)?;
        let updates = compose_tensor(name, base_meta, &base, &parsed_patches, req.on_conflict, cx)?;
        total_entries += updates.len() as u64;
        tensor_updates.insert(name.clone(), updates);
    }

    let applied_records: Vec<AppliedPatchRecord> = parsed_patches
        .iter()
        .map(|(spec, p)| AppliedPatchRecord {
            path: fs::canonicalize(&spec.path)
                .unwrap_or_else(|_| spec.path.clone())
                .to_string_lossy()
                .into_owned(),
            sha256: p.checkpoint.snapshot_sha256().to_string(),
            weight: spec.weight,
        })
        .collect();

    if req.in_place {
        apply_in_place(
            req,
            &base,
            &weights_rel,
            &tensor_updates,
            &applied_records,
            existing_state.as_ref(),
            total_entries,
            cx,
        )
    } else {
        let out_path = req.out.as_ref().expect("checked above");
        let out = OutputTarget::check(out_path)?;
        if is_within(out.parent(), &model_canonical) {
            return Err(ToolError::OutputInsideInput {
                output: out_path.clone(),
                model: req.model.clone(),
            });
        }
        apply_to_out(
            req,
            &out,
            &base,
            &weights_rel,
            &tensor_updates,
            &applied_records,
            total_entries,
            cx,
        )
    }
}

/// Restores an `--in-place` patched model directory back to its exact original base weights.
pub fn restore_backup(model: &Path, cx: &mut Cx) -> Result<RestoreReport, ToolError> {
    check_input_dir(model)?;
    let state = read_state(model)?.ok_or_else(|| ToolError::Backup {
        path: model.join(STATE_FILE),
        detail: "no .weightspace-state.json found in this model directory".into(),
    })?;
    let backup_rel = state.backup_rel_path.ok_or_else(|| ToolError::Backup {
        path: model.join(STATE_FILE),
        detail: "state does not reference a local backup".into(),
    })?;
    let backup_path = model.join(&backup_rel);
    if !backup_path.is_file() {
        return Err(ToolError::Backup {
            path: backup_path,
            detail: "backup weights file is missing".into(),
        });
    }
    let backup_sha = hash_file_with_progress(&backup_path, cx.cancelled, &mut |done, total| {
        cx.emit_bytes("restore", Phase::Verify, done, total);
    })?;
    if backup_sha.as_str() != state.base_sha256 {
        return Err(ToolError::Backup {
            path: backup_path,
            detail: format!(
                "backup integrity check failed: expected {}, got {}",
                state.base_sha256, backup_sha
            ),
        });
    }

    let weights_path = model.join(&state.weights_file);
    let staging = model.join(".weightspace-restore.tmp");
    if staging.exists() {
        fs::remove_file(&staging).map_err(|e| ToolError::io(&staging, e))?;
    }
    fs::copy(&backup_path, &staging).map_err(|e| ToolError::io(&staging, e))?;
    let staged_bytes = fs::metadata(&staging)
        .map_err(|e| ToolError::io(&staging, e))?
        .len();
    let staged_sha = hash_file_with_progress(&staging, cx.cancelled, &mut |done, total| {
        cx.emit_bytes("restore", Phase::Hash, done, total);
    })?;
    if staged_sha != backup_sha {
        let _ = fs::remove_file(&staging);
        return Err(ToolError::internal(
            "staged restore copy digest differed from verified backup",
        ));
    }

    replace_file(&staging, &weights_path)?;
    sync_dir(model)?;

    let state_path = model.join(STATE_FILE);
    if state_path.exists() {
        fs::remove_file(&state_path).map_err(|e| ToolError::io(&state_path, e))?;
    }
    let backup_root = model.join(BACKUP_DIR);
    if backup_root.exists() {
        fs::remove_dir_all(&backup_root).map_err(|e| ToolError::io(&backup_root, e))?;
    }
    sync_dir(model)?;

    Ok(RestoreReport {
        weights_path,
        restored_sha256: staged_sha,
        restored_bytes: staged_bytes,
    })
}

pub fn read_state(model: &Path) -> Result<Option<PatchedState>, ToolError> {
    let path = model.join(STATE_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).map_err(|e| ToolError::io(&path, e))?;
    let state: PatchedState = serde_json::from_str(&raw).map_err(|e| ToolError::Backup {
        path,
        detail: format!("invalid state JSON: {e}"),
    })?;
    Ok(Some(state))
}

#[allow(clippy::too_many_arguments)]
fn apply_to_out(
    req: &ApplyStack,
    out: &OutputTarget,
    base: &Checkpoint,
    weights_rel: &Path,
    tensor_updates: &BTreeMap<String, Vec<(u64, Vec<u8>)>>,
    applied_records: &[AppliedPatchRecord],
    total_entries: u64,
    cx: &mut Cx,
) -> Result<Report, ToolError> {
    let temp = TempDir::create_in(out.parent())?;
    let target_path = temp.path().join(weights_rel);
    let mut copied_dirs = Vec::new();
    copy_model_dir(
        &req.model,
        temp.path(),
        cx,
        &mut copied_dirs,
        Some(&target_path),
    )?;
    // Never carry over an existing model's backup/state into a fresh copy.
    let copied_backup = temp.path().join(BACKUP_DIR);
    if copied_backup.exists() {
        fs::remove_dir_all(&copied_backup).map_err(|e| ToolError::io(&copied_backup, e))?;
        copied_dirs.retain(|d| !d.starts_with(&copied_backup));
    }
    write_snapshot_to_file(base, &target_path, cx)?;
    let out_bytes = write_composed_updates(base, &target_path, tensor_updates, cx)?;

    let state = PatchedState {
        contract_version: 1,
        weights_file: weights_rel.to_string_lossy().into_owned(),
        base_sha256: base.snapshot_sha256().to_string(),
        backup_rel_path: None,
        on_conflict: req.on_conflict,
        applied_patches: applied_records.to_vec(),
    };
    write_state_file(&temp.path().join(STATE_FILE), &state)?;

    for dir in copied_dirs.iter().rev() {
        sync_dir(dir)?;
    }
    let out_sha256 = hash_file_with_progress(&target_path, cx.cancelled, &mut |done, total| {
        cx.emit_bytes("apply", Phase::Hash, done, total);
    })?;
    cx.check()?;
    cx.emit_bytes("apply", Phase::Publish, 0, 0);
    temp.publish(out)?;

    Ok(Report {
        tensors_changed: tensor_updates.len() as u64,
        entries_changed: total_entries,
        out_sha256,
        out_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_in_place(
    req: &ApplyStack,
    base: &Checkpoint,
    weights_rel: &Path,
    tensor_updates: &BTreeMap<String, Vec<(u64, Vec<u8>)>>,
    applied_records: &[AppliedPatchRecord],
    existing_state: Option<&PatchedState>,
    total_entries: u64,
    cx: &mut Cx,
) -> Result<Report, ToolError> {
    let base_sha = base.snapshot_sha256().to_string();
    let backup_rel = match existing_state.and_then(|s| s.backup_rel_path.clone()) {
        Some(rel) => rel,
        None => {
            let rel = format!("{BACKUP_DIR}/{base_sha}/{}", weights_rel.to_string_lossy());
            let backup_path = req.model.join(&rel);
            if let Some(parent) = backup_path.parent() {
                fs::create_dir_all(parent).map_err(|e| ToolError::io(parent, e))?;
            }
            if !backup_path.exists() {
                write_snapshot_to_file(base, &backup_path, cx)?;
                let check_sha =
                    hash_file_with_progress(&backup_path, cx.cancelled, &mut |_, _| {})?;
                if check_sha.as_str() != base_sha {
                    let _ = fs::remove_file(&backup_path);
                    return Err(ToolError::internal(
                        "backup file hash differed from verified base snapshot",
                    ));
                }
            }
            rel
        }
    };

    let target_weights = req.model.join(weights_rel);
    let staged_weights = req.model.join(".weightspace-apply.tmp");
    if staged_weights.exists() {
        fs::remove_file(&staged_weights).map_err(|e| ToolError::io(&staged_weights, e))?;
    }
    write_snapshot_to_file(base, &staged_weights, cx)?;
    let out_bytes = write_composed_updates(base, &staged_weights, tensor_updates, cx)?;
    let out_sha256 = hash_file_with_progress(&staged_weights, cx.cancelled, &mut |done, total| {
        cx.emit_bytes("apply", Phase::Hash, done, total);
    })?;
    cx.check()?;

    replace_file(&staged_weights, &target_weights)?;
    let state = PatchedState {
        contract_version: 1,
        weights_file: weights_rel.to_string_lossy().into_owned(),
        base_sha256: base_sha,
        backup_rel_path: Some(backup_rel),
        on_conflict: req.on_conflict,
        applied_patches: applied_records.to_vec(),
    };
    write_state_file(&req.model.join(STATE_FILE), &state)?;
    sync_dir(&req.model)?;

    Ok(Report {
        tensors_changed: tensor_updates.len() as u64,
        entries_changed: total_entries,
        out_sha256,
        out_bytes,
    })
}

fn replace_file(staged: &Path, dest: &Path) -> Result<(), ToolError> {
    #[cfg(windows)]
    {
        if dest.exists() {
            fs::remove_file(dest).map_err(|e| ToolError::io(dest, e))?;
        }
    }
    fs::rename(staged, dest).map_err(|e| ToolError::io(dest, e))
}

fn write_state_file(path: &Path, state: &PatchedState) -> Result<(), ToolError> {
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| ToolError::internal(format!("serialize state: {e}")))?;
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| ToolError::io(path, e))?;
    f.write_all(json.as_bytes())
        .and_then(|_| f.write_all(b"\n"))
        .and_then(|_| f.sync_all())
        .map_err(|e| ToolError::io(path, e))
}

fn write_snapshot_to_file(
    base: &Checkpoint,
    target_path: &Path,
    cx: &mut Cx,
) -> Result<(), ToolError> {
    let snapshot = base.snapshot_bytes();
    let total = snapshot.len() as u64;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target_path)
        .map_err(|e| ToolError::io(target_path, e))?;
    let mut hasher = Sha256::new();
    let mut done = 0u64;
    cx.emit_bytes("apply", Phase::Copy, 0, total);
    for chunk in snapshot.chunks(crate::files::CHUNK_BYTES) {
        cx.check()?;
        file.write_all(chunk)
            .map_err(|e| ToolError::io(target_path, e))?;
        hasher.update(chunk);
        done += chunk.len() as u64;
        cx.emit_bytes("apply", Phase::Copy, done, total);
    }
    file.flush().map_err(|e| ToolError::io(target_path, e))?;
    file.sync_all().map_err(|e| ToolError::io(target_path, e))?;
    drop(file);
    if Sha256Hex::from_digest(&hasher.finalize()) != *base.snapshot_sha256() {
        return Err(ToolError::internal(format!(
            "{}: staged copy hash differs from verified snapshot",
            target_path.display()
        )));
    }
    Ok(())
}

fn write_composed_updates(
    base: &Checkpoint,
    target_path: &Path,
    tensor_updates: &BTreeMap<String, Vec<(u64, Vec<u8>)>>,
    cx: &mut Cx,
) -> Result<u64, ToolError> {
    let mut target = OpenOptions::new()
        .read(true)
        .write(true)
        .open(target_path)
        .map_err(|e| ToolError::io(target_path, e))?;
    for (name, updates) in tensor_updates {
        cx.check()?;
        let base_meta = base.tensor(name)?;
        let elem = base_meta.elem_bytes();
        let target_start = base.data_start() + base_meta.data_offsets.0;
        for (index, bytes) in updates {
            let at = target_start + (*index) * elem;
            write_at(&mut target, target_path, at, bytes)?;
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

fn compose_tensor(
    name: &str,
    base_meta: &TensorMeta,
    base: &Checkpoint,
    patches: &[(PatchSpec, PatchFile)],
    policy: ConflictPolicy,
    cx: &Cx,
) -> Result<Vec<(u64, Vec<u8>)>, ToolError> {
    let elem = base_meta.elem_bytes();
    let elem_usize = usize::try_from(elem).expect("element fits usize");
    let numel = base_meta.numel(&base.path, name)?;

    // Coordinate -> list of (patch_weight, new_bytes) in stack order.
    let mut per_coord: BTreeMap<u64, Vec<(f64, Vec<u8>)>> = BTreeMap::new();

    for (spec, pf) in patches {
        let Some(pt) = pf.tensors.get(name) else {
            continue;
        };
        match pt {
            PatchTensor::Sparse { idx, new, .. } => {
                let mut values = Vec::new();
                for_each_index_block(&pf.checkpoint, name, idx, cx.cancelled, |block, ordinal| {
                    if let Some(&bad) = block.iter().find(|&&i| i as u64 >= numel) {
                        return Err(ToolError::WrongBase {
                            tensor: name.to_owned(),
                            detail: format!("sparse index {bad} outside tensor ({numel})"),
                        });
                    }
                    let block_bytes = usize::try_from(block.len() as u64 * elem).expect("usize");
                    values.clear();
                    values.resize(block_bytes, 0);
                    pf.checkpoint
                        .read_tensor_at(new, ordinal * elem, &mut values)?;
                    for (i, &coord) in block.iter().enumerate() {
                        let slice = values[i * elem_usize..(i + 1) * elem_usize].to_vec();
                        per_coord
                            .entry(coord as u64)
                            .or_default()
                            .push((spec.weight, slice));
                    }
                    Ok(())
                })?;
            }
            PatchTensor::Full { meta } => {
                let full_bytes = pf.checkpoint.tensor_bytes(meta)?;
                let base_bytes = base.tensor_bytes(base_meta)?;
                for idx in 0..numel {
                    let off = usize::try_from(idx * elem).expect("usize");
                    let new_slice = &full_bytes[off..off + elem_usize];
                    let old_slice = &base_bytes[off..off + elem_usize];
                    if new_slice != old_slice {
                        per_coord
                            .entry(idx)
                            .or_default()
                            .push((spec.weight, new_slice.to_vec()));
                    }
                }
            }
        }
    }

    let mut overlapping_count = 0u64;
    let mut out = Vec::with_capacity(per_coord.len());
    let mut base_elem = vec![0u8; elem_usize];

    for (coord, entries) in per_coord {
        cx.check()?;
        if entries.len() == 1 {
            let (weight, bytes) = &entries[0];
            if policy == ConflictPolicy::Weighted && (*weight - 1.0).abs() > f64::EPSILON {
                base.read_tensor_at(base_meta, coord * elem, &mut base_elem)?;
                let w0 = decode_float(base_meta.dtype, &base_elem, &base.path, name)?;
                let w1 = decode_float(base_meta.dtype, bytes, &base.path, name)?;
                let merged = w0 + weight * (w1 - w0);
                out.push((
                    coord,
                    encode_float(base_meta.dtype, merged, &base.path, name)?,
                ));
            } else {
                out.push((coord, bytes.clone()));
            }
            continue;
        }

        overlapping_count += 1;
        match policy {
            ConflictPolicy::Error => continue,
            ConflictPolicy::LastWins => {
                out.push((coord, entries.last().unwrap().1.clone()));
            }
            ConflictPolicy::FirstWins => {
                out.push((coord, entries.first().unwrap().1.clone()));
            }
            ConflictPolicy::Add => {
                base.read_tensor_at(base_meta, coord * elem, &mut base_elem)?;
                let w0 = decode_float(base_meta.dtype, &base_elem, &base.path, name)?;
                let mut sum_delta = 0.0f64;
                for (_, bytes) in &entries {
                    let wk = decode_float(base_meta.dtype, bytes, &base.path, name)?;
                    sum_delta += wk - w0;
                }
                let merged = w0 + sum_delta;
                out.push((
                    coord,
                    encode_float(base_meta.dtype, merged, &base.path, name)?,
                ));
            }
            ConflictPolicy::Average => {
                base.read_tensor_at(base_meta, coord * elem, &mut base_elem)?;
                let mut sum = 0.0f64;
                for (_, bytes) in &entries {
                    sum += decode_float(base_meta.dtype, bytes, &base.path, name)?;
                }
                let merged = sum / (entries.len() as f64);
                out.push((
                    coord,
                    encode_float(base_meta.dtype, merged, &base.path, name)?,
                ));
            }
            ConflictPolicy::Weighted => {
                base.read_tensor_at(base_meta, coord * elem, &mut base_elem)?;
                let w0 = decode_float(base_meta.dtype, &base_elem, &base.path, name)?;
                let mut acc = w0;
                for (weight, bytes) in &entries {
                    let wk = decode_float(base_meta.dtype, bytes, &base.path, name)?;
                    acc += weight * (wk - w0);
                }
                out.push((coord, encode_float(base_meta.dtype, acc, &base.path, name)?));
            }
        }
    }

    if overlapping_count > 0 && policy == ConflictPolicy::Error {
        return Err(ToolError::PatchConflict {
            tensor: name.to_owned(),
            overlapping_entries: overlapping_count,
            detail: "re-run with --on-conflict=add|average|weighted|last-wins|first-wins".into(),
        });
    }

    Ok(out)
}

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
    let numel = base_meta.numel(&base.path, name)?;
    let mut base_elem = vec![0u8; elem_usize];
    let mut old_values = Vec::new();
    for_each_index_block(patch, name, idx, cx.cancelled, |block, ordinal| {
        if let Some(&bad) = block.iter().find(|&&i| i as u64 >= numel) {
            return Err(ToolError::WrongBase {
                tensor: name.to_owned(),
                detail: format!("sparse index {bad} is outside base tensor ({numel} elements)"),
            });
        }
        let block_bytes = usize::try_from(block.len() as u64 * elem).expect("block fits usize");
        old_values.clear();
        old_values.resize(block_bytes, 0);
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

fn decode_float(dtype: Dtype, bytes: &[u8], path: &Path, tensor: &str) -> Result<f64, ToolError> {
    match dtype {
        Dtype::BF16 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f32::from_bits(u32::from(bits) << 16) as f64)
        }
        Dtype::F16 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f16_to_f64(bits))
        }
        Dtype::F32 => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64),
        Dtype::F64 => Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])),
        other => Err(ToolError::UnsupportedDtype {
            path: path.to_path_buf(),
            tensor: tensor.to_owned(),
            dtype: other,
        }),
    }
}

fn encode_float(dtype: Dtype, value: f64, path: &Path, tensor: &str) -> Result<Vec<u8>, ToolError> {
    if !value.is_finite() {
        return Err(ToolError::InvalidArgument {
            detail: format!("tensor {tensor}: merged value overflowed ({value})"),
        });
    }
    match dtype {
        Dtype::BF16 => {
            let f = value as f32;
            if !f.is_finite() {
                return Err(ToolError::InvalidArgument {
                    detail: format!("tensor {tensor}: merged value overflowed BF16"),
                });
            }
            let bits = f.to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            let out = (rounded >> 16) as u16;
            Ok(out.to_le_bytes().to_vec())
        }
        Dtype::F16 => Ok(f64_to_f16(value).to_le_bytes().to_vec()),
        Dtype::F32 => Ok((value as f32).to_le_bytes().to_vec()),
        Dtype::F64 => Ok(value.to_le_bytes().to_vec()),
        other => Err(ToolError::UnsupportedDtype {
            path: path.to_path_buf(),
            tensor: tensor.to_owned(),
            dtype: other,
        }),
    }
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    match exp {
        0 => sign * (frac as f64) * 2.0f64.powi(-24),
        31 => {
            if frac == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + (frac as f64) / 1024.0) * 2.0f64.powi(exp - 15),
    }
}

fn f64_to_f16(value: f64) -> u16 {
    let f = value as f32;
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - half_exp);
        let round = (m >> 12) & 1;
        let shifted = (m >> 13) + round;
        return sign | (shifted as u16);
    }
    let round_bit = 0x1000;
    let mant_rounded = mant + round_bit - 1 + ((mant >> 13) & 1);
    if mant_rounded & 0x0080_0000 != 0 {
        let next_exp = half_exp + 1;
        if next_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((next_exp as u16) << 10);
    }
    sign | ((half_exp as u16) << 10) | ((mant_rounded >> 13) as u16)
}
