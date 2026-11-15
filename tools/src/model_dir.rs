use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::ToolError;
use crate::files::{copy_range_with_progress, open_input};
use crate::ops::{Cx, Phase};

/// Slice 1 supports exactly one weights file at the model root:
/// `model.safetensors`. An index file or several safetensors files are
/// a sharded layout and are refused, never guessed at.
pub(crate) fn single_weights_file(model: &Path) -> Result<PathBuf, ToolError> {
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

pub(crate) fn read_sorted_dir(dir: &Path) -> Result<Vec<fs::DirEntry>, ToolError> {
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
pub(crate) fn copy_model_dir(
    src: &Path,
    dst: &Path,
    cx: &mut Cx,
    copied_dirs: &mut Vec<PathBuf>,
    skip_target: Option<&Path>,
) -> Result<(), ToolError> {
    let mut stack: Vec<(PathBuf, PathBuf)> = vec![(src.to_path_buf(), dst.to_path_buf())];
    copied_dirs.push(dst.to_path_buf());
    while let Some((from, to)) = stack.pop() {
        for entry in read_sorted_dir(&from)? {
            cx.check()?;
            let entry_path = entry.path();
            let target = to.join(entry.file_name());
            if skip_target == Some(target.as_path()) {
                // Staged separately from the verified snapshot
                // mapping by the caller (see ops::apply::stage_copy).
                continue;
            }
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
                return Err(ToolError::NotRegularFile { path: entry_path });
            }
        }
    }
    Ok(())
}

pub(crate) fn copy_one_file(
    src_path: &Path,
    src: &mut File,
    dst: &Path,
    cx: &mut Cx,
) -> Result<(), ToolError> {
    let len = src
        .metadata()
        .map_err(|e| ToolError::io(src_path, e))?
        .len();
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dst)
        .map_err(|e| ToolError::io(dst, e))?;
    copy_range_with_progress(
        src,
        src_path,
        0,
        len,
        &mut out,
        dst,
        cx.cancelled,
        &mut |done, total| {
            cx.emit_bytes("apply", Phase::Copy, done, total);
        },
    )?;
    out.flush().map_err(|e| ToolError::io(dst, e))?;
    out.sync_all().map_err(|e| ToolError::io(dst, e))
}
