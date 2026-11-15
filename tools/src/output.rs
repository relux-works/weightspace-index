use std::fs;
use std::path::{Path, PathBuf};

use crate::error::ToolError;

fn output_symlink_refused(at: PathBuf, output: &Path) -> ToolError {
    ToolError::SymlinkRefused {
        path: at,
        detail: format!(
            "output path {} must not pass through a symlink",
            output.display()
        ),
    }
}

/// Verifies that `next`, one tail component under the canonical output
/// parent, is a real directory. A symlink found here was planted where
/// a fresh directory was expected (or raced in after creation) and is
/// refused; descending through it would write outside the canonical
/// parent the caller resolved.
fn verify_real_dir(next: &Path, output: &Path) -> Result<(), ToolError> {
    let meta = fs::symlink_metadata(next).map_err(|e| ToolError::io(next, e))?;
    if meta.file_type().is_symlink() {
        return Err(output_symlink_refused(next.to_path_buf(), output));
    }
    if !meta.is_dir() {
        return Err(ToolError::refused(format!(
            "{}: output parent is not a directory",
            next.display()
        )));
    }
    Ok(())
}

/// Creates each component of the relative `tail` under the canonical
/// `root` one by one with `create_dir`, verifying via
/// `symlink_metadata` that what exists is a real directory before
/// descending. `create_dir` never follows a final-component symlink
/// (it fails with `AlreadyExists`), and the verification then refuses
/// that planted link instead of descending through it. A component a
/// concurrent creator already made as a real directory is descended
/// into. Returns the full canonical parent.
fn create_tail_dirs(root: &Path, tail: &Path, output: &Path) -> Result<PathBuf, ToolError> {
    let mut cur = root.to_path_buf();
    for component in tail.components() {
        let name = match component {
            std::path::Component::Normal(name) => name,
            std::path::Component::CurDir => continue,
            _ => {
                return Err(ToolError::InvalidArgument {
                    detail: format!(
                        "output path {}: parent components that do not yet exist \
                         must be plain directory names",
                        output.display()
                    ),
                });
            }
        };
        let next = cur.join(name);
        match fs::create_dir(&next) {
            Ok(()) => verify_real_dir(&next, output)?,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                verify_real_dir(&next, output)?;
            }
            Err(e) => return Err(ToolError::io(&next, e)),
        }
        cur = next;
    }
    Ok(cur)
}

/// Resolves an output path to the canonical location that must be
/// written, refusing anything already at the destination.
///
/// Symlinks in the already-existing ancestor chain above the output
/// are resolved, not refused: the nearest (deepest) existing ancestor
/// of the parent is canonicalized and treated as the real parent, so
/// outputs under system-symlinked locations (macOS `/tmp`, `/var`,
/// the default `$TMPDIR` under `/var/folders`, a Linux symlinked
/// `TMPDIR`) work. This Linux evidence is a proxy for macOS; no macOS
/// run is claimed here. The final output component is refused in
/// every existing form — a symlink there is `SymlinkRefused`, any
/// other existing entry is `OutputExists` — and is never written
/// through. Parent components below the canonicalized ancestor that
/// do not yet exist are created one by one by [`create_tail_dirs`],
/// refusing a planted symlink in that tail; empty parent directories
/// created here may remain if a later stage refuses, but no output
/// is published by this check.
///
/// The resolution is check-then-act: a same-UID actor who can write
/// in the parent could swap a component between this resolution and
/// publication. That race is outside the guarantee (the documented
/// model; file publication is still no-clobber, directory
/// publication keeps the `TempDir::publish` limit) and no atomicity
/// is claimed for the resolution itself. Callers must create
/// temporaries in, and publish to, the returned canonical path so no
/// write spells its destination through the unresolved ancestors.
pub(crate) fn check_output_path(path: &Path) -> Result<PathBuf, ToolError> {
    let Some(file_name) = path.file_name().map(|n| n.to_os_string()) else {
        return match fs::symlink_metadata(path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                Err(output_symlink_refused(path.to_path_buf(), path))
            }
            Ok(_) => Err(ToolError::OutputExists {
                path: path.to_path_buf(),
            }),
            Err(_) => Err(ToolError::InvalidArgument {
                detail: format!(
                    "output path {} has no final file or directory name",
                    path.display()
                ),
            }),
        };
    };
    // Fast final-component refusal under the caller's spelling (the
    // operating system follows ancestor symlinks to reach it and
    // `symlink_metadata` lstats only the final component). A missing
    // parent reports an error here and is diagnosed below instead.
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            return Err(output_symlink_refused(path.to_path_buf(), path));
        }
        return Err(ToolError::OutputExists {
            path: path.to_path_buf(),
        });
    }
    let parent = parent_dir(path);
    let mut existing: Option<(PathBuf, fs::Metadata)> = None;
    for ancestor in parent.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(meta) => {
                existing = Some((ancestor.to_path_buf(), meta));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(ToolError::io(ancestor, e)),
        }
    }
    let Some((ancestor, ancestor_meta)) = existing else {
        return Err(ToolError::refused(format!(
            "{}: no existing ancestor of the output parent",
            parent.display()
        )));
    };
    let tail = parent
        .strip_prefix(&ancestor)
        .map_err(|_| ToolError::InvalidArgument {
            detail: format!(
                "output path {}: cannot split the parent at its existing ancestor",
                path.display()
            ),
        })?;
    let canonical_ancestor = match fs::canonicalize(&ancestor) {
        Ok(canonical) => canonical,
        Err(_) if ancestor_meta.file_type().is_symlink() => {
            return Err(output_symlink_refused(ancestor, path));
        }
        Err(e) => return Err(ToolError::io(&ancestor, e)),
    };
    let meta = fs::metadata(&canonical_ancestor).map_err(|e| ToolError::io(&parent, e))?;
    if !meta.is_dir() {
        return Err(ToolError::refused(format!(
            "{}: output parent is not a directory",
            parent.display()
        )));
    }
    let canonical_parent = create_tail_dirs(&canonical_ancestor, tail, path)?;
    let canonical_out = canonical_parent.join(&file_name);
    match fs::symlink_metadata(&canonical_out) {
        Ok(meta) if meta.file_type().is_symlink() => {
            Err(output_symlink_refused(canonical_out, path))
        }
        Ok(_) => Err(ToolError::OutputExists {
            path: canonical_out,
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(canonical_out),
        Err(e) => Err(ToolError::io(&canonical_out, e)),
    }
}

pub(crate) fn parent_dir(path: &Path) -> PathBuf {
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn output_under_existing_symlinked_ancestor_resolves() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let real = root.join("real");
        fs::create_dir(&real).unwrap();
        let link = root.join("link");
        symlink(&real, &link).unwrap();
        // The symlinked ancestor already exists above the output (the
        // macOS `/var` / symlinked-TMPDIR shape): it resolves to the
        // canonical parent instead of being refused.
        let resolved = check_output_path(&link.join("out.bin")).unwrap();
        assert_eq!(resolved, real.join("out.bin"));
        assert!(!resolved.exists());
    }

    #[cfg(unix)]
    #[test]
    fn output_missing_tail_created_under_canonical_ancestor() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let real = root.join("real");
        fs::create_dir(&real).unwrap();
        let link = root.join("link");
        symlink(&real, &link).unwrap();
        let resolved = check_output_path(&link.join("a").join("b").join("out.bin")).unwrap();
        assert_eq!(resolved, real.join("a").join("b").join("out.bin"));
        for dir in [real.join("a"), real.join("a").join("b")] {
            let meta = fs::symlink_metadata(&dir).unwrap();
            assert!(meta.is_dir() && !meta.file_type().is_symlink(), "{dir:?}");
        }
        assert!(!resolved.exists());
    }

    #[cfg(unix)]
    #[test]
    fn output_final_symlink_and_existing_refused() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let target = root.join("target.bin");
        fs::write(&target, b"x").unwrap();
        let out_link = root.join("out-link.bin");
        symlink(&target, &out_link).unwrap();
        let err = check_output_path(&out_link).unwrap_err();
        assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
        let err = check_output_path(&target).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
        let existing_dir = root.join("existing-dir");
        fs::create_dir(&existing_dir).unwrap();
        let err = check_output_path(&existing_dir).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn planted_tail_symlink_refused() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let elsewhere = root.join("elsewhere");
        fs::create_dir(&elsewhere).unwrap();
        // A symlink planted where the tail expects to create a fresh
        // directory below the canonical root is refused, and nothing
        // is created through it.
        symlink(&elsewhere, root.join("planted")).unwrap();
        let output = root.join("planted").join("out.bin");
        let err = create_tail_dirs(&root, Path::new("planted"), &output).unwrap_err();
        assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
        assert!(!elsewhere.join("out.bin").exists());
        // End to end, a dangling intermediate symlink is likewise
        // refused rather than written through.
        symlink(root.join("missing-target"), root.join("dangling")).unwrap();
        let err = check_output_path(&root.join("dangling").join("out.bin")).unwrap_err();
        assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
    }
}
