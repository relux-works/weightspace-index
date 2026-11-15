//! File custody: read-only inputs, streaming hashes, and publication
//! that refuses existing destinations (files by atomic no-clobber
//! hard link; directories by check-then-rename — see
//! `TempDir::publish` for the concurrent-writer limit).
//!
//! Custody rules (keeper amendments): a read-only input symlink is
//! allowed only as the resolved regular file reached by canonicalization
//! (the ordinary Hugging Face snapshot layout); an input directory
//! whose final path component is a symlink and special files are
//! refused, under any spelling of that component (trailing `/`, `/.`,
//! nested forms — see [`check_input_dir`]), while symlinked ancestors
//! above the input directory are followed by the operating system as
//! part of reaching it and are not themselves refused. Outputs:
//! symlinks in the already-existing ancestor chain above the output
//! (for example macOS `/tmp`, `/var`, or a `$TMPDIR` under
//! `/var/folders`) are resolved by canonicalizing the nearest
//! existing ancestor, and the canonical location is the real parent
//! every later write uses; the final output component must not
//! already exist in any form and is never written through; parent
//! components below the canonicalized ancestor that do not yet exist
//! are created one by one as real directories, refusing a planted
//! symlink in that tail (see [`check_output_path`]). Check-then-create
//! races against a same-UID actor who can write in the parent remain
//! outside the guarantee; no atomicity is claimed for the path
//! resolution itself.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::error::ToolError;

/// Bounded chunk size for every streaming read, copy, compare and hash.
pub(crate) const CHUNK_BYTES: usize = 64 * 1024;

/// A SHA-256 digest as 64 lowercase hex characters.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Hex(String);

impl Sha256Hex {
    /// Parses 64 ASCII hex characters, normalized to lowercase.
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        let ok = s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
        if ok {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(ToolError::InvalidArgument {
                detail: format!("expected 64 hex characters of SHA-256, got {s:?}"),
            })
        }
    }

    pub(crate) fn from_digest(bytes: &[u8]) -> Self {
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Sha256Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub(crate) fn check_cancelled(cancelled: &AtomicBool) -> Result<(), ToolError> {
    if cancelled.load(Ordering::Relaxed) {
        Err(ToolError::Cancelled)
    } else {
        Ok(())
    }
}

/// Opens an input file read-only. A symlink input is canonicalized and
/// the resolved path must be a regular file; the caller verifies the
/// resolved bytes (hashes, `#old` values) before trusting them.
pub(crate) fn open_input(path: &Path) -> Result<(PathBuf, File), ToolError> {
    let resolved = fs::canonicalize(path).map_err(|e| ToolError::io(path, e))?;
    let meta = fs::metadata(&resolved).map_err(|e| ToolError::io(path, e))?;
    if !meta.is_file() {
        return Err(ToolError::refused(format!(
            "{}: input is not a regular file",
            path.display()
        )));
    }
    let file = File::open(&resolved).map_err(|e| ToolError::io(path, e))?;
    Ok((resolved, file))
}

/// An input directory must be a real directory, never a symlink.
///
/// The check must not depend on how the path is spelled: on Linux,
/// `symlink_metadata` on `link/` or `link/.` follows the link and
/// reports a plain directory. Rebuilding the path from its components
/// drops trailing separators and `.` components
/// (`Path::components` normalization), so `link`, `link/`, `link/.` and nested
/// `<dir>/link` forms all `lstat` the final component itself and a
/// symlinked model directory is refused under every spelling. Only
/// the final component is checked: symlinked ancestors above the
/// model directory are followed while reaching it, as they are for
/// file inputs, and are not refused.
pub(crate) fn check_input_dir(path: &Path) -> Result<(), ToolError> {
    let stripped: PathBuf = path.components().collect();
    let meta = fs::symlink_metadata(&stripped).map_err(|e| ToolError::io(path, e))?;
    if meta.file_type().is_symlink() {
        return Err(ToolError::SymlinkRefused {
            path: stripped,
            detail: "input directory symlinks are refused".into(),
        });
    }
    if !meta.is_dir() {
        return Err(ToolError::refused(format!(
            "{}: not a directory",
            path.display()
        )));
    }
    Ok(())
}

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

/// Writes exactly `buf` at absolute file offset `start`.
pub(crate) fn write_at(
    file: &mut File,
    path: &Path,
    start: u64,
    buf: &[u8],
) -> Result<(), ToolError> {
    file.seek(SeekFrom::Start(start))
        .map_err(|e| ToolError::io(path, e))?;
    file.write_all(buf).map_err(|e| ToolError::io(path, e))
}

/// Streams `len` bytes at `start` through SHA-256 in bounded chunks.
pub(crate) fn hash_range(
    file: &mut File,
    path: &Path,
    start: u64,
    len: u64,
    cancelled: &AtomicBool,
) -> Result<Sha256Hex, ToolError> {
    file.seek(SeekFrom::Start(start))
        .map_err(|e| ToolError::io(path, e))?;
    let mut hasher = Sha256::new();
    let mut left = len;
    let mut buf = vec![0u8; CHUNK_BYTES];
    while left > 0 {
        check_cancelled(cancelled)?;
        let n = usize::try_from(left.min(CHUNK_BYTES as u64)).expect("chunk fits usize");
        file.read_exact(&mut buf[..n])
            .map_err(|e| ToolError::io(path, e))?;
        hasher.update(&buf[..n]);
        left -= n as u64;
    }
    Ok(Sha256Hex::from_digest(&hasher.finalize()))
}

/// SHA-256 of a whole resolved input file, streamed.
pub(crate) fn hash_file(path: &Path, cancelled: &AtomicBool) -> Result<Sha256Hex, ToolError> {
    let (resolved, mut file) = open_input(path)?;
    let len = file.metadata().map_err(|e| ToolError::io(path, e))?.len();
    hash_range(&mut file, &resolved, 0, len, cancelled)
}

/// Copies `len` bytes at `start` from `src` to `dst` at its current
/// position, in bounded chunks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_range(
    src: &mut File,
    src_path: &Path,
    start: u64,
    len: u64,
    dst: &mut File,
    dst_path: &Path,
    cancelled: &AtomicBool,
) -> Result<(), ToolError> {
    src.seek(SeekFrom::Start(start))
        .map_err(|e| ToolError::io(src_path, e))?;
    let mut left = len;
    let mut buf = vec![0u8; CHUNK_BYTES];
    while left > 0 {
        check_cancelled(cancelled)?;
        let n = usize::try_from(left.min(CHUNK_BYTES as u64)).expect("chunk fits usize");
        src.read_exact(&mut buf[..n])
            .map_err(|e| ToolError::io(src_path, e))?;
        dst.write_all(&buf[..n])
            .map_err(|e| ToolError::io(dst_path, e))?;
        left -= n as u64;
    }
    Ok(())
}

pub(crate) fn fsync_dir(dir: &Path) -> Result<(), ToolError> {
    File::open(dir)
        .and_then(|f| f.sync_all())
        .map_err(|e| ToolError::io(dir, e))
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_candidate(dir: &Path, tag: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    dir.join(format!(
        ".weightspace-{tag}-{}-{n}-{nanos}.tmp",
        std::process::id()
    ))
}

/// A same-directory temporary file that is removed unless published.
pub(crate) struct TempFile {
    path: PathBuf,
    file: Option<File>,
    published: bool,
}

impl TempFile {
    pub(crate) fn create_in(dir: &Path) -> Result<Self, ToolError> {
        loop {
            let path = temp_candidate(dir, "file");
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                        published: false,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(ToolError::io(&path, e)),
            }
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn file_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("temp file open until publish")
    }

    /// Flushes and fsyncs the temporary file.
    pub(crate) fn sync(&mut self) -> Result<(), ToolError> {
        let path = self.path.clone();
        let file = self.file_mut();
        file.flush().map_err(|e| ToolError::io(&path, e))?;
        file.sync_all().map_err(|e| ToolError::io(&path, e))
    }

    /// Publishes atomically without clobbering: a hard link to the
    /// destination (which fails if it appeared), then the temp name is
    /// removed and the destination directory fsynced. Once the hard
    /// link has succeeded the destination is published; a later I/O
    /// error removing the temp name or fsyncing the directory is
    /// still returned as `Err`, but a complete visible output may
    /// remain at the destination in that case.
    pub(crate) fn publish(mut self, dest: &Path) -> Result<(), ToolError> {
        self.sync()?;
        drop(self.file.take());
        fs::hard_link(&self.path, dest).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                ToolError::OutputExists {
                    path: dest.to_path_buf(),
                }
            } else {
                ToolError::io(dest, e)
            }
        })?;
        self.published = true;
        fs::remove_file(&self.path).map_err(|e| ToolError::io(&self.path, e))?;
        fsync_dir(&parent_dir(dest))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.published {
            drop(self.file.take());
            drop(fs::remove_file(&self.path));
        }
    }
}

/// A sibling temporary directory that is removed unless published.
pub(crate) struct TempDir {
    path: PathBuf,
    published: bool,
}

impl TempDir {
    pub(crate) fn create_in(parent: &Path) -> Result<Self, ToolError> {
        loop {
            let path = temp_candidate(parent, "dir");
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        published: false,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(ToolError::io(&path, e)),
            }
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Renames the finished directory into place, after a local
    /// check that the destination does not exist.
    ///
    /// This is check-then-rename, not atomic no-clobber: `std` gives
    /// no no-replace directory rename, so a concurrent writer in the
    /// destination directory could create the destination between
    /// the check and the rename (a rename onto an existing empty
    /// directory then succeeds and replaces it). Absent a concurrent
    /// writer the check refuses an existing destination and the
    /// rename publishes the complete directory at once. Once the
    /// rename has succeeded the destination is published; a later
    /// `fsync` error is still returned as `Err`, but a complete
    /// visible output may remain at the destination in that case.
    pub(crate) fn publish(mut self, dest: &Path) -> Result<(), ToolError> {
        fsync_dir(&self.path)?;
        if fs::symlink_metadata(dest).is_ok() {
            return Err(ToolError::OutputExists {
                path: dest.to_path_buf(),
            });
        }
        fs::rename(&self.path, dest).map_err(|e| ToolError::io(dest, e))?;
        self.published = true;
        fsync_dir(&parent_dir(dest))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if !self.published {
            drop(fs::remove_dir_all(&self.path));
        }
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
