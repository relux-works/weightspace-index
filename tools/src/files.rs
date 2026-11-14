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
//! part of reaching it and are not themselves refused; every output
//! path and output component symlink is refused; nothing is ever
//! written through a symlink.

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

/// Refuses an existing output and any symlink at or above the output path
/// (every existing ancestor component is checked). The parent directory
/// must already exist.
pub(crate) fn check_output_path(path: &Path) -> Result<(), ToolError> {
    let mut components: Vec<PathBuf> = path.ancestors().map(Path::to_path_buf).collect();
    components.reverse();
    for prefix in components {
        match fs::symlink_metadata(&prefix) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(ToolError::SymlinkRefused {
                    path: prefix,
                    detail: format!(
                        "output path {} must not pass through a symlink",
                        path.display()
                    ),
                });
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(ToolError::io(&prefix, e)),
        }
    }
    if fs::symlink_metadata(path).is_ok() {
        return Err(ToolError::OutputExists {
            path: path.to_path_buf(),
        });
    }
    let parent = parent_dir(path);
    let meta = fs::metadata(&parent).map_err(|e| ToolError::io(&parent, e))?;
    if !meta.is_dir() {
        return Err(ToolError::refused(format!(
            "{}: output parent is not a directory",
            parent.display()
        )));
    }
    Ok(())
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
