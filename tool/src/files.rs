//! File custody: read-only inputs, streaming hashes, and publication
//! that refuses existing destinations (files by atomic no-clobber
//! hard link; directories by claim-then-rename on Unix and
//! check-then-rename on Windows — see `crate::publish::publish_dir`
//! for the concurrent-writer residual).
//!
//! Custody rules (keeper amendments): a read-only input symlink is
//! allowed only as the resolved regular file reached by canonicalization
//! (the ordinary Hugging Face snapshot layout); an input directory
//! whose final path component is a symlink and special files are
//! refused, under any spelling of that component (trailing `/`, `/.`,
//! nested forms — see [`check_input_dir`]), while symlinked ancestors
//! above the input directory are followed by the operating system as
//! part of reaching it and are not themselves refused. Outputs are
//! checked once into an `OutputTarget` (see `crate::output`): the
//! output parent as spelled must already be a real directory, never
//! a symlink; links above the parent (for example macOS `/tmp`,
//! `/var`, or a `$TMPDIR` under `/var/folders`) are resolved by one
//! canonicalization, and the canonical parent plus name is what
//! every later write uses; the final output component must not
//! already exist in any form and is never written through.
//! Check-then-act races against a same-UID actor who can write in
//! the parent remain outside the guarantee; no atomicity is claimed
//! for the path resolution itself.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

use crate::error::ToolError;

pub use crate::digest::{ParseDigestError, Sha256Hex};

pub(crate) use crate::output::{OutputTarget, is_within};
pub(crate) use crate::publish::{TempDir, TempFile, sync_dir};
/// Bounded chunk size for every streaming read, copy, compare and hash.
pub(crate) const CHUNK_BYTES: usize = 64 * 1024;

pub(crate) fn check_cancelled(cancelled: &AtomicBool) -> Result<(), ToolError> {
    if cancelled.load(Ordering::Relaxed) {
        Err(ToolError::Cancelled)
    } else {
        Ok(())
    }
}

/// Opens an input file read-only. A symlink input is canonicalized
/// and the resolved file is opened first, then inspected through
/// the opened handle: there is no check-then-open race between a
/// metadata check and the open. The caller verifies the resolved
/// bytes (hashes, `#old` values) before trusting them.
pub(crate) fn open_input(path: &Path) -> Result<(PathBuf, File), ToolError> {
    let resolved = fs::canonicalize(path).map_err(|e| ToolError::io(path, e))?;
    let file = open_input_file(&resolved).map_err(|e| ToolError::io(path, e))?;
    let meta = file.metadata().map_err(|e| ToolError::io(path, e))?;
    if !meta.is_file() {
        return Err(ToolError::NotRegularFile {
            path: path.to_owned(),
        });
    }
    Ok((resolved, file))
}

/// Opens the resolved input. On Unix the open is non-blocking so a
/// FIFO (or other special file) is inspected and refused through
/// the handle instead of blocking in `open`; `O_NONBLOCK` has no
/// effect on reads from a regular file. The flag value is the
/// platform's own `O_NONBLOCK`, spelled here because std exposes
/// `custom_flags` but not the constant; other Unixes fall back to a
/// plain open, where a FIFO can still block (the residual the
/// cross-platform consultation records).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_input_file(resolved: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    const O_NONBLOCK: i32 = 0o4000;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK)
        .open(resolved)
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
fn open_input_file(resolved: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    const O_NONBLOCK: i32 = 0x0004;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK)
        .open(resolved)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
)))]
fn open_input_file(resolved: &Path) -> std::io::Result<File> {
    File::open(resolved)
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
///
/// On Windows this also refuses junctioned model directories: std
/// reports every name-surrogate reparse point (symlinks and
/// junctions) as a symlink. Non-surrogate reparse points, such as
/// OneDrive cloud placeholders, are treated as regular files and
/// are hydrated on read.
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
        return Err(ToolError::NotDirectory {
            path: path.to_owned(),
        });
    }
    Ok(())
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
// Preserved no-sink wrapper; current callers use the `_with_progress` form.
#[allow(dead_code)]
pub(crate) fn hash_range(
    file: &mut File,
    path: &Path,
    start: u64,
    len: u64,
    cancelled: &AtomicBool,
) -> Result<Sha256Hex, ToolError> {
    hash_range_with_progress(file, path, start, len, cancelled, &mut |_, _| {})
}

/// Like [`hash_range`], additionally reporting `(bytes_done, len)`
/// after every hashed chunk. Cancellation is still checked per chunk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hash_range_with_progress(
    file: &mut File,
    path: &Path,
    start: u64,
    len: u64,
    cancelled: &AtomicBool,
    progress: &mut (dyn FnMut(u64, u64) + '_),
) -> Result<Sha256Hex, ToolError> {
    file.seek(SeekFrom::Start(start))
        .map_err(|e| ToolError::io(path, e))?;
    let mut hasher = Sha256::new();
    let mut left = len;
    let mut done = 0u64;
    let mut buf = vec![0u8; CHUNK_BYTES];
    while left > 0 {
        check_cancelled(cancelled)?;
        let n = usize::try_from(left.min(CHUNK_BYTES as u64)).expect("chunk fits usize");
        file.read_exact(&mut buf[..n])
            .map_err(|e| ToolError::io(path, e))?;
        hasher.update(&buf[..n]);
        left -= n as u64;
        done += n as u64;
        progress(done, len);
    }
    Ok(Sha256Hex::from_digest(&hasher.finalize()))
}

/// SHA-256 of a whole resolved input file, streamed.
// Preserved no-sink wrapper; current callers use the `_with_progress` form.
#[allow(dead_code)]
pub(crate) fn hash_file(path: &Path, cancelled: &AtomicBool) -> Result<Sha256Hex, ToolError> {
    hash_file_with_progress(path, cancelled, &mut |_, _| {})
}

/// Like [`hash_file`], additionally reporting `(bytes_done, len)`
/// after every hashed chunk.
pub(crate) fn hash_file_with_progress(
    path: &Path,
    cancelled: &AtomicBool,
    progress: &mut (dyn FnMut(u64, u64) + '_),
) -> Result<Sha256Hex, ToolError> {
    let (resolved, mut file) = open_input(path)?;
    let len = file.metadata().map_err(|e| ToolError::io(path, e))?.len();
    hash_range_with_progress(&mut file, &resolved, 0, len, cancelled, progress)
}

/// Copies `len` bytes at `start` from `src` to `dst` at its current
/// position, in bounded chunks.
// Preserved no-sink wrapper; current callers use the `_with_progress` form.
#[allow(dead_code)]
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
    copy_range_with_progress(
        src,
        src_path,
        start,
        len,
        dst,
        dst_path,
        cancelled,
        &mut |_, _| {},
    )
}

/// Like [`copy_range`], additionally reporting `(bytes_done, len)`
/// after every copied chunk. Cancellation is still checked per chunk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_range_with_progress(
    src: &mut File,
    src_path: &Path,
    start: u64,
    len: u64,
    dst: &mut File,
    dst_path: &Path,
    cancelled: &AtomicBool,
    progress: &mut (dyn FnMut(u64, u64) + '_),
) -> Result<(), ToolError> {
    src.seek(SeekFrom::Start(start))
        .map_err(|e| ToolError::io(src_path, e))?;
    let mut left = len;
    let mut done = 0u64;
    let mut buf = vec![0u8; CHUNK_BYTES];
    while left > 0 {
        check_cancelled(cancelled)?;
        let n = usize::try_from(left.min(CHUNK_BYTES as u64)).expect("chunk fits usize");
        src.read_exact(&mut buf[..n])
            .map_err(|e| ToolError::io(src_path, e))?;
        dst.write_all(&buf[..n])
            .map_err(|e| ToolError::io(dst_path, e))?;
        left -= n as u64;
        done += n as u64;
        progress(done, len);
    }
    Ok(())
}
