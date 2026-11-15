//! Private snapshot custody for safetensors inputs, one design on
//! every platform: copy each input into a private file created by
//! std alone (see [`private_file`]), then map that same handle
//! read-only through the crate's single `unsafe` site and hash the
//! mapped bytes. The official `safetensors` reader only ever sees
//! those mapped snapshot bytes, never the caller's original file,
//! so mutating or replacing the original afterwards cannot change
//! what was parsed or what later ranged reads return.
//!
//! Custody, stated once: on Unix the private file is created
//! `O_CREAT|O_EXCL` with mode 0600 and unlinked immediately,
//! before any byte is written, and std opens it `O_CLOEXEC`; after
//! the unlink it has no name, and a handle can only come from
//! process inspection (`/proc/<pid>/fd`, ptrace, `task_for_pid`),
//! which takes the same UID or root. On Windows the file is
//! created with share mode 0, `FILE_ATTRIBUTE_TEMPORARY` and
//! `FILE_FLAG_DELETE_ON_CLOSE`, and is never reopened, so every
//! later open asking for read, write or delete access fails while
//! the mapping (which holds a duplicate of the handle) lives; std
//! creates the handle non-inheritable, and a handle can only be
//! duplicated out of this process, which takes the same user or
//! an administrator. The same user and root/administrators are
//! trusted on both families: they can write this process's memory
//! directly, so no file-level custody could exclude them. Other
//! users and the original input file are excluded. Nothing
//! stronger is claimed, and no atomicity is claimed for the
//! before/after source-size check that detects a source changing
//! during the copy.
//!
//! The digest is computed over the mapped bytes, not over the
//! writes: a mapped view is not guaranteed coherent with
//! `WriteFile` on Windows, and hashing what the parser sees makes
//! the digest and the parsed bytes the same object on every OS.
//! Snapshots are not synced: durability is pointless for a file
//! that cannot survive a crash, and no correctness claim depends
//! on it once the digest is over the mapping. Published outputs
//! keep their `sync_all` (see `crate::publish`).

mod private_file;

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use memmap2::Mmap;
use sha2::{Digest, Sha256};

use crate::error::ToolError;
use crate::files::{CHUNK_BYTES, Sha256Hex, check_cancelled, open_input};

/// A mapped private snapshot of one input file.
pub(crate) struct Snapshot {
    map: Mmap,
    sha256: Sha256Hex,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("len", &self.bytes().len())
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

impl Snapshot {
    /// Copies the regular file at `input` into a private file in
    /// `temp_dir`, maps it, and hashes the mapped bytes. Returns
    /// the resolved input path and the snapshot. A source whose
    /// size changes during the copy is refused: the snapshot must
    /// be a single consistent byte sequence. `temp_dir` needs room
    /// for a private copy of the input (on Linux the default
    /// system temp directory is often tmpfs, i.e. RAM).
    pub(crate) fn create(
        input: &Path,
        temp_dir: &Path,
        cancelled: &AtomicBool,
    ) -> Result<(PathBuf, Self), ToolError> {
        // Open first, then inspect the handle (see `open_input`):
        // no check-then-open race.
        let (resolved, source) = open_input(input)?;
        let expected = len_of(&source, input)?;
        let snapshot = Self::from_source(input, source, expected, temp_dir, cancelled)?;
        Ok((resolved, snapshot))
    }

    /// The copy-and-compare step over an already-open source whose
    /// recorded length is `expected`. Split from [`Self::create`]
    /// so a test can pass a mismatched length directly instead of
    /// racing a concurrent writer.
    fn from_source(
        input: &Path,
        mut source: File,
        expected: u64,
        temp_dir: &Path,
        cancelled: &AtomicBool,
    ) -> Result<Self, ToolError> {
        let mut private =
            private_file::create_in(temp_dir).map_err(|e| ToolError::io(temp_dir, e))?;
        let copied = copy_chunked(&mut source, &mut private, input, temp_dir, cancelled)?;
        if copied != expected || len_of(&source, input)? != expected {
            return Err(ToolError::InputChanged {
                path: input.to_owned(),
            });
        }
        drop(source);

        let map = seal(private).map_err(|e| ToolError::io(temp_dir, e))?;
        if map.len() as u64 != copied {
            return Err(ToolError::InputChanged {
                path: input.to_owned(),
            });
        }
        let sha256 = sha256_chunked(&map, cancelled)?;
        Ok(Self { map, sha256 })
    }

    /// The snapshot bytes: exactly the mapped bytes the digest was
    /// computed over.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.map
    }

    /// SHA-256 of the mapped snapshot bytes.
    pub(crate) fn sha256(&self) -> &Sha256Hex {
        &self.sha256
    }
}

fn len_of(file: &File, input: &Path) -> Result<u64, ToolError> {
    file.metadata()
        .map(|m| m.len())
        .map_err(|e| ToolError::io(input, e))
}

/// Copies `source` into `private` in bounded chunks, checking
/// cancellation per chunk, and returns the byte count.
fn copy_chunked(
    source: &mut File,
    private: &mut File,
    input: &Path,
    temp_dir: &Path,
    cancelled: &AtomicBool,
) -> Result<u64, ToolError> {
    let mut copied = 0u64;
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        check_cancelled(cancelled)?;
        let n = source.read(&mut buf).map_err(|e| ToolError::io(input, e))?;
        if n == 0 {
            return Ok(copied);
        }
        private
            .write_all(&buf[..n])
            .map_err(|e| ToolError::io(temp_dir, e))?;
        copied += n as u64;
    }
}

/// SHA-256 over the mapped bytes, in bounded chunks, checking
/// cancellation per chunk.
fn sha256_chunked(bytes: &[u8], cancelled: &AtomicBool) -> Result<Sha256Hex, ToolError> {
    let mut hasher = Sha256::new();
    for chunk in bytes.chunks(CHUNK_BYTES) {
        check_cancelled(cancelled)?;
        hasher.update(chunk);
    }
    Ok(Sha256Hex::from_digest(&hasher.finalize()))
}

/// Maps the finished snapshot read-only. Taking `file` by value is
/// the seal: after this call no handle to the snapshot exists in
/// safe code, so nothing in this process can write, extend or
/// truncate it. This is the crate's single `unsafe` site.
#[allow(unsafe_code)]
fn seal(file: File) -> std::io::Result<Mmap> {
    // SAFETY: `Mmap::map` is unsafe because the mapped file must
    // not be modified or truncated while the map is alive. `file`
    // comes from `private_file::create_in` and is consumed here,
    // so this process cannot write to it again. Another process
    // would need its own handle to the file:
    // - Unix: the file was unlinked right after `O_CREAT|O_EXCL`
    //   with mode 0600 and before any byte was written, and std
    //   opens it `O_CLOEXEC`; a handle can only come from process
    //   inspection (`/proc/<pid>/fd`, ptrace, `task_for_pid`),
    //   which takes the same UID or root.
    // - Windows: the file was created with share mode 0 and is
    //   never reopened, so every open that asks for read, write or
    //   delete access fails with a sharing violation for as long
    //   as the map (which holds a duplicate of the handle) lives;
    //   std creates the handle non-inheritable. A handle can only
    //   be duplicated out of this process (`PROCESS_DUP_HANDLE`),
    //   which takes the same user or an administrator.
    // The same user and root/administrators are trusted: they can
    // write this process's memory directly. Nothing stronger is
    // claimed.
    unsafe { Mmap::map(&file) }
}

#[cfg(test)]
mod tests;
