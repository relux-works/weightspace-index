//! Private snapshot custody for safetensors inputs.
//!
//! Every checkpoint or patch file the official reader parses is first
//! copied, in bounded chunks, to a private temporary snapshot owned
//! by the operation. The copy is hashed while it is made, fully
//! written and synced, and then memory-mapped read-only; the
//! official `safetensors` reader only ever sees those mapped
//! snapshot bytes, never the caller's original file. Mutating or
//! replacing the original path afterwards therefore cannot change
//! what was parsed or what later ranged reads return.
//!
//! This module holds the crate's single `unsafe` call:
//! `MmapOptions::map` in [`map_readonly`]. On Unix the snapshot
//! pathname is unlinked immediately after creation, before any
//! bytes are copied, so the snapshot's lifetime is bound to its
//! already-open file rather than to its pathname: the copy, sync
//! and mapping all go through file handles to that one inode, the
//! pathname is never used again, and no other process can open
//! the snapshot by path at any point. A foreign file later
//! created at the former snapshot path — even one that reuses
//! the freed inode number, as ext4-family filesystems do on
//! unlink+recreate — is therefore neither read nor deleted by
//! this module, by construction and not by an identity check
//! that inode reuse could defeat. On other platforms, where an
//! open file cannot be unlinked this way, the previous pathname
//! flow is kept under the documented weaker temp-directory trust
//! (see the `SAFETY` argument in [`map_readonly`]).
//!
//! This custody does not discharge a same-UID actor that already
//! holds a writable descriptor to the snapshot inode — obtained
//! through the private pathname in the brief window before the
//! early unlink, or through another hard link taken in that
//! window — and modifies the inode in place through it; that
//! pre-opened-writer residual is explicitly out of scope (see
//! the `SAFETY` argument). No adversarial atomicity is claimed.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use crate::error::ToolError;
use crate::files::{CHUNK_BYTES, Sha256Hex, open_input};

/// A mapped private snapshot of one input file. The mapping is
/// dropped before the guard (field order), so the read-only file
/// handle the mapping was made from is closed only after the
/// mapping is gone, and on platforms where the snapshot name
/// outlives the mapping the file is unlinked only after it is
/// unmapped.
pub(crate) struct Snapshot {
    /// `None` only for an empty input, which has nothing to map;
    /// `bytes` then yields the empty slice the official reader
    /// refuses.
    mmap: Option<Mmap>,
    _guard: SnapshotGuard,
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
    /// Resolves `path` through the ordinary input custody
    /// ([`open_input`]), stream-copies the resolved regular file to
    /// a fresh no-clobber snapshot in the system temporary
    /// directory while hashing it, syncs the snapshot, and maps it
    /// read-only. On Unix the snapshot pathname is unlinked right
    /// after creation (see the module docs), so the copy, sync and
    /// mapping are bound to the already-open snapshot file, never
    /// to its pathname. Returns the resolved source path and the
    /// snapshot. A source whose size changes during the copy is
    /// refused: the snapshot must be a single consistent byte
    /// sequence.
    pub(crate) fn create(path: &Path) -> Result<(PathBuf, Self), ToolError> {
        Self::create_impl(path, None)
    }

    /// `after_unlink` is a test seam invoked with the snapshot's
    /// former pathname: on Unix immediately after the early
    /// unlink, before any bytes are copied; elsewhere after the
    /// copy is synced, just before the read-only reopen (the
    /// pathname is still live there). Production passes `None`.
    fn create_impl(
        path: &Path,
        after_unlink: Option<&dyn Fn(&Path)>,
    ) -> Result<(PathBuf, Self), ToolError> {
        let (resolved, mut source) = open_input(path)?;
        let expected_len = source.metadata().map_err(|e| ToolError::io(path, e))?.len();

        let (snap_path, mut writer) = create_snapshot_file(path)?;
        // Identity of the file this process created, taken from the
        // writable handle itself, never from the pathname. The
        // length is refreshed once the copy is complete below.
        let created = identity_of_file(&writer).map_err(|e| ToolError::io(path, e))?;
        let mut guard = SnapshotGuard {
            path: snap_path,
            named: true,
            identity: created,
            _readonly: None,
        };
        // Unix: unlink the pathname now, before copying any bytes,
        // binding the snapshot's lifetime to the already-open
        // `writer` file instead of the pathname. From here on the
        // pathname is never used again on Unix — not for the copy,
        // not for the read-only reopen, not for cleanup — so a
        // foreign file created at the former path (even one that
        // reuses the freed inode) can neither be read as the
        // snapshot nor be deleted by guard cleanup. If this unlink
        // itself fails the open refuses; the guard still holds the
        // live name then, so its drop performs the pre-unlink
        // identity-checked cleanup documented on `SnapshotGuard`.
        #[cfg(unix)]
        {
            fs::remove_file(guard.path()).map_err(|e| ToolError::io(path, e))?;
            guard.named = false;
            if let Some(seam) = after_unlink {
                seam(guard.path());
            }
        }
        let mut hasher = Sha256::new();
        let mut copied = 0u64;
        let mut buf = vec![0u8; CHUNK_BYTES];
        loop {
            let n = source.read(&mut buf).map_err(|e| ToolError::io(path, e))?;
            if n == 0 {
                break;
            }
            writer
                .write_all(&buf[..n])
                .map_err(|e| ToolError::io(guard.path(), e))?;
            hasher.update(&buf[..n]);
            copied += n as u64;
        }
        let final_len = source.metadata().map_err(|e| ToolError::io(path, e))?.len();
        drop(source);
        if copied != expected_len || copied != final_len {
            return Err(ToolError::refused(format!(
                "{}: input changed size while it was being snapshotted",
                path.display()
            )));
        }
        writer.flush().map_err(|e| ToolError::io(guard.path(), e))?;
        writer
            .sync_all()
            .map_err(|e| ToolError::io(guard.path(), e))?;
        // Final identity of the created file (device + inode on
        // Unix, plus the copied length), still from the writable
        // handle. The read-only handle mapped below must match
        // this record before anything is mapped.
        guard.identity = identity_of_file(&writer).map_err(|e| ToolError::io(path, e))?;
        let sha256 = Sha256Hex::from_digest(&hasher.finalize());

        // An empty file is refused by the official reader and cannot
        // be mapped (mmap needs a non-zero length); represent it as
        // an empty byte slice instead of touching the mapper.
        #[cfg(unix)]
        let mmap = {
            // Reopen the same inode read-only through the retained
            // writable file descriptor, never by pathname (the
            // pathname was unlinked before the copy). The writable
            // handle pins the inode, so this route cannot land on
            // a different file — including a foreign file that
            // reused the inode number at the former path, since a
            // pinned inode is never freed for reuse. The identity
            // check confirms the route yielded the recorded file
            // with the recorded length; on mismatch, or if the
            // fd-based route is unavailable, the open refuses and
            // nothing is mapped (there is no pathname fallback).
            let readonly = reopen_readonly_via_fd(&writer).map_err(|e| ToolError::io(path, e))?;
            let reopened = identity_of_file(&readonly).map_err(|e| ToolError::io(path, e))?;
            if reopened != guard.identity {
                return Err(ToolError::refused(format!(
                    "{}: snapshot file identity changed before it could be mapped; refusing",
                    path.display()
                )));
            }
            // No writable handle survives to map time: only the
            // read-only handle (kept in the guard, closed after
            // the mapping per `Snapshot` field order) and the
            // mapping itself remain.
            drop(writer);
            let mmap = if copied == 0 {
                None
            } else {
                let mmap = map_readonly(&readonly, path)?;
                if mmap.len() as u64 != copied {
                    return Err(ToolError::refused(format!(
                        "{}: snapshot mapping length differs from the copied file",
                        path.display()
                    )));
                }
                Some(mmap)
            };
            guard._readonly = Some(readonly);
            mmap
        };
        #[cfg(not(unix))]
        let mmap = {
            // Previous pathname flow, kept only where an open file
            // cannot be unlinked: reopen by pathname and bind the
            // reopen to the recorded identity before mapping, as
            // before. The recorded identity here is length only,
            // the documented weaker fallback (see `map_readonly`).
            drop(writer);
            if let Some(seam) = after_unlink {
                seam(guard.path());
            }
            if copied == 0 {
                None
            } else {
                let readonly = File::open(guard.path()).map_err(|e| ToolError::io(path, e))?;
                let reopened = identity_of_file(&readonly).map_err(|e| ToolError::io(path, e))?;
                if reopened != guard.identity {
                    return Err(ToolError::refused(format!(
                        "{}: snapshot file was substituted before it could be mapped; refusing",
                        path.display()
                    )));
                }
                let mmap = map_readonly(&readonly, path)?;
                drop(readonly);
                if mmap.len() as u64 != copied {
                    return Err(ToolError::refused(format!(
                        "{}: snapshot mapping length differs from the copied file",
                        path.display()
                    )));
                }
                Some(mmap)
            }
        };
        Ok((
            resolved,
            Self {
                mmap,
                _guard: guard,
                sha256,
            },
        ))
    }

    /// The snapshot bytes: the bytes hashed during the copy, in
    /// the created snapshot file (on Unix bound to its open file
    /// handles, its pathname unlinked before the copy; the
    /// same-UID pre-opened-writer residual in the module docs is
    /// out of scope).
    pub(crate) fn bytes(&self) -> &[u8] {
        self.mmap.as_ref().map_or(&[], |m| m)
    }

    /// SHA-256 of the snapshot bytes, computed while copying.
    pub(crate) fn sha256(&self) -> &Sha256Hex {
        &self.sha256
    }
}

/// Identity of the snapshot file this process created, recorded
/// from a handle to it. On Unix this is the device + inode
/// (`MetadataExt`) plus length; on other platforms only the length
/// is available from `std` metadata, which is the documented
/// weaker fallback there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    len: u64,
}

fn identity_of_metadata(meta: &fs::Metadata) -> FileIdentity {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        FileIdentity {
            dev: meta.dev(),
            ino: meta.ino(),
            len: meta.len(),
        }
    }
    #[cfg(not(unix))]
    {
        FileIdentity { len: meta.len() }
    }
}

fn identity_of_file(file: &File) -> std::io::Result<FileIdentity> {
    file.metadata().map(|m| identity_of_metadata(&m))
}

/// Whether the file at `meta` is the recorded created file, for
/// the pathname-cleanup decisions that remain (see
/// [`remove_if_still_ours`]). On Unix this compares device +
/// inode only: the created file legitimately grows during the
/// copy, so its length at an early error differs from the final
/// recorded length. (The reopen checks in `create_impl` are
/// stricter and also compare the final length.) On other
/// platforms the length is the only available identity.
///
/// Device + inode alone is *not* an identity proof across an
/// unlink: ext4-family filesystems reuse the freed inode for a
/// file recreated at the same path. That is exactly why the Unix
/// snapshot flow no longer decides anything by pathname once the
/// early unlink has happened; this comparison now serves only
/// the pre-unlink and non-Unix cleanup scoped in
/// [`remove_if_still_ours`].
fn is_recorded_file(recorded: &FileIdentity, meta: &fs::Metadata) -> bool {
    let found = identity_of_metadata(meta);
    #[cfg(unix)]
    {
        found.dev == recorded.dev && found.ino == recorded.ino
    }
    #[cfg(not(unix))]
    {
        found.len == recorded.len
    }
}

/// Unlinks `path` only while it still names the recorded created
/// file. Returns whether the name was removed.
///
/// Remaining uses, each still in the regime where this process's
/// snapshot file has not been unlinked yet: non-Unix guard drop
/// (the retained pathname flow there), and Unix guard drop when
/// the early unlink itself failed and the open is refusing. It
/// is never used after a successful Unix early unlink — the
/// guard holds no live name then — so on Unix it can no longer
/// be the mechanism that decides a foreign replacement's fate.
fn remove_if_still_ours(path: &Path, recorded: &FileIdentity) -> bool {
    match fs::symlink_metadata(path) {
        Ok(meta) if is_recorded_file(recorded, &meta) => fs::remove_file(path).is_ok(),
        _ => false,
    }
}

/// Reopens the inode behind `file` as a new read-only handle,
/// without using any pathname of the file itself (on Unix the
/// snapshot pathname was already unlinked before the copy). The
/// route goes through the process's own descriptor table
/// (`/proc/self/fd/N`, falling back to `/dev/fd/N`), so the
/// handle it yields refers to the inode `file` has open —
/// pinned, hence never freed for inode reuse — with read-only
/// access. Callers must still verify the reopened handle's
/// identity against the recorded one before mapping, and must
/// refuse if this route is unavailable rather than fall back to
/// a pathname.
#[cfg(unix)]
fn reopen_readonly_via_fd(file: &File) -> std::io::Result<File> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    match OpenOptions::new()
        .read(true)
        .open(format!("/proc/self/fd/{fd}"))
    {
        Ok(readonly) => Ok(readonly),
        Err(proc_err) => OpenOptions::new()
            .read(true)
            .open(format!("/dev/fd/{fd}"))
            .map_err(|_| proc_err),
    }
}

/// Snapshot custody after creation. `path` is the snapshot's
/// pathname: on Unix only its former name (kept for diagnostics;
/// unlinked before the copy and never used again), elsewhere
/// its live name until drop. `named` says whether that pathname
/// may still name our file and so needs cleanup on drop: it is
/// cleared by the successful Unix early unlink, after which
/// drop has no pathname to remove and can never delete a
/// foreign replacement; it stays set on other platforms, and
/// on Unix only when the early unlink itself failed (the
/// pre-unlink regime [`remove_if_still_ours`] still serves).
/// On Unix the guard also keeps the read-only handle the
/// mapping was made from, so `Snapshot` field order closes it
/// only after the mapping is dropped.
struct SnapshotGuard {
    path: PathBuf,
    named: bool,
    identity: FileIdentity,
    _readonly: Option<File>,
}

impl SnapshotGuard {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SnapshotGuard {
    fn drop(&mut self) {
        if self.named {
            let _ = remove_if_still_ours(&self.path, &self.identity);
        }
    }
}

static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Creates a fresh no-clobber snapshot file in the system temporary
/// directory, private to this user where the platform supports it
/// (mode `0600` on Unix, requested at creation).
fn create_snapshot_file(input: &Path) -> Result<(PathBuf, File), ToolError> {
    let dir = std::env::temp_dir();
    loop {
        let n = SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let candidate = dir.join(format!(
            ".weightspace-snapshot-{}-{n}-{nanos}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(ToolError::io(
                    input,
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "cannot create snapshot {} for {}: {e}",
                            candidate.display(),
                            input.display()
                        ),
                    ),
                ));
            }
        }
    }
}

/// The crate's single `unsafe` call: map a private snapshot
/// read-only.
///
/// SAFETY: `memmap2` marks mapping `unsafe` because modifying or
/// truncating the backing file while it is mapped — from any
/// process — is undefined behavior. That cannot happen to a
/// snapshot mapped here, by the following custody:
///
/// - The file was created `create_new` by this process in the
///   system temporary directory (`0600` on Unix), fully written
///   through its writable handle, flushed and `sync_all`-ed
///   above. This process never writes, extends or truncates it
///   again; the mapping is read-only.
/// - On Unix the snapshot pathname was unlinked immediately
///   after creation, before any bytes were copied, so no process
///   can open the inode by path during the copy or for the
///   mapping's lifetime; cleanup after that point never unlinks
///   by pathname, so an inode-reusing foreign replacement at
///   the former path is irrelevant to this mapping. The handle
///   mapped here is a new read-only handle to that same inode
///   obtained through the retained writable descriptor
///   (`reopen_readonly_via_fd`, never by pathname) and verified
///   against the identity recorded from the writable handle
///   (device + inode + final length, the length fixed from that
///   read-only handle before mapping). The writable handle was
///   closed before this call, leaving only the read-only handle
///   and the mapping alive; `Snapshot` field order drops
///   (unmaps) the mapping before that handle is closed. Out of
///   scope, explicitly: a same-UID actor that already holds a
///   writable descriptor to the inode — obtained through the
///   private pathname before the early unlink, or through
///   another hard link taken in that window — could still
///   modify the inode through it; no adversarial atomicity
///   against that pre-opened-writer residual is claimed. (The
///   *source* file changing during the copy is a separate
///   concern, handled by the before/after size checks in
///   `create_impl` and bounded further by the official parse
///   and `apply`'s hash gate, as before.)
/// - On other platforms the name stays in the per-user temporary
///   directory until drop and the read-only handle is reopened
///   by pathname after the same identity verification (length
///   only there), with the writable handle likewise closed
///   before mapping. The argument is then the ordinary
///   temp-file trust boundary: only a process running as the
///   same user could open and modify the snapshot underneath
///   the mapping, the same trust already placed in the
///   temporary output files this tool publishes from. This is a
///   weaker guarantee than the Unix early unlink and is
///   documented as such.
#[allow(unsafe_code)]
fn map_readonly(file: &File, input: &Path) -> Result<Mmap, ToolError> {
    // SAFETY: private immutable snapshot custody, argued above.
    let mmap = unsafe { MmapOptions::new().map(file) };
    mmap.map_err(|e| ToolError::io(input, e))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Snapshot-identity regression: a same-length crafted file
    /// placed at the former snapshot pathname — here at the
    /// deterministic `after_unlink` seam, so it is present for
    /// the whole copy, fd-reopen and mapping — is neither read
    /// nor deleted. The snapshot succeeds with the copied source
    /// bytes because custody is bound to the open snapshot file,
    /// not to the pathname an inode-reusing filesystem could
    /// recycle. (This supersedes the former
    /// `substituted_snapshot_path_is_refused_and_foreign_file_kept`
    /// pathname-substitution expectation with the stronger
    /// fd-bound one: the foreign bytes are still never parsed
    /// and the foreign file still survives.)
    #[test]
    fn foreign_file_at_former_snapshot_path_is_ignored_and_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.bin");
        let crafted = tmp.path().join("crafted.bin");
        fs::write(&source, vec![b'A'; 1024]).unwrap();
        fs::write(&crafted, vec![b'B'; 1024]).unwrap();

        let snap_path: RefCell<Option<PathBuf>> = RefCell::new(None);
        let seam = |path: &Path| {
            assert!(
                !path.exists(),
                "snapshot pathname must be unlinked before the copy"
            );
            *snap_path.borrow_mut() = Some(path.to_path_buf());
            // Same length as the source: nothing about this file
            // may influence the snapshot, whatever inode the
            // filesystem assigns it.
            if fs::hard_link(&crafted, path).is_err() {
                fs::write(path, vec![b'B'; 1024]).unwrap();
            }
        };
        let (_resolved, snapshot) = Snapshot::create_impl(&source, Some(&seam))
            .expect("fd-bound snapshot must ignore the former pathname");
        assert_eq!(snapshot.bytes(), vec![b'A'; 1024].as_slice());
        assert_eq!(
            snapshot.sha256(),
            &Sha256Hex::from_digest(&Sha256::digest(vec![b'A'; 1024]))
        );

        let former = snap_path.borrow().clone().expect("seam ran");
        // The foreign file was neither read as the snapshot nor
        // deleted while the snapshot is alive or on drop.
        assert_eq!(fs::read(&former).unwrap(), vec![b'B'; 1024]);
        drop(snapshot);
        assert_eq!(fs::read(&former).unwrap(), vec![b'B'; 1024]);
        fs::remove_file(&former).unwrap();
        assert_eq!(fs::read(&crafted).unwrap(), vec![b'B'; 1024]);
        // The caller's source is untouched.
        assert_eq!(fs::read(&source).unwrap(), vec![b'A'; 1024]);
    }

    /// The blocked-validation failure on ext4, in its exact
    /// sequence shape: the former snapshot pathname is recreated
    /// with foreign bytes after the snapshot file was unlinked,
    /// which on ext4-family filesystems reuses the freed inode
    /// (so a device+inode cleanup check cannot tell it from
    /// ours). Guard drop is now safe by construction — after the
    /// early unlink it has no pathname to remove — so the
    /// foreign file survives snapshot drop and the snapshot
    /// bytes never change. The `named` cleanup that remains for
    /// the pre-unlink/non-Unix regime is still pinned by the
    /// unsubstituted control at the end.
    #[test]
    fn guard_drop_never_deletes_a_foreign_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.bin");
        fs::write(&source, b"ours").unwrap();

        let snap_path: RefCell<Option<PathBuf>> = RefCell::new(None);
        let seam = |path: &Path| {
            *snap_path.borrow_mut() = Some(path.to_path_buf());
        };
        let (_resolved, snapshot) = Snapshot::create_impl(&source, Some(&seam)).unwrap();
        let former = snap_path.borrow().clone().expect("seam ran");
        assert!(!former.exists(), "snapshot pathname absent while alive");
        assert_eq!(snapshot.bytes(), b"ours");

        // Foreign replacement at the former path after open:
        // neither read through the snapshot nor deleted on drop.
        fs::write(&former, b"foreign!").unwrap();
        assert_eq!(snapshot.bytes(), b"ours");
        drop(snapshot);
        assert_eq!(fs::read(&former).unwrap(), b"foreign!");
        fs::remove_file(&former).unwrap();

        // Control: a still-named guard (pre-unlink/non-Unix
        // custody) still removes its own unsubstituted snapshot.
        let input = tmp.path().join("input.bin");
        fs::write(&input, b"x").unwrap();
        let (path, mut writer) = create_snapshot_file(&input).unwrap();
        writer.write_all(b"ours").unwrap();
        writer.sync_all().unwrap();
        let identity = identity_of_file(&writer).unwrap();
        drop(writer);
        let guard = SnapshotGuard {
            path: path.clone(),
            named: true,
            identity,
            _readonly: None,
        };
        drop(guard);
        assert!(!path.exists(), "own snapshot must be cleaned up");
    }

    /// On Unix the snapshot pathname does not exist at any point
    /// of the snapshot's observable lifetime, and dropping the
    /// snapshot leaves nothing behind at the former path.
    #[test]
    fn snapshot_pathname_absent_while_alive() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.bin");
        fs::write(&source, vec![b'A'; 64]).unwrap();

        let (_resolved, snapshot) = Snapshot::create(&source).unwrap();
        let former = snapshot._guard.path().to_path_buf();
        assert!(
            !former.exists(),
            "snapshot pathname must be absent while the snapshot is alive"
        );
        assert_eq!(snapshot.bytes(), vec![b'A'; 64].as_slice());
        drop(snapshot);
        assert!(!former.exists(), "no snapshot file must remain after drop");
    }

    /// A refused snapshot (source changed size during the copy)
    /// leaves no snapshot file: on Unix the copy went into the
    /// already-unlinked inode, which is freed when its handles
    /// close, with no pathname cleanup involved.
    #[test]
    fn failed_snapshot_leaves_no_snapshot_file() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.bin");
        fs::write(&source, b"ours").unwrap();

        let snap_path: RefCell<Option<PathBuf>> = RefCell::new(None);
        let grown = source.clone();
        let seam = |path: &Path| {
            *snap_path.borrow_mut() = Some(path.to_path_buf());
            // Grow the source before the copy starts: the copied
            // length can no longer match the recorded one.
            OpenOptions::new()
                .append(true)
                .open(&grown)
                .unwrap()
                .write_all(b"-more")
                .unwrap();
        };
        let err = Snapshot::create_impl(&source, Some(&seam))
            .expect_err("size-changed source must be refused");
        assert!(matches!(err, ToolError::Refused { .. }), "{err}");
        assert!(err.to_string().contains("changed size"), "{err}");

        let former = snap_path.borrow().clone().expect("seam ran");
        assert!(
            !former.exists(),
            "failed snapshot must leave no snapshot file"
        );
    }
}
