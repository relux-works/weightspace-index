//! Private snapshot custody for safetensors inputs.
//!
//! Every checkpoint or patch file the official reader parses is first
//! copied, in bounded chunks, to a private temporary snapshot owned
//! by the operation. The copy is hashed while it is made, fully
//! written and synced, sealed, and then memory-mapped read-only; the
//! official `safetensors` reader only ever sees those mapped
//! snapshot bytes, never the caller's original file. Mutating or
//! replacing the original path afterwards therefore does not change
//! what was parsed or what later ranged reads return.
//!
//! This module holds the crate's single `unsafe` call:
//! `MmapOptions::map` in [`map_readonly`]. On Unix the snapshot is
//! created read-write and its pathname is unlinked immediately
//! after creation, before any bytes are copied. The copy, hash and
//! sync go through the retained file; the file is then sealed (see
//! [`SealedSnapshot`]) and that same already-open file is mapped
//! read-only. No reopen by pathname and no reopen through a
//! descriptor-table path is used on any platform, so a foreign file
//! later created at the former snapshot path — even one that
//! reuses the freed inode number, as ext4-family filesystems do on
//! unlink+recreate — is neither read nor deleted by this module.
//! On non-Unix platforms, where this custody relies on
//! unlink-while-open, snapshot creation is refused instead of
//! falling back to a weaker flow.
//!
//! Trust boundary, stated exactly: the pathname is unlinked before
//! the copy; the bytes parsed are the copied snapshot bytes; the
//! snapshot length is fixed at seal and the mapping length is
//! checked against it. A same-UID actor who already holds a
//! descriptor to the snapshot inode, or who can open
//! `/proc/<pid>/fd` entries of this process, is outside the
//! guarantee this custody provides for the mapping lifetime.
//! Source tearing during the copy is detected only by the
//! before/after source size checks
//! in `Snapshot::create_impl`. No adversarial atomicity result is
//! claimed beyond that boundary.

use std::fs::File;
#[cfg(unix)]
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(unix)]
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::Mmap;
#[cfg(unix)]
use memmap2::MmapOptions;
#[cfg(unix)]
use sha2::{Digest, Sha256};

use crate::error::ToolError;
use crate::files::Sha256Hex;
#[cfg(unix)]
use crate::files::{CHUNK_BYTES, open_input};

/// A mapped private snapshot of one input file. The mapping is
/// dropped before the sealed file (field order), so the file the
/// mapping was made from is closed only after the mapping is gone.
pub(crate) struct Snapshot {
    /// `None` only for an empty input, which has nothing to map;
    /// `bytes` then yields the empty slice the official reader
    /// refuses.
    mmap: Option<Mmap>,
    sealed: SealedSnapshot,
    sha256: Sha256Hex,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("len", &self.bytes().len())
            .field("former_path", &self.sealed.former_path())
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

impl Snapshot {
    /// Resolves `path` through the ordinary input custody
    /// ([`open_input`]), stream-copies the resolved regular file to
    /// a fresh no-clobber snapshot in the system temporary
    /// directory while hashing it, syncs and seals the snapshot,
    /// and maps the same already-open file read-only. On Unix the
    /// snapshot pathname is unlinked right after creation (see the
    /// module docs), so the copy, sync, seal and mapping are bound
    /// to the already-open snapshot file, never to its pathname.
    /// Returns the resolved source path and the snapshot. A source
    /// whose size changes during the copy is refused: the snapshot
    /// must be a single consistent byte sequence.
    pub(crate) fn create(path: &Path) -> Result<(PathBuf, Self), ToolError> {
        Self::create_impl(path, None)
    }

    /// `after_unlink` is a test seam invoked with the snapshot's
    /// former pathname immediately after the early unlink, before
    /// any bytes are copied. Production passes `None`.
    #[cfg(unix)]
    fn create_impl(
        path: &Path,
        after_unlink: Option<&dyn Fn(&Path)>,
    ) -> Result<(PathBuf, Self), ToolError> {
        let (resolved, mut source) = open_input(path)?;
        let expected_len = source.metadata().map_err(|e| ToolError::io(path, e))?.len();

        let mut writable = WritableSnapshot::create(path)?;
        // Unlink the pathname now, before copying any bytes. From
        // here on the pathname is never used again — not for the
        // copy, not for the mapping, not for cleanup — so a foreign
        // file created at the former path (even one that reuses the
        // freed inode) is neither read as the snapshot nor deleted
        // by cleanup. If this step refuses, the still-named
        // writable value cleans up on drop with its file retained
        // through the identity check (see `WritableSnapshot`) or
        // preserves the foreign path.
        writable.unlink_before_copy(path)?;
        if let Some(seam) = after_unlink {
            seam(writable.former_path());
        }
        let mut hasher = Sha256::new();
        let mut copied = 0u64;
        let mut buf = vec![0u8; CHUNK_BYTES];
        loop {
            let n = source.read(&mut buf).map_err(|e| ToolError::io(path, e))?;
            if n == 0 {
                break;
            }
            writable
                .write_all(&buf[..n])
                .map_err(|e| ToolError::io(writable.former_path(), e))?;
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
        writable
            .flush_and_sync()
            .map_err(|e| ToolError::io(writable.former_path(), e))?;
        // The snapshot length, read from the retained file, fixes
        // what the seal and the mapping mean below.
        let snapshot_len = writable.file_len().map_err(|e| ToolError::io(path, e))?;
        if snapshot_len != copied {
            return Err(ToolError::refused(format!(
                "{}: snapshot length differs from the copied file; refusing",
                path.display()
            )));
        }
        let sha256 = Sha256Hex::from_digest(&hasher.finalize());

        // Seal: the writable value is consumed and only a sealed
        // value remains. The mapping is produced from that sealed
        // value over the same already-open file; there is no
        // reopen of any kind.
        let sealed = writable.seal(copied);
        // An empty file is refused by the official reader and
        // cannot be mapped (mapping needs a non-zero length);
        // represent it as an empty byte slice instead of touching
        // the mapper.
        let mmap = if copied == 0 {
            None
        } else {
            let mmap = sealed.map(path)?;
            if mmap.len() as u64 != sealed.len() {
                return Err(ToolError::refused(format!(
                    "{}: snapshot mapping length differs from the copied file",
                    path.display()
                )));
            }
            Some(mmap)
        };
        Ok((
            resolved,
            Self {
                mmap,
                sealed,
                sha256,
            },
        ))
    }

    #[cfg(not(unix))]
    fn create_impl(
        path: &Path,
        _after_unlink: Option<&dyn Fn(&Path)>,
    ) -> Result<(PathBuf, Self), ToolError> {
        Err(ToolError::refused(format!(
            "{}: private snapshot custody requires Unix unlink-while-open; \
             this platform is not supported",
            path.display()
        )))
    }

    /// The snapshot bytes: the bytes hashed during the copy, in
    /// the sealed snapshot file whose pathname was unlinked before
    /// the copy (the same-UID descriptor scope limit in the module
    /// docs is outside this custody).
    pub(crate) fn bytes(&self) -> &[u8] {
        self.mmap.as_ref().map_or(&[], |m| m)
    }

    /// SHA-256 of the snapshot bytes, computed while copying.
    pub(crate) fn sha256(&self) -> &Sha256Hex {
        &self.sha256
    }
}

/// A snapshot file after the seal point. The retained file is
/// private to this module and no writable capability is reachable
/// through this type: it hands out no `File`, no mutable reference
/// and no write or truncate operation. The read-only mapping is
/// the only view produced from it. `former_path` is kept only for
/// diagnostics; after the early unlink it names nothing of ours
/// and is never used for I/O or cleanup.
// On non-Unix this value is never constructed (creation refuses);
// the fields still define the sealed shape the Unix flow seals into.
#[cfg_attr(not(unix), allow(dead_code))]
struct SealedSnapshot {
    file: File,
    former_path: PathBuf,
    len: u64,
}

impl SealedSnapshot {
    fn former_path(&self) -> &Path {
        &self.former_path
    }

    #[cfg(unix)]
    fn len(&self) -> u64 {
        self.len
    }

    #[cfg(unix)]
    fn map(&self, input: &Path) -> Result<Mmap, ToolError> {
        map_readonly(&self.file, input)
    }
}

/// Identity (device + inode) of the snapshot file this process
/// created, recorded from its retained file. It serves only the
/// pre-unlink window: deciding whether the just-created pathname
/// still names our file before that pathname is unlinked or, if
/// the early unlink itself fails, cleaned up. After a successful
/// early unlink nothing is decided by pathname, so no reopen or
/// mapping decision uses this record — device + inode would in
/// any case not be an identity proof across an unlink, since a
/// freed inode number can be reused by a later file.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CreatedIdentity {
    dev: u64,
    ino: u64,
}

#[cfg(unix)]
fn identity_of_metadata(meta: &fs::Metadata) -> CreatedIdentity {
    use std::os::unix::fs::MetadataExt;
    CreatedIdentity {
        dev: meta.dev(),
        ino: meta.ino(),
    }
}

#[cfg(unix)]
fn identity_of_file(file: &File) -> std::io::Result<CreatedIdentity> {
    file.metadata().map(|m| identity_of_metadata(&m))
}

/// Unlinks `path` only while it still names the recorded created
/// file. Returns whether the name was removed. Callers retain the
/// created file across this check and unlink, so its inode stays
/// pinned and cannot have been freed for reuse by a foreign file
/// in between; a foreign path fails the check and is preserved.
/// Used only before a successful early unlink.
#[cfg(unix)]
fn remove_if_still_ours(path: &Path, recorded: &CreatedIdentity) -> bool {
    match fs::symlink_metadata(path) {
        Ok(meta) if identity_of_metadata(&meta) == *recorded => fs::remove_file(path).is_ok(),
        _ => false,
    }
}

#[cfg(unix)]
enum EarlyUnlink {
    Unlinked,
    AlreadyUnnamed,
    ForeignPath,
}

/// Identity-checked early unlink of the just-created pathname.
/// The retained file pins the inode through the check and the
/// unlink. The check and the unlink remain two acts (`std` offers
/// no single compare-and-unlink step), which is part of the
/// same-UID scope limit in the module docs.
#[cfg(unix)]
fn unlink_created_path(path: &Path, recorded: &CreatedIdentity) -> std::io::Result<EarlyUnlink> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(EarlyUnlink::AlreadyUnnamed),
        Err(e) => Err(e),
        Ok(meta) if identity_of_metadata(&meta) == *recorded => {
            fs::remove_file(path)?;
            Ok(EarlyUnlink::Unlinked)
        }
        Ok(_) => Ok(EarlyUnlink::ForeignPath),
    }
}

/// The writable phase of a snapshot: a freshly created read-write
/// file, briefly named until [`WritableSnapshot::unlink_before_copy`]
/// succeeds. While `named` holds, drop removes the pathname only
/// if it still names this file, with the file retained in `file`
/// across that decision; a foreign replacement is preserved and
/// the open that owns this value refuses. Once unnamed, drop only
/// closes the file, which frees the unlinked inode — there is no
/// cleanup by pathname from that point on. [`WritableSnapshot::seal`]
/// consumes this value so the writable file cannot be used, or
/// escape, after the seal point.
#[cfg(unix)]
struct WritableSnapshot {
    file: Option<File>,
    former_path: PathBuf,
    identity: CreatedIdentity,
    named: bool,
}

#[cfg(unix)]
impl WritableSnapshot {
    /// Creates a fresh no-clobber read-write snapshot file in the
    /// system temporary directory, private to this user (mode
    /// `0600`, requested at creation), and records its identity
    /// from the retained file for the pre-unlink decision.
    fn create(input: &Path) -> Result<Self, ToolError> {
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
            options.read(true).write(true).create_new(true);
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&candidate) {
                Ok(file) => match identity_of_file(&file) {
                    Ok(identity) => {
                        return Ok(Self {
                            file: Some(file),
                            former_path: candidate,
                            identity,
                            named: true,
                        });
                    }
                    // Metadata on a just-created retained file
                    // failing is practically unreachable. Without
                    // an identity no checked unlink is possible, so
                    // refuse and preserve whatever the pathname
                    // names rather than unlink it unchecked; the
                    // retained file closes with this error return.
                    Err(e) => return Err(ToolError::io(input, e)),
                },
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

    fn former_path(&self) -> &Path {
        &self.former_path
    }

    /// Unlinks the just-created pathname before any bytes are
    /// copied. A foreign file at the pathname is preserved and
    /// the open refuses; an already-absent pathname means our
    /// file is already unnamed and the copy proceeds unnamed.
    fn unlink_before_copy(&mut self, input: &Path) -> Result<(), ToolError> {
        match unlink_created_path(&self.former_path, &self.identity)
            .map_err(|e| ToolError::io(input, e))?
        {
            EarlyUnlink::Unlinked | EarlyUnlink::AlreadyUnnamed => {
                self.named = false;
                Ok(())
            }
            EarlyUnlink::ForeignPath => {
                // The pathname no longer names our file, so this
                // value is unnamed; dropping it must not touch the
                // foreign path.
                self.named = false;
                Err(ToolError::refused(format!(
                    "{}: snapshot pathname was replaced before it could be \
                     unlinked; refusing",
                    input.display()
                )))
            }
        }
    }

    fn file_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("writable snapshot file present")
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.file_mut().write_all(buf)
    }

    fn flush_and_sync(&mut self) -> std::io::Result<()> {
        let file = self.file_mut();
        file.flush()?;
        file.sync_all()
    }

    fn file_len(&self) -> std::io::Result<u64> {
        self.file
            .as_ref()
            .expect("writable snapshot file present")
            .metadata()
            .map(|m| m.len())
    }

    /// Ends the writable phase. The file moves into a sealed
    /// value that exposes no writable capability; this consumed
    /// value then drops with nothing left to clean up.
    fn seal(mut self, len: u64) -> SealedSnapshot {
        debug_assert!(!self.named, "snapshot sealed while still named");
        self.named = false;
        SealedSnapshot {
            file: self.file.take().expect("writable snapshot sealed once"),
            former_path: self.former_path.clone(),
            len,
        }
    }
}

#[cfg(unix)]
impl Drop for WritableSnapshot {
    fn drop(&mut self) {
        if self.named {
            // `file` is still retained here (fields drop after
            // this body), pinning the inode through the check.
            let _ = remove_if_still_ours(&self.former_path, &self.identity);
        }
    }
}

#[cfg(unix)]
static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The crate's single `unsafe` call: map a sealed private
/// snapshot read-only.
///
/// SAFETY: `memmap2` marks mapping `unsafe` because modifying or
/// truncating the backing file while it is mapped — from any
/// process — is undefined behavior. For a snapshot mapped here
/// this process upholds its side of that requirement, and the
/// remainder is the stated trust boundary:
///
/// - The file was created `create_new` read-write by this process
///   in the system temporary directory (`0600` on Unix). Its
///   pathname was unlinked, through an identity check against the
///   retained file, immediately after creation and before any
///   bytes were copied, so through the copy and the mapping
///   lifetime this module reaches the inode only through
///   already-open descriptors, not through its former pathname.
///   Cleanup after
///   that point touches no pathname, so a foreign replacement at
///   the former path — including one reusing the freed inode
///   number — is not read and not deleted by this module.
/// - The copy, hash and `sync_all` went through the writable
///   phase's retained file. The file length was then read from
///   that file, checked against the copied length, and fixed in
///   the sealed value; the mapping length is checked against it
///   before the snapshot is returned. Sealing consumed the
///   writable value: `SealedSnapshot` exposes no file, no mutable
///   reference and no write or truncate operation, and this
///   process performs no write, extend or truncate after the
///   seal. The handle mapped here is that same already-open file
///   (no reopen by pathname or through a descriptor-table path),
///   mapped as a read-only `Mmap`; `Snapshot` field order drops
///   (unmaps) the mapping before the sealed file is closed.
/// - Outside this custody, explicitly, for the whole mapping
///   lifetime: a same-UID actor who already holds a descriptor
///   to the snapshot inode, or who can open `/proc/<pid>/fd`
///   entries of this process, could still modify the inode
///   through that access. No result is claimed against that
///   actor. The sealed file's own descriptor stays open for the
///   mapping, but it is reachable only through the sealed value
///   above, which offers no write path.
/// - The *source* file changing during the copy is a separate
///   concern: in this module it is detected only by the
///   before/after source size checks in `Snapshot::create_impl`.
///   Downstream parse and hash gates in callers are separate
///   checks and do not widen what is stated here.
///
/// Non-Unix snapshot creation refuses before reaching this call.
#[cfg(unix)]
#[allow(unsafe_code)]
fn map_readonly(file: &File, input: &Path) -> Result<Mmap, ToolError> {
    // SAFETY: sealed private snapshot custody, argued above.
    let mmap = unsafe { MmapOptions::new().map(file) };
    mmap.map_err(|e| ToolError::io(input, e))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A same-length crafted file placed at the former snapshot
    /// pathname — here at the deterministic `after_unlink` seam,
    /// so it is present for the whole copy, seal and mapping — is
    /// neither read nor deleted. The snapshot succeeds with the
    /// copied source bytes because custody is bound to the
    /// retained sealed file, not to the pathname an inode-reusing
    /// filesystem could recycle: the foreign bytes are never
    /// parsed and the foreign file still survives.
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
            .expect("sealed-fd snapshot must ignore the former pathname");
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

    /// The former snapshot pathname is recreated with foreign
    /// bytes after the snapshot file was unlinked, which on
    /// ext4-family filesystems reuses the freed inode (so a
    /// device+inode cleanup check could not tell it from ours).
    /// Snapshot drop is safe here because after the early unlink
    /// it has no pathname to remove: the foreign file survives
    /// snapshot drop and the snapshot bytes never change. The
    /// still-named pre-unlink custody that remains is pinned by
    /// the controls at the end: an unsubstituted writable
    /// snapshot is removed on drop, a substituted one is kept.
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

        // Control: a still-named writable snapshot (pre-unlink
        // custody) still removes its own unsubstituted file.
        let input = tmp.path().join("input.bin");
        fs::write(&input, b"x").unwrap();
        let mut writable = WritableSnapshot::create(&input).unwrap();
        let path = writable.former_path().to_path_buf();
        writable.write_all(b"ours").unwrap();
        writable.flush_and_sync().unwrap();
        drop(writable);
        assert!(!path.exists(), "own snapshot must be cleaned up");

        // Control: the same still-named drop preserves a foreign
        // replacement at the pathname.
        let writable = WritableSnapshot::create(&input).unwrap();
        let path = writable.former_path().to_path_buf();
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"foreign!").unwrap();
        drop(writable);
        assert_eq!(fs::read(&path).unwrap(), b"foreign!");
        fs::remove_file(&path).unwrap();
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
        let former = snapshot.sealed.former_path().to_path_buf();
        assert!(
            !former.exists(),
            "snapshot pathname must be absent while the snapshot is alive"
        );
        assert_eq!(snapshot.bytes(), vec![b'A'; 64].as_slice());
        drop(snapshot);
        assert!(!former.exists(), "no snapshot file must remain after drop");
    }

    /// A refused snapshot (source changed size during the copy)
    /// leaves no snapshot file: the copy went into the
    /// already-unlinked inode, which is freed when its file
    /// closes, with no pathname cleanup involved. With a foreign
    /// file planted at the former pathname before the same
    /// triggered failure, the refusal still publishes nothing and
    /// the foreign file is kept byte-intact.
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

        // Same triggered failure with a foreign file at the
        // former pathname: it is neither read nor deleted.
        let source2 = tmp.path().join("source2.bin");
        fs::write(&source2, b"ours").unwrap();
        let snap_path2: RefCell<Option<PathBuf>> = RefCell::new(None);
        let grown2 = source2.clone();
        let seam2 = |path: &Path| {
            *snap_path2.borrow_mut() = Some(path.to_path_buf());
            fs::write(path, b"foreign!").unwrap();
            OpenOptions::new()
                .append(true)
                .open(&grown2)
                .unwrap()
                .write_all(b"-more")
                .unwrap();
        };
        let err = Snapshot::create_impl(&source2, Some(&seam2))
            .expect_err("size-changed source must be refused");
        assert!(matches!(err, ToolError::Refused { .. }), "{err}");
        assert!(err.to_string().contains("changed size"), "{err}");
        let former2 = snap_path2.borrow().clone().expect("seam ran");
        assert_eq!(fs::read(&former2).unwrap(), b"foreign!");
        fs::remove_file(&former2).unwrap();
    }
}
