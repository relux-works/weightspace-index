use std::fs::{self, File, OpenOptions};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::ToolError;
use crate::output::OutputTarget;

/// Makes a just-published name durable. Unix only: std cannot open a
/// directory on Windows without `FILE_FLAG_BACKUP_SEMANTICS`, and
/// `FlushFileBuffers` needs write access; Windows publication relies on
/// NTFS metadata journaling and claims no more.
#[cfg(unix)]
pub(crate) fn sync_dir(dir: &Path) -> Result<(), ToolError> {
    fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| ToolError::io(dir, e))
}

#[cfg(windows)]
pub(crate) fn sync_dir(_dir: &Path) -> Result<(), ToolError> {
    Ok(())
}

/// Publishes the finished file at `temp` (created in `target.parent()`)
/// under the target name. `link(2)` and `CreateHardLinkW` both fail if
/// the name exists, so this cannot clobber. Errors after the link
/// leave a complete output in place. On a filesystem without hard
/// links (FAT32, exFAT) the link fails and surfaces as a typed `Io`
/// naming the destination and cause.
pub(crate) fn publish_file(temp: &Path, target: &OutputTarget) -> Result<(), ToolError> {
    let dest = target.path();
    match fs::hard_link(temp, &dest) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            return Err(ToolError::OutputExists {
                path: target.spelled().to_owned(),
            });
        }
        Err(e) => return Err(ToolError::io(&dest, e)),
    }
    fs::remove_file(temp).map_err(|e| ToolError::io(temp, e))?;
    sync_dir(target.parent())
}

/// Unix: `mkdir` claims the name atomically, then `rename(2)` replaces
/// that still-empty claim atomically. If anything was put into the
/// claim meanwhile, rename fails (`ENOTEMPTY`/`EEXIST`) and nothing is
/// replaced.
#[cfg(unix)]
pub(crate) fn publish_dir(temp: &Path, target: &OutputTarget) -> Result<(), ToolError> {
    let dest = target.path();
    sync_dir(temp)?;
    fs::create_dir(&dest).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => ToolError::OutputExists {
            path: target.spelled().to_owned(),
        },
        _ => ToolError::io(&dest, e),
    })?;
    if let Err(e) = fs::rename(temp, &dest) {
        // Removes the claim only while it is still empty.
        let _ = fs::remove_dir(&dest);
        return Err(match e.kind() {
            io::ErrorKind::DirectoryNotEmpty | io::ErrorKind::AlreadyExists => {
                ToolError::OutputExists {
                    path: target.spelled().to_owned(),
                }
            }
            _ => ToolError::io(&dest, e),
        });
    }
    sync_dir(target.parent())
}

/// Windows: no claim. `std::fs::rename` calls `MoveFileExW` with
/// `MOVEFILE_REPLACE_EXISTING`, which reports an error for an existing
/// directory destination, but on `ERROR_ACCESS_DENIED` std retries with
/// POSIX-semantics replace (library/std/src/sys/fs/windows.rs:1305-1370),
/// which may replace an empty directory. So: check, then rename; a
/// concurrent same-user writer in the parent is the documented residual.
#[cfg(windows)]
pub(crate) fn publish_dir(temp: &Path, target: &OutputTarget) -> Result<(), ToolError> {
    let dest = target.path();
    if fs::symlink_metadata(&dest).is_ok() {
        return Err(ToolError::OutputExists {
            path: target.spelled().to_owned(),
        });
    }
    fs::rename(temp, &dest).map_err(|e| ToolError::io(&dest, e))
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

    /// Syncs the finished file and publishes it under the checked
    /// target name without clobbering (see [`publish_file`]). Once
    /// the hard link has succeeded the destination is published; a
    /// later I/O error removing the temp name or syncing the
    /// directory is still returned as `Err`, but a complete visible
    /// output may remain at the destination in that case.
    pub(crate) fn publish(mut self, target: &OutputTarget) -> Result<(), ToolError> {
        self.sync()?;
        drop(self.file.take());
        publish_file(&self.path, target)?;
        self.published = true;
        Ok(())
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

    /// Publishes the finished directory under the checked target
    /// name (see [`publish_dir`]). Once the rename has succeeded the
    /// destination is published; a later directory-sync error is
    /// still returned as `Err`, but a complete visible output may
    /// remain at the destination in that case.
    pub(crate) fn publish(mut self, target: &OutputTarget) -> Result<(), ToolError> {
        publish_dir(&self.path, target)?;
        self.published = true;
        Ok(())
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

    fn canonical_tmp() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        (tmp, root)
    }

    // P1: a name created after the check refuses publish_file and keeps its old bytes.
    #[test]
    fn publish_file_no_clobber() {
        let (_tmp, root) = canonical_tmp();
        let target = OutputTarget::check(&root.join("out.bin")).unwrap();
        let mut temp = TempFile::create_in(target.parent()).unwrap();
        temp.file_mut().write_all(b"new").unwrap();
        fs::write(root.join("out.bin"), b"old").unwrap();
        let temp_path = temp.path().to_path_buf();
        let err = temp.publish(&target).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
        assert_eq!(fs::read(root.join("out.bin")).unwrap(), b"old");
        assert!(!temp_path.exists());
        // Positive control: an unraced target publishes.
        let target = OutputTarget::check(&root.join("ok.bin")).unwrap();
        let mut temp = TempFile::create_in(target.parent()).unwrap();
        temp.file_mut().write_all(b"new").unwrap();
        temp.publish(&target).unwrap();
        assert_eq!(fs::read(root.join("ok.bin")).unwrap(), b"new");
    }

    // P2: publish_dir never replaces a foreign directory that appeared after the check.
    #[test]
    fn publish_dir_never_replaces_foreign_dir() {
        let (_tmp, root) = canonical_tmp();
        let target = OutputTarget::check(&root.join("out-dir")).unwrap();
        let temp = TempDir::create_in(target.parent()).unwrap();
        fs::write(temp.path().join("mine.txt"), b"mine").unwrap();
        let foreign = root.join("out-dir");
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join("foreign.txt"), b"foreign").unwrap();
        let temp_path = temp.path().to_path_buf();
        let err = temp.publish(&target).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
        assert_eq!(fs::read(foreign.join("foreign.txt")).unwrap(), b"foreign");
        assert!(!foreign.join("mine.txt").exists());
        assert!(!temp_path.exists());
        // Positive control: an unraced target publishes.
        let target = OutputTarget::check(&root.join("ok-dir")).unwrap();
        let temp = TempDir::create_in(target.parent()).unwrap();
        fs::write(temp.path().join("mine.txt"), b"mine").unwrap();
        temp.publish(&target).unwrap();
        assert_eq!(
            fs::read(root.join("ok-dir").join("mine.txt")).unwrap(),
            b"mine"
        );
    }

    // P3: publish_dir onto a foreign file that appeared after the check is refused.
    #[test]
    fn publish_dir_onto_foreign_file_refused() {
        let (_tmp, root) = canonical_tmp();
        let target = OutputTarget::check(&root.join("out-dir")).unwrap();
        let temp = TempDir::create_in(target.parent()).unwrap();
        fs::write(temp.path().join("mine.txt"), b"mine").unwrap();
        fs::write(root.join("out-dir"), b"foreign file").unwrap();
        let temp_path = temp.path().to_path_buf();
        let err = temp.publish(&target).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
        assert_eq!(fs::read(root.join("out-dir")).unwrap(), b"foreign file");
        assert!(!temp_path.exists());
    }
}
