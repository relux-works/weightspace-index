use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::ToolError;
use crate::output::parent_dir;

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
