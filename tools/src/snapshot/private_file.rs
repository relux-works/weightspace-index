//! The one platform-specific snapshot step: create a fresh file no
//! other process can open by name. The custody each body gives is
//! stated once in the parent module docs; here only the creation
//! mechanics differ. Both bodies have the identical signature, use
//! std alone, and share the bounded name loop below (which takes
//! the prepared `OpenOptions` as data, not a closure).

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_ATTEMPTS: u32 = 64;

/// Creates a fresh file in `dir` that no other process can open by
/// name: unlinked before return on Unix; opened with share mode 0
/// and delete-on-close on Windows. The handle is readable and
/// writable.
#[cfg(unix)]
pub(super) fn create_in(dir: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true).mode(0o600);
    let (path, file) = create_unique(dir, &options)?;
    std::fs::remove_file(&path)?;
    Ok(file)
}

#[cfg(windows)]
pub(super) fn create_in(dir: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    // Values from WinNT.h / WinBase.h; the same combination
    // tempfile 3.27 uses for its unnamed temp files
    // (src/file/imp/windows.rs:49-55).
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x0000_0100;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;

    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create_new(true)
        .share_mode(0)
        .attributes(FILE_ATTRIBUTE_TEMPORARY)
        .custom_flags(FILE_FLAG_DELETE_ON_CLOSE);
    let (_path, file) = create_unique(dir, &options)?;
    Ok(file)
}

/// `create_new` maps to `O_CREAT|O_EXCL` on Unix and to
/// `CREATE_NEW` plus `FILE_FLAG_OPEN_REPARSE_POINT` on Windows, so
/// a name planted in advance (file or link) is never opened, only
/// skipped. Retries are bounded.
fn create_unique(dir: &Path, options: &OpenOptions) -> io::Result<(PathBuf, File)> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    for _ in 0..MAX_ATTEMPTS {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let path = dir.join(format!(
            ".weightspace-snapshot-{}-{n}-{nanos}.tmp",
            std::process::id()
        ));
        match options.open(&path) {
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            result => return result.map(|file| (path, file)),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "no free snapshot name after 64 attempts",
    ))
}
