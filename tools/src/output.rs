//! The output path contract: one implementation on every OS.
//!
//! An output location is usable only as an [`OutputTarget`], which
//! can be constructed solely by [`OutputTarget::check`]; publication
//! (`publish`) takes that type, so an unchecked path cannot be
//! published. The check is symmetric with `files::check_input_dir`:
//! only the parent's own final component and the output name itself
//! are checked as spelled; links above the parent are resolved by
//! one canonicalization, and every later create, link and rename
//! uses the canonical parent plus the name. Messages always show
//! the user's spelled path ([`OutputTarget::spelled`]), never a
//! canonical `\\?\` form.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use crate::error::ToolError;

/// An output location that passed [`OutputTarget::check`]. Publication
/// takes this type, so an unchecked path cannot be published.
#[derive(Debug)]
pub(crate) struct OutputTarget {
    spelled: PathBuf,
    parent: PathBuf,
    name: OsString,
}

impl OutputTarget {
    /// Checks `spelled` as an output location:
    ///
    /// 1. The final component (after `components()` normalization)
    ///    must be `Component::Normal`; `.`, `..`, root/prefix-only
    ///    and separator-only spellings are `InvalidOutputName`. On
    ///    Windows, DOS device names and names ending in `.` or space
    ///    are `InvalidOutputName` too (through a `\\?\` canonical
    ///    parent they would create files the shell cannot handle;
    ///    without that prefix they reach devices). The filter is a
    ///    `cfg!(windows)` expression, so it compiles and is
    ///    unit-tested on every OS.
    /// 2. The parent as spelled (after the same normalization, so
    ///    `link`, `link/` and `link/.` parent spellings are covered)
    ///    must be a real directory: its `symlink_metadata` must not
    ///    be a symlink and must be a directory. On Windows std
    ///    reports every name-surrogate reparse point as a symlink,
    ///    which covers symlinks and junctions alike. The parent
    ///    must already exist; missing parents are not created.
    /// 3. Links above the parent are followed: `canonicalize(parent)`
    ///    once (macOS `/var`, `/tmp`, Windows short names and the
    ///    `\\?\` prefix all resolve here).
    /// 4. The final name must not exist in any form;
    ///    `symlink_metadata` covers dangling links. A final-component
    ///    symlink is `SymlinkRefused`, any other existing entry is
    ///    `OutputExists`; neither is ever written through.
    ///
    /// The resolution is check-then-act: a same-UID actor who can
    /// write in the parent could swap a component between this check
    /// and publication. That race is outside the guarantee (file
    /// publication is still no-clobber, directory publication keeps
    /// the `publish_dir` limits) and no atomicity is claimed for the
    /// resolution itself. A link planted after the check cannot
    /// redirect the writes themselves: temporaries are created with
    /// `create_new`, which does not follow links, and neither
    /// `link(2)`/`CreateHardLinkW` nor the directory claim follows
    /// a destination link.
    pub(crate) fn check(spelled: &Path) -> Result<Self, ToolError> {
        let invalid = || ToolError::InvalidOutputName {
            path: spelled.to_owned(),
        };
        let normalized: PathBuf = spelled.components().collect();
        let name = match normalized.components().next_back() {
            Some(Component::Normal(name)) => name.to_owned(),
            _ => return Err(invalid()),
        };
        if cfg!(windows) && is_unsafe_windows_name(&name) {
            return Err(invalid());
        }

        let parent_spelled = match normalized.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let meta =
            fs::symlink_metadata(parent_spelled).map_err(|e| ToolError::io(parent_spelled, e))?;
        if meta.file_type().is_symlink() {
            return Err(ToolError::SymlinkRefused {
                path: parent_spelled.to_owned(),
                detail: "output parent directory must not be a symlink or junction".into(),
            });
        }
        if !meta.is_dir() {
            return Err(ToolError::OutputParentNotDir {
                path: parent_spelled.to_owned(),
            });
        }
        let parent =
            fs::canonicalize(parent_spelled).map_err(|e| ToolError::io(parent_spelled, e))?;

        match fs::symlink_metadata(parent.join(&name)) {
            Ok(meta) if meta.file_type().is_symlink() => Err(ToolError::SymlinkRefused {
                path: spelled.to_owned(),
                detail: format!("output path {} must not be a symlink", spelled.display()),
            }),
            Ok(_) => Err(ToolError::OutputExists {
                path: spelled.to_owned(),
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self {
                spelled: spelled.to_owned(),
                parent,
                name,
            }),
            Err(e) => Err(ToolError::io(spelled, e)),
        }
    }

    /// Canonical parent directory: where temp files are created.
    pub(crate) fn parent(&self) -> &Path {
        &self.parent
    }
    /// Canonical destination path.
    pub(crate) fn path(&self) -> PathBuf {
        self.parent.join(&self.name)
    }
    /// The path as the user wrote it, for messages.
    pub(crate) fn spelled(&self) -> &Path {
        &self.spelled
    }
}

/// Whether canonical path `inner` is `outer` or lies below it.
/// Components compare ASCII-case-insensitively on every platform: on
/// APFS and NTFS defaults that is the file system's own rule, and on
/// case-sensitive file systems it can only refuse more, never less.
pub(crate) fn is_within(inner: &Path, outer: &Path) -> bool {
    let mut inner = inner.components();
    outer.components().all(|o| {
        inner
            .next()
            .is_some_and(|i| i.as_os_str().eq_ignore_ascii_case(o.as_os_str()))
    })
}

/// Whether `name` is unsafe as a Windows file name: a DOS device name
/// (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`, with any
/// extension), or a name ending in `.` or space. The superscript-digit
/// device names Microsoft lists (`COM¹` and kin) are not covered.
/// Compiled and unit-tested on every OS; applied under
/// `cfg!(windows)`.
fn is_unsafe_windows_name(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    if name.ends_with('.') || name.ends_with(' ') {
        return true;
    }
    let stem = name.split('.').next().unwrap_or("");
    let upper = stem.to_ascii_uppercase();
    matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.len() == 4
            && upper.as_bytes()[3].is_ascii_digit()
            && upper.as_bytes()[3] != b'0')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn canonical_tmp() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        (tmp, root)
    }

    // X1: a symlinked ancestor above a real output parent is accepted and resolved.
    #[cfg(unix)]
    #[test]
    fn symlinked_ancestor_accepted() {
        use std::os::unix::fs::symlink;
        let (_tmp, root) = canonical_tmp();
        // The macOS shape, constructed: var -> private/var, with the
        // output parent a real directory below the link.
        let real_parent = root.join("private").join("var").join("parent");
        fs::create_dir_all(&real_parent).unwrap();
        symlink(root.join("private").join("var"), root.join("var")).unwrap();
        let spelled = root.join("var").join("parent").join("out.bin");
        let target = OutputTarget::check(&spelled).unwrap();
        assert_eq!(target.path(), real_parent.join("out.bin"));
        assert_eq!(target.parent(), real_parent);
        assert_eq!(target.spelled(), spelled);
        assert!(!target.path().exists());
    }

    // X2: a symlinked output parent is refused under link, link/ and link/. spellings.
    #[cfg(unix)]
    #[test]
    fn symlinked_output_parent_refused() {
        use std::os::unix::fs::symlink;
        let (_tmp, root) = canonical_tmp();
        let real = root.join("real");
        fs::create_dir(&real).unwrap();
        let link = root.join("link");
        symlink(&real, &link).unwrap();
        for spelled in [
            link.join("out.bin"),
            PathBuf::from(format!("{}//out.bin", link.display())),
            PathBuf::from(format!("{}/./out.bin", link.display())),
        ] {
            let err = OutputTarget::check(&spelled).unwrap_err();
            assert!(
                matches!(err, ToolError::SymlinkRefused { .. }),
                "{}: {err}",
                spelled.display()
            );
        }
        assert!(!real.join("out.bin").exists());
    }

    // X3/X4: a final-component link (live or dangling) and any existing entry are refused, target untouched.
    #[cfg(unix)]
    #[test]
    fn output_final_symlink_and_existing_refused() {
        use std::os::unix::fs::symlink;
        let (_tmp, root) = canonical_tmp();
        let victim = root.join("victim.bin");
        fs::write(&victim, b"x").unwrap();
        let out_link = root.join("out-link.bin");
        symlink(&victim, &out_link).unwrap();
        let err = OutputTarget::check(&out_link).unwrap_err();
        assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
        let dangling = root.join("dangling.bin");
        symlink(root.join("missing-target"), &dangling).unwrap();
        let err = OutputTarget::check(&dangling).unwrap_err();
        assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
        assert_eq!(fs::read(&victim).unwrap(), b"x");
        let err = OutputTarget::check(&victim).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
        let existing_dir = root.join("existing-dir");
        fs::create_dir(&existing_dir).unwrap();
        let err = OutputTarget::check(&existing_dir).unwrap_err();
        assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
    }

    // A dangling symlink as the output parent is refused, never written through.
    #[cfg(unix)]
    #[test]
    fn planted_tail_symlink_refused() {
        use std::os::unix::fs::symlink;
        let (_tmp, root) = canonical_tmp();
        symlink(root.join("missing-target"), root.join("dangling")).unwrap();
        let err = OutputTarget::check(&root.join("dangling").join("out.bin")).unwrap_err();
        assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
        assert!(!root.join("missing-target").exists());
    }

    // A missing output parent is a typed Io refusal; no parent directories are created.
    #[test]
    fn missing_output_parent_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("missing");
        let err = OutputTarget::check(&missing.join("out.bin")).unwrap_err();
        assert!(
            matches!(&err, ToolError::Io { source, .. } if source.kind() == io::ErrorKind::NotFound),
            "{err}"
        );
        assert!(!missing.exists());
    }

    // X8: dot and dot-dot spellings normalize per contract; name-less spellings are InvalidOutputName.
    #[test]
    fn relative_and_dot_spellings() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        for spelled in [
            PathBuf::from(format!("{}/./out.bin", root.display())),
            PathBuf::from(format!("{}/sub/../out.bin", root.display())),
        ] {
            let target = OutputTarget::check(&spelled).unwrap();
            assert_eq!(target.path(), root.join("out.bin"), "{}", spelled.display());
            assert_eq!(target.spelled(), spelled);
        }
        // A bare relative name checks against the cwd parent.
        let target = OutputTarget::check(Path::new(".weightspace-x8-probe.bin")).unwrap();
        assert_eq!(target.parent(), fs::canonicalize(".").unwrap());
        for spelled in [
            Path::new("."),
            Path::new(".."),
            Path::new("/"),
            Path::new(""),
        ] {
            let err = OutputTarget::check(spelled).unwrap_err();
            assert!(
                matches!(err, ToolError::InvalidOutputName { .. }),
                "{}: {err}",
                spelled.display()
            );
        }
    }

    // R3 (unit, every OS): DOS device names and trailing dot/space are unsafe Windows names.
    #[test]
    fn windows_unsafe_output_names_refused() {
        for name in [
            "NUL",
            "NUL.safetensors",
            "nul.txt",
            "CON",
            "con.foo",
            "PRN",
            "AUX",
            "COM1",
            "com9.bin",
            "LPT3",
            "lpt1.safetensors",
            "out.",
            "out ",
        ] {
            assert!(is_unsafe_windows_name(OsStr::new(name)), "{name}");
        }
        for name in ["out.bin", "null", "com10", "com0", "console", "lpt", "aux1"] {
            assert!(!is_unsafe_windows_name(OsStr::new(name)), "{name}");
        }
    }

    // X6 (unit): containment is component-wise and ASCII-case-insensitive on every OS.
    #[test]
    fn is_within_is_componentwise_case_insensitive() {
        assert!(is_within(Path::new("/a/Model"), Path::new("/a/model")));
        assert!(is_within(Path::new("/a/model/sub"), Path::new("/A/MODEL")));
        assert!(is_within(Path::new("/a/model"), Path::new("/a/model")));
        assert!(!is_within(Path::new("/a/model2"), Path::new("/a/model")));
        assert!(!is_within(Path::new("/a"), Path::new("/a/model")));
    }
}
