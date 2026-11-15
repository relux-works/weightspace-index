use std::path::Path;

use crate::error::ToolError;

/// Which part of a sparse trio an entry name denotes, or a full tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Full,
    Idx,
    New,
    Old,
}

/// Splits an entry name into its tensor name and kind, mirroring
/// `layout::check_replacement_file`: a `#` introduces exactly one of
/// the `idx`/`new`/`old` suffixes; anything else is refused.
pub fn parse_entry_name<'a>(path: &Path, name: &'a str) -> Result<(&'a str, EntryKind), ToolError> {
    match name.split_once('#') {
        None => Ok((name, EntryKind::Full)),
        Some((base, "idx")) => Ok((base, EntryKind::Idx)),
        Some((base, "new")) => Ok((base, EntryKind::New)),
        Some((base, "old")) => Ok((base, EntryKind::Old)),
        Some((_, suffix)) => Err(ToolError::malformed(
            path,
            format!("{name}: unknown suffix #{suffix}"),
        )),
    }
}
