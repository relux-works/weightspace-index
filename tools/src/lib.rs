//! Checks for the Weightspace Index: every JSON record against
//! `schema/records.schema.json`, and the layout rules a single record
//! cannot express. CI and the intake tool share these checks.
//!
//! The patch tooling (`checkpoint`, `patch`, `files`, `ops`, `error`)
//! builds and applies safetensors replacement files over synthetic or
//! real checkpoints with typed refusals and outputs that refuse
//! existing destinations: single files publish by atomic no-clobber
//! hard link, while model directories publish by local
//! check-then-rename, which is not atomic no-clobber against a
//! concurrent writer in the destination directory (see `files`).
//! Checkpoints are read through the official `safetensors` reader
//! over a private mapped snapshot of each input (see `checkpoint`
//! and `snapshot`).
//!
//! Nothing here touches the network or executes repository content.
// `forbid` gave way to `deny` for exactly one allow: the
// snapshot mapping call in `snapshot::map_readonly`, whose SAFETY
// argument rests on private immutable snapshot custody. Everything
// else in the crate is unsafe-free.
#![deny(unsafe_code)]

pub mod checkpoint;
pub mod error;
pub mod files;
pub mod generate;
pub mod layout;
pub mod ops;
pub mod patch;
pub mod records;
mod snapshot;
