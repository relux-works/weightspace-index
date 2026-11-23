//! Checks, generator, patch tools, multi-registry client, engine discovery,
//! and interactive TUI for the Weightspace Index.
//!
//! Offline index/patch modules (`checkpoint`, `patch`, `files`, `ops`,
//! `derive_lora`, `records`, `layout`, `generate`) never touch the network or
//! execute repository content.
// `forbid` gave way to `deny` for exactly one allow: the
// snapshot mapping call in `snapshot::seal`, whose SAFETY
// argument rests on private snapshot custody. Everything
// else in the crate is unsafe-free.
#![deny(unsafe_code)]

pub mod checkpoint;
pub mod cli_app;
pub mod derive_lora;
pub(crate) mod digest;
pub mod download;
pub mod engines;
pub mod error;
pub mod files;
pub mod generate;
pub mod layout;
pub(crate) mod model_dir;
pub mod ops;
pub(crate) mod output;
pub mod patch;
pub(crate) mod publish;
pub mod records;
pub mod registry;
mod snapshot;
pub mod tui;
