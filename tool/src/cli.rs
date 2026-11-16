//! Command-line argument types only. Parsing, mapping to library
//! operations and printing live in `main.rs`; semantics live in
//! `weightspace_tools::ops`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use weightspace_tools::error::ToolError;
use weightspace_tools::files::Sha256Hex;

#[derive(Parser)]
#[command(
    name = "weightspace",
    about = "Checks, generator and patch tools for the Weightspace Index"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run every offline check; fails when a generated file is missing,
    /// out of date or stale.
    Check {
        /// Repository root (default: current directory).
        repo: Option<PathBuf>,
    },
    /// Rewrite lookup/, models/index.json, models/README.md and per-model
    /// indexes/tables in the repository and docs/example/.
    Generate {
        /// Repository root (default: current directory).
        repo: Option<PathBuf>,
    },
    /// Build one replacement safetensors file from two checkpoints with
    /// the exact same tensor name/shape/dtype set.
    Replacement {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        edited: PathBuf,
        /// Optional authors' manifest (tensors[] hash pairs) gate.
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// Directory for the private snapshots of the inputs
        /// (default: the system temporary directory). It needs
        /// free space for a copy of each input checkpoint held at
        /// once; on Linux the default is often tmpfs (RAM), so
        /// pass a disk directory for large checkpoints.
        #[arg(long = "temp-dir")]
        temp_dir: Option<PathBuf>,
    },
    /// Bake a replacement file into a new copy of a model directory
    /// holding a single root model.safetensors. The original is never
    /// modified.
    Apply {
        #[arg(long)]
        patch: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Expected base SHA-256 for one full-form patch tensor, as
        /// NAME=HEX. Mandatory for every full-form tensor: the patch
        /// file does not record the bytes it replaces.
        #[arg(long = "expect-tensor-sha256")]
        expect_tensor_sha256: Vec<String>,
        /// Directory for the private snapshots of the inputs
        /// (default: the system temporary directory). It needs
        /// free space for a copy of each input checkpoint held at
        /// once; on Linux the default is often tmpfs (RAM), so
        /// pass a disk directory for large checkpoints.
        #[arg(long = "temp-dir")]
        temp_dir: Option<PathBuf>,
    },
}

/// Parses repeated `--expect-tensor-sha256 NAME=HEX` arguments.
pub fn parse_expect(items: &[String]) -> Result<BTreeMap<String, Sha256Hex>, ToolError> {
    let mut out = BTreeMap::new();
    for item in items {
        let invalid = |detail: String| ToolError::InvalidArgument { detail };
        let (name, hex) = item.split_once('=').ok_or_else(|| {
            invalid(format!(
                "--expect-tensor-sha256 expects NAME=HEX, got {item:?}"
            ))
        })?;
        if name.is_empty() {
            return Err(invalid(format!(
                "--expect-tensor-sha256 expects a non-empty NAME, got {item:?}"
            )));
        }
        let digest = Sha256Hex::parse(hex).map_err(|_| {
            invalid(format!(
                "--expect-tensor-sha256 {name}: expected 64 hex characters, got {hex:?}"
            ))
        })?;
        if out.insert(name.to_owned(), digest).is_some() {
            return Err(invalid(format!(
                "--expect-tensor-sha256 names {name} twice"
            )));
        }
    }
    Ok(out)
}
