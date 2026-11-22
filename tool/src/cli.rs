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
    /// Derive experimental LoRA factors. Study claims NEVER transfer to LoRA;
    /// it needs its own evaluation. PEFT/llama.cpp/vLLM loading is UNQUALIFIED.
    DeriveLora {
        #[arg(
            long,
            required_unless_present = "tensor_shapes",
            conflicts_with = "tensor_shapes"
        )]
        model: Option<PathBuf>,
        /// Offline sparse derivation: JSON name -> [rows,width], selector-exact only.
        #[arg(long)]
        tensor_shapes: Option<PathBuf>,
        #[arg(long)]
        patch: PathBuf,
        #[arg(long)]
        rank: Option<usize>,
        /// Stored factor dtype (F32 only in this version).
        #[arg(long, value_parser = ["f32"])]
        factor_dtype: String,
        #[arg(long, value_parser = ["svd", "selector", "selector-exact"])]
        method: String,
        #[arg(long)]
        out: PathBuf,
        /// Mandatory expected base digest for every full-form patch tensor.
        #[arg(long = "expect-tensor-sha256")]
        expect_tensor_sha256: Vec<String>,
        #[arg(long = "temp-dir")]
        temp_dir: Option<PathBuf>,
    },
    /// Bake one or more replacement files into a new copy of a model directory
    /// holding a single root model.safetensors (default), or modify in place
    /// with an automatic verified backup (`--in-place`).
    Apply {
        /// One or more replacement `.safetensors` files (pass repeatedly to stack patches;
        /// use `PATH:WEIGHT` with `--on-conflict=weighted`).
        #[arg(long, required = true)]
        patch: Vec<String>,
        #[arg(long)]
        model: PathBuf,
        /// Destination directory for the patched copy (required unless `--in-place` is set).
        #[arg(
            long,
            required_unless_present = "in_place",
            conflicts_with = "in_place"
        )]
        out: Option<PathBuf>,
        /// Patch the model directory in place after saving a SHA-256-verified backup under
        /// `.weightspace-backup/`. Reversible via `weightspace restore --model <DIR>`.
        #[arg(long = "in-place")]
        in_place: bool,
        /// Merge policy when stacked patches modify the same weight coordinate:
        /// `error` (default), `add` (W0 + sum(dW)), `average` ((A+B)/2),
        /// `weighted` (W0 + sum(alpha_k * dW_k)), `last-wins`, or `first-wins`.
        #[arg(long = "on-conflict", default_value = "error")]
        on_conflict: String,
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
    /// Restore original base weights from an `--in-place` backup.
    Restore {
        /// Model directory previously patched with `apply --in-place`.
        #[arg(long)]
        model: PathBuf,
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
