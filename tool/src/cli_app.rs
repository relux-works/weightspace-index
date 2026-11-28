//! Unified CLI and TUI entrypoint shared by the `weightspace` and `wsp` binaries.
//!
//! Offline authoring/index commands (`check`, `generate`, `replacement`,
//! `derive-lora`, local `apply`, `restore`) never touch the network.
//! Client commands (`tui`, `search`, `info`, `pull`, `models`, `engines`,
//! `run`, `registry`) follow a Hugging Face CLI-style interface with zero
//! authentication required.

use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::sync::atomic::AtomicBool;

use clap::{CommandFactory, Parser, Subcommand};

use crate::download::{PullFormat, download_base_profile, pull_patch};
use crate::engines::{
    DiscoveryRoots, EngineKind, LaunchPlan, best_local_base, detect_engines, discover_local_models,
    matches_base, plan_launch,
};
use crate::error::ToolError;
use crate::files::Sha256Hex;
use crate::generate;
use crate::layout::{check_hygiene, check_lookups, check_models};
use crate::ops::{
    ApplyStack, BuildReplacement, ConflictPolicy, Cx, PatchSpec, Progress, Report, apply_stack,
    build_replacement, restore_backup,
};
use crate::records::Records;
use crate::registry::{AppPaths, RegistryConfig, RegistryKind, load_merged_catalog, sync_registry};
use crate::tui::{TuiExitAction, human_bytes, run_tui};

#[derive(Parser)]
#[command(
    name = "wsp",
    bin_name = "wsp",
    version,
    about = "Weightspace Index CLI & TUI: discover, pull, stack, and run targeted model patches and LoRAs (no auth required)"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Launch the interactive TUI (also the default when running `wsp` in a terminal).
    Tui {
        /// Restrict catalog view to a single configured registry.
        #[arg(long)]
        registry: Option<String>,
    },
    /// Search patches and LoRA adapters across configured registries.
    Search {
        /// Free-text query matched against model, slug, title, summary, and tags.
        query: Option<String>,
        /// Filter by patch kind (`lora` or `replacement`).
        #[arg(long)]
        kind: Option<String>,
        /// Filter by base model substring (e.g. `Qwen3.8-27B`).
        #[arg(long)]
        base: Option<String>,
        /// Restrict search to one registry name.
        #[arg(long)]
        registry: Option<String>,
        /// Output machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show detailed metadata, files, SHA-256 digests, and formats for a patch.
    Info {
        /// Patch address `<org>/<model>/<slug>` (or unique `<slug>`).
        address: String,
        /// Restrict lookup to one registry name.
        #[arg(long)]
        registry: Option<String>,
        /// Output machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Download and SHA-256-verify a patch/LoRA (and optionally its pinned base model).
    Pull {
        /// Patch address `<org>/<model>/<slug>` (or unique `<slug>`).
        address: String,
        /// Also download the pinned base model profile from Hugging Face (no auth).
        #[arg(long = "with-base")]
        with_base: bool,
        /// Artifact format to pull: `auto`, `peft`, `gguf`, `replacement`, or `all`.
        #[arg(long, default_value = "auto")]
        format: String,
        /// Optional destination directory for pulled patch files.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Restrict lookup to one registry name.
        #[arg(long)]
        registry: Option<String>,
    },
    /// Scan local caches (HF, llama.cpp, Ollama, LM Studio, Strata, ds4, weightspace)
    /// and match them against available patches.
    Models {
        /// Prefer format matching a specific engine (`llama.cpp`, `vllm`, `ollama`, etc.).
        #[arg(long)]
        engine: Option<String>,
        /// Restrict catalog matching to one registry name.
        #[arg(long)]
        registry: Option<String>,
        /// Output machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Detect locally installed inference engines and show install hints for missing ones.
    Engines {
        /// Output machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Prepare a patch (pull + apply if replacement) and launch the chosen engine or print paths.
    Run {
        /// Patch address `<org>/<model>/<slug>` (or unique `<slug>`).
        address: String,
        /// Target inference engine (`llama.cpp`, `vllm`, `transformers`, `ollama`, `lm-studio`, `mlx-lm`, `strata`, `ds4`).
        #[arg(long, default_value = "llama.cpp")]
        engine: String,
        /// Explicit local base model path (defaults to auto-detected cache entry).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Automatically download the pinned base model if not found locally.
        #[arg(long = "download-base")]
        download_base: bool,
        /// Apply replacement patches in place (with automatic verified backup) instead of a managed copy.
        #[arg(long = "in-place")]
        in_place: bool,
        /// Conflict merge policy when stacking replacements (`error`, `add`, `average`, `weighted`, `last-wins`, `first-wins`).
        #[arg(long = "on-conflict", default_value = "error")]
        on_conflict: String,
        /// Print absolute paths and launch command instead of spawning the engine.
        #[arg(long = "print-paths")]
        print_paths: bool,
        /// Restrict lookup to one registry name.
        #[arg(long)]
        registry: Option<String>,
    },
    /// Manage index registries (official default + custom Git/GitLab/Gitea/SSH/HTTP/local + pinning).
    Registry {
        #[command(subcommand)]
        action: RegistryCommand,
    },
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
        /// JSON object file mapping full-form tensor names to expected base SHA-256 hex digests.
        #[arg(long = "expect-tensor-sha256-file")]
        expect_tensor_sha256_file: Option<PathBuf>,
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
        /// JSON object file mapping full-form tensor names to expected base SHA-256 hex digests.
        #[arg(long = "expect-tensor-sha256-file")]
        expect_tensor_sha256_file: Option<PathBuf>,
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

#[derive(Subcommand)]
pub enum RegistryCommand {
    /// List configured registries, priority, pin, and default marker.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Add a Git (GitHub/GitLab/Gitea/SSH), static HTTP, or local directory registry.
    Add {
        /// Short alias name (ASCII letters, digits, '-' or '_').
        name: String,
        /// Git URL, static HTTP base URL, or local directory path.
        url: String,
        /// Explicit kind (`git`, `http`, `local`); inferred automatically when omitted.
        #[arg(long)]
        kind: Option<String>,
        /// Pin this registry to a specific commit SHA, tag, or branch.
        #[arg(long)]
        pin: Option<String>,
        /// Resolution priority (higher wins on `(base, slug)` collisions).
        #[arg(long)]
        priority: Option<i32>,
        /// Make this registry the default registry.
        #[arg(long)]
        default: bool,
        /// Immediately sync the newly added registry.
        #[arg(long)]
        sync: bool,
    },
    /// Remove a configured registry by name.
    Remove { name: String },
    /// Pin a registry to a specific commit SHA, tag, or branch.
    Pin { name: String, rev: String },
    /// Remove the pin from a registry so it tracks its upstream default branch.
    Unpin { name: String },
    /// Set the default registry.
    Default { name: String },
    /// Sync one or all enabled registries into the local cache.
    Sync {
        /// Optional single registry name (syncs all enabled registries if omitted).
        name: Option<String>,
    },
}

/// Parses repeated `--expect-tensor-sha256 NAME=HEX` arguments.
pub fn parse_expect(items: &[String]) -> Result<BTreeMap<String, Sha256Hex>, ToolError> {
    parse_expect_with_file(items, None)
}

/// Parses expected base tensor SHA-256 digests from an optional JSON object file
/// (`{ "tensor.name": "<64-hex>" }`) and repeated `NAME=HEX` CLI arguments.
pub fn parse_expect_with_file(
    items: &[String],
    file: Option<&Path>,
) -> Result<BTreeMap<String, Sha256Hex>, ToolError> {
    let invalid = |detail: String| ToolError::InvalidArgument { detail };
    let mut out = BTreeMap::new();
    if let Some(path) = file {
        let raw = std::fs::read(path).map_err(|e| ToolError::io(path, e))?;
        let val: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| {
            invalid(format!(
                "--expect-tensor-sha256-file {}: invalid JSON: {e}",
                path.display()
            ))
        })?;
        let map = val.as_object().ok_or_else(|| {
            invalid(format!(
                "--expect-tensor-sha256-file {}: expected a JSON object mapping tensor names to 64-hex SHA-256 digests",
                path.display()
            ))
        })?;
        for (name, hex_val) in map {
            if name.is_empty() {
                return Err(invalid(format!(
                    "--expect-tensor-sha256-file {}: tensor name must be non-empty",
                    path.display()
                )));
            }
            let hex = hex_val.as_str().ok_or_else(|| {
                invalid(format!(
                    "--expect-tensor-sha256-file {} {name}: expected a 64-hex string, got {hex_val}",
                    path.display()
                ))
            })?;
            let digest = Sha256Hex::parse(hex).map_err(|_| {
                invalid(format!(
                    "--expect-tensor-sha256-file {} {name}: expected 64 hex characters, got {hex:?}",
                    path.display()
                ))
            })?;
            out.insert(name.clone(), digest);
        }
    }
    for item in items {
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

pub fn run() -> ExitCode {
    let cli = Cli::parse();
    let Some(cmd) = cli.command else {
        if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            return finish_client(cmd_tui(None));
        }
        let _ = Cli::command().print_help();
        println!();
        return ExitCode::SUCCESS;
    };

    match cmd {
        Command::Tui { registry } => finish_client(cmd_tui(registry.as_deref())),
        Command::Search {
            query,
            kind,
            base,
            registry,
            json,
        } => finish_client(cmd_search(
            query.as_deref(),
            kind.as_deref(),
            base.as_deref(),
            registry.as_deref(),
            json,
        )),
        Command::Info {
            address,
            registry,
            json,
        } => finish_client(cmd_info(&address, registry.as_deref(), json)),
        Command::Pull {
            address,
            with_base,
            format,
            out,
            registry,
        } => finish_client(cmd_pull(
            &address,
            with_base,
            &format,
            out.as_deref(),
            registry.as_deref(),
        )),
        Command::Models {
            engine,
            registry,
            json,
        } => finish_client(cmd_models(engine.as_deref(), registry.as_deref(), json)),
        Command::Engines { json } => finish_client(cmd_engines(json)),
        Command::Run {
            address,
            engine,
            model,
            download_base,
            in_place,
            on_conflict,
            print_paths,
            registry,
        } => finish_client(cmd_run(
            &address,
            &engine,
            model.as_deref(),
            download_base,
            in_place,
            &on_conflict,
            print_paths,
            registry.as_deref(),
        )),
        Command::Registry { action } => finish_client(cmd_registry(action)),
        Command::Check { repo } => legacy(check(&repo_or_current(repo))),
        Command::Generate { repo } => legacy(regenerate(&repo_or_current(repo))),
        Command::Replacement {
            base,
            edited,
            manifest,
            out,
            temp_dir,
        } => run_op(
            "replacement written",
            &out,
            with_cx(temp_dir, |cx| {
                build_replacement(
                    &BuildReplacement {
                        base,
                        edited,
                        manifest,
                        out: out.clone(),
                    },
                    cx,
                )
            }),
        ),
        Command::DeriveLora {
            model,
            patch,
            rank,
            factor_dtype,
            tensor_shapes,
            method,
            out,
            expect_tensor_sha256,
            expect_tensor_sha256_file,
            temp_dir,
        } => {
            debug_assert_eq!(factor_dtype, "f32");
            let result =
                parse_expect_with_file(&expect_tensor_sha256, expect_tensor_sha256_file.as_deref())
                    .and_then(|expected| {
                        with_cx(temp_dir, |cx| {
                            use crate::derive_lora::*;
                            if method == "selector-exact" {
                                if rank.is_some() {
                                    return Err(ToolError::InvalidArgument {
                                        detail:
                                            "selector-exact chooses rank per tensor; omit --rank"
                                                .into(),
                                    });
                                }
                                derive_exact_lora(
                                    &DeriveExactLora {
                                        model,
                                        tensor_shapes,
                                        patch,
                                        out: out.clone(),
                                        expect_tensor_sha256: expected,
                                    },
                                    cx,
                                )
                            } else {
                                let model = model.ok_or_else(|| ToolError::InvalidArgument {
                                    detail: "explicit-rank methods require --model".into(),
                                })?;
                                let rank = rank.ok_or_else(|| ToolError::InvalidArgument {
                                    detail: "explicit-rank methods require --rank".into(),
                                })?;
                                derive_lora(
                                    &DeriveLora {
                                        model,
                                        patch,
                                        rank,
                                        out: out.clone(),
                                        expect_tensor_sha256: expected,
                                        method: if method == "svd" {
                                            Method::Svd
                                        } else {
                                            Method::Selector
                                        },
                                    },
                                    cx,
                                )
                            }
                        })
                    });
            println!("{}", crate::derive_lora::DISCLAIMER);
            run_op("derived experimental LoRA written", &out, result)
        }
        Command::Apply {
            patch,
            model,
            out,
            in_place,
            on_conflict,
            expect_tensor_sha256,
            expect_tensor_sha256_file,
            temp_dir,
        } => {
            let display_target = out.clone().unwrap_or_else(|| model.clone());
            let result = (|| {
                let expected = parse_expect_with_file(
                    &expect_tensor_sha256,
                    expect_tensor_sha256_file.as_deref(),
                )?;
                let policy = ConflictPolicy::parse(&on_conflict)?;
                let specs: Result<Vec<PatchSpec>, ToolError> =
                    patch.iter().map(|p| PatchSpec::parse(p)).collect();
                let specs = specs?;
                with_cx(temp_dir, |cx| {
                    apply_stack(
                        &ApplyStack {
                            patches: specs,
                            model,
                            out,
                            in_place,
                            on_conflict: policy,
                            expect_tensor_sha256: expected,
                        },
                        cx,
                    )
                })
            })();
            run_op("patched model written", &display_target, result)
        }
        Command::Restore { model } => {
            let cancelled = AtomicBool::new(false);
            let mut sink = |_: Progress| {};
            let mut cx = Cx::new(&cancelled, &mut sink);
            match restore_backup(&model, &mut cx) {
                Ok(r) => {
                    println!(
                        "restored original weights: {} ({} bytes, sha256 {})",
                        r.weights_path.display(),
                        r.restored_bytes,
                        r.restored_sha256
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

fn finish_client(res: Result<(), ToolError>) -> ExitCode {
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_tui(registry_filter: Option<&str>) -> Result<(), ToolError> {
    let paths = AppPaths::resolve();
    let cfg = RegistryConfig::load(&paths)?;
    match run_tui(&cfg, &paths, registry_filter)? {
        TuiExitAction::Quit => Ok(()),
        TuiExitAction::PrintPaths(plan) => {
            print_launch_plan(&plan);
            Ok(())
        }
        TuiExitAction::LaunchEngine(plan) => execute_or_print_plan(&plan),
    }
}

fn cmd_search(
    query: Option<&str>,
    kind: Option<&str>,
    base: Option<&str>,
    registry: Option<&str>,
    json: bool,
) -> Result<(), ToolError> {
    let paths = AppPaths::resolve();
    let cfg = RegistryConfig::load(&paths)?;
    let catalog = load_merged_catalog(&cfg, &paths, registry, true)?;
    let matches = catalog.search(query, kind, base, registry);
    if json {
        let out = serde_json::to_string_pretty(&matches)
            .map_err(|e| ToolError::internal(format!("serialize search: {e}")))?;
        println!("{out}");
        return Ok(());
    }
    if matches.is_empty() {
        println!("No patches matched.");
        return Ok(());
    }
    for p in matches {
        let formats = p.format_badges().join("+");
        println!(
            "{:<48}  {:<11}  {:<12}  [{}]  {}",
            p.address(),
            p.kind,
            formats,
            p.registry,
            p.summary
        );
    }
    Ok(())
}

fn cmd_info(address: &str, registry: Option<&str>, json: bool) -> Result<(), ToolError> {
    let paths = AppPaths::resolve();
    let cfg = RegistryConfig::load(&paths)?;
    let catalog = load_merged_catalog(&cfg, &paths, registry, true)?;
    let patch = catalog.resolve_patch(address)?;
    if json {
        let out = serde_json::to_string_pretty(patch)
            .map_err(|e| ToolError::internal(format!("serialize patch: {e}")))?;
        println!("{out}");
        return Ok(());
    }
    println!("Address:   {} (v{})", patch.address(), patch.version);
    println!("Title:     {}", patch.title);
    println!(
        "Kind:      {} ({})",
        patch.kind,
        patch.format_badges().join(", ")
    );
    println!("Registry:  {}", patch.registry);
    println!("License:   {}", patch.license);
    println!("Profile:   {}", patch.target_profile);
    println!("Summary:   {}", patch.summary);
    if let Some(ref h) = patch.highlight {
        println!("Reported:  {h}");
    }
    println!("Files:");
    for f in &patch.files {
        println!(
            "  - {:<36} {:<8} {:>9}  sha256:{}",
            f.name,
            f.role,
            human_bytes(f.size_bytes),
            &f.sha256[..12]
        );
    }
    Ok(())
}

fn cmd_pull(
    address: &str,
    with_base: bool,
    format: &str,
    out: Option<&Path>,
    registry: Option<&str>,
) -> Result<(), ToolError> {
    let fmt = PullFormat::parse(format)?;
    let paths = AppPaths::resolve();
    let cfg = RegistryConfig::load(&paths)?;
    let catalog = load_merged_catalog(&cfg, &paths, registry, true)?;
    let patch = catalog.resolve_patch(address)?;

    if with_base {
        let base_entry = catalog
            .bases
            .get(&patch.base)
            .ok_or_else(|| ToolError::NotFound {
                detail: format!("base metadata missing for {}", patch.base),
            })?;
        let profile = base_entry
            .profiles
            .iter()
            .find(|p| p.id == patch.target_profile)
            .or_else(|| base_entry.profiles.first())
            .ok_or_else(|| ToolError::NotFound {
                detail: format!("no base profile for {}", patch.base),
            })?;
        eprintln!(
            "Downloading base {} ({} profile, {} files)...",
            patch.base,
            profile.id,
            profile.files.len()
        );
        let base_dir =
            download_base_profile(&patch.base, profile, &paths, &mut |name, done, total| {
                if done == total && total > 0 {
                    eprintln!("  ✓ {name} ({})", human_bytes(total));
                }
            })?;
        println!("BASE_PATH={}", base_dir.display());
    }

    let pulled = pull_patch(
        patch,
        &paths,
        out,
        fmt,
        fmt == PullFormat::Gguf,
        &mut |name, done, total| {
            if done == total && total > 0 {
                eprintln!("  ✓ {name} ({})", human_bytes(total));
            }
        },
    )?;
    println!("PATCH_DIR={}", pulled.dir.display());
    if let Some(ref p) = pulled.replacement_file {
        println!("REPLACEMENT_FILE={}", p.display());
    }
    if let Some(ref p) = pulled.gguf_adapter {
        println!("GGUF_ADAPTER={}", p.display());
    }
    if let Some(ref p) = pulled.peft_adapter {
        println!("PEFT_ADAPTER={}", p.display());
    }
    Ok(())
}

fn cmd_models(engine: Option<&str>, registry: Option<&str>, json: bool) -> Result<(), ToolError> {
    let paths = AppPaths::resolve();
    let cfg = RegistryConfig::load(&paths)?;
    let catalog = load_merged_catalog(&cfg, &paths, registry, false)?;
    let roots = DiscoveryRoots::from_env(&paths);
    let locals = discover_local_models(&roots);
    let eng_filter = engine.map(EngineKind::parse).transpose()?;

    if json {
        let out = serde_json::to_string_pretty(&locals)
            .map_err(|e| ToolError::internal(format!("serialize models: {e}")))?;
        println!("{out}");
        return Ok(());
    }

    if locals.is_empty() {
        println!(
            "No local models discovered in default caches (HF, llama.cpp, Ollama, LM Studio, Strata, ds4, weightspace)."
        );
        println!(
            "Run `wsp search` to browse available patches or `wsp pull <addr> --with-base` to download a pinned base."
        );
        return Ok(());
    }

    for m in &locals {
        if let Some(eng) = eng_filter
            && !eng.prefers_gguf()
            && m.format != crate::engines::ModelWeightFormat::Safetensors
        {
            continue;
        }
        let matching_patches: Vec<&str> = catalog
            .patches
            .iter()
            .filter(|p| matches_base(m, &p.base))
            .map(|p| p.slug.as_str())
            .collect();
        let patch_note = if matching_patches.is_empty() {
            "no indexed patches".to_owned()
        } else {
            format!("patches: {}", matching_patches.join(", "))
        };
        println!(
            "{:<34}  {:<11}  {:<11}  {}  ({})",
            m.model_id,
            m.source.as_str(),
            format!("{:?}", m.format).to_ascii_lowercase(),
            m.path.display(),
            patch_note
        );
    }
    Ok(())
}

fn cmd_engines(json: bool) -> Result<(), ToolError> {
    let engines = detect_engines();
    if json {
        let out = serde_json::to_string_pretty(&engines)
            .map_err(|e| ToolError::internal(format!("serialize engines: {e}")))?;
        println!("{out}");
        return Ok(());
    }
    for e in engines {
        let state = if e.installed {
            "installed"
        } else {
            "missing  "
        };
        let path_or_hint = e
            .binary_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| e.install_hint.to_owned());
        println!("{:<14}  [{state}]  {path_or_hint}", e.id);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    address: &str,
    engine_name: &str,
    explicit_model: Option<&Path>,
    download_base: bool,
    in_place: bool,
    on_conflict: &str,
    print_paths: bool,
    registry: Option<&str>,
) -> Result<(), ToolError> {
    let engine = EngineKind::parse(engine_name)?;
    let policy = ConflictPolicy::parse(on_conflict)?;
    let paths = AppPaths::resolve();
    let cfg = RegistryConfig::load(&paths)?;
    let catalog = load_merged_catalog(&cfg, &paths, registry, true)?;
    let patch = catalog.resolve_patch(address)?;

    let roots = DiscoveryRoots::from_env(&paths);
    let locals = discover_local_models(&roots);

    let mut base_path = explicit_model
        .map(Path::to_path_buf)
        .or_else(|| best_local_base(&locals, &patch.base, engine).map(|m| m.path.clone()));

    if base_path.is_none() && download_base {
        let base_entry = catalog
            .bases
            .get(&patch.base)
            .ok_or_else(|| ToolError::NotFound {
                detail: format!("base metadata missing for {}", patch.base),
            })?;
        let profile = base_entry
            .profiles
            .iter()
            .find(|p| p.id == patch.target_profile)
            .or_else(|| base_entry.profiles.first())
            .ok_or_else(|| ToolError::NotFound {
                detail: format!("no base profile for {}", patch.base),
            })?;
        let downloaded =
            download_base_profile(&patch.base, profile, &paths, &mut |name, done, total| {
                if done == total && total > 0 {
                    eprintln!("  ✓ {name} ({})", human_bytes(total));
                }
            })?;
        base_path = Some(downloaded);
    }

    let pulled = pull_patch(
        patch,
        &paths,
        None,
        PullFormat::Auto,
        engine.prefers_gguf(),
        &mut |name, done, total| {
            if done == total && total > 0 {
                eprintln!("  ✓ {name} ({})", human_bytes(total));
            }
        },
    )?;

    let mut patched_model_path = None;
    if let Some(ref rep_file) = pulled.replacement_file
        && let Some(ref base_dir) = base_path
        && base_dir.is_dir()
    {
        let out_dir = if in_place {
            None
        } else {
            Some(
                paths
                    .patched_models_dir()
                    .join(&patch.base)
                    .join(format!("{}-v{}", patch.slug, patch.version)),
            )
        };
        if let Some(ref existing) = out_dir
            && existing.is_dir()
        {
            patched_model_path = Some(existing.clone());
        } else {
            let cancelled = AtomicBool::new(false);
            let mut sink = |_: Progress| {};
            let mut cx = Cx::new(&cancelled, &mut sink);
            apply_stack(
                &ApplyStack {
                    patches: vec![PatchSpec {
                        path: rep_file.clone(),
                        weight: 1.0,
                    }],
                    model: base_dir.clone(),
                    out: out_dir.clone(),
                    in_place,
                    on_conflict: policy,
                    expect_tensor_sha256: BTreeMap::new(),
                },
                &mut cx,
            )?;
            patched_model_path = out_dir.or_else(|| Some(base_dir.clone()));
        }
    }

    let adapter = if engine.prefers_gguf() {
        pulled
            .gguf_adapter
            .as_deref()
            .or(pulled.peft_adapter.as_deref())
    } else {
        pulled
            .peft_adapter
            .as_deref()
            .or(pulled.gguf_adapter.as_deref())
    };

    let plan = plan_launch(
        engine,
        patch,
        base_path.as_deref(),
        patched_model_path.as_deref(),
        adapter,
    );
    if print_paths {
        print_launch_plan(&plan);
        Ok(())
    } else {
        execute_or_print_plan(&plan)
    }
}

fn print_launch_plan(plan: &LaunchPlan) {
    if let Some(ref b) = plan.base_path {
        println!("BASE_PATH={}", b.display());
    }
    if let Some(ref pm) = plan.patched_model_path {
        println!("PATCHED_MODEL={}", pm.display());
    }
    if let Some(ref ad) = plan.adapter_path {
        println!("ADAPTER_PATH={}", ad.display());
    }
    println!("COMMAND={}", plan.display_command);
    for note in &plan.notes {
        eprintln!("note: {note}");
    }
}

fn execute_or_print_plan(plan: &LaunchPlan) -> Result<(), ToolError> {
    if !plan.engine_installed {
        eprintln!(
            "{} was not found in PATH. Install it via:\n  {}\n",
            plan.engine.label(),
            plan.install_hint
        );
        print_launch_plan(plan);
        return Ok(());
    }
    let Some(ref exe) = plan.executable else {
        print_launch_plan(plan);
        return Ok(());
    };
    eprintln!("Launching: {}", plan.display_command);
    let status = ProcessCommand::new(exe)
        .args(&plan.args)
        .status()
        .map_err(|e| ToolError::Engine {
            engine: plan.engine.id().to_owned(),
            detail: format!("failed to spawn {}: {e}", exe.display()),
        })?;
    if !status.success() {
        return Err(ToolError::Engine {
            engine: plan.engine.id().to_owned(),
            detail: format!("process exited with status {status}"),
        });
    }
    Ok(())
}

fn cmd_registry(action: RegistryCommand) -> Result<(), ToolError> {
    let paths = AppPaths::resolve();
    let mut cfg = RegistryConfig::load(&paths)?;

    match action {
        RegistryCommand::List { json } => {
            if json {
                let out = serde_json::to_string_pretty(&cfg)
                    .map_err(|e| ToolError::internal(format!("serialize registries: {e}")))?;
                println!("{out}");
                return Ok(());
            }
            for r in cfg.ordered_enabled() {
                let def = if r.name == cfg.default { "*" } else { " " };
                let pin = r.pin.as_deref().unwrap_or("HEAD");
                println!(
                    "{def} {:<16}  {:<5}  prio={:<4}  pin={:<12}  {}",
                    r.name,
                    r.kind.as_str(),
                    r.priority,
                    pin,
                    r.url
                );
            }
            Ok(())
        }
        RegistryCommand::Add {
            name,
            url,
            kind,
            pin,
            priority,
            default,
            sync,
        } => {
            let parsed_kind = kind.as_deref().map(RegistryKind::parse).transpose()?;
            cfg.add(name.clone(), url, parsed_kind, pin, priority, default)?;
            cfg.save(&paths)?;
            println!("added registry: {name}");
            if sync && let Some(entry) = cfg.registries.iter().find(|r| r.name == name) {
                let dest = sync_registry(entry, &paths)?;
                println!("synced {name} -> {}", dest.display());
            }
            Ok(())
        }
        RegistryCommand::Remove { name } => {
            cfg.remove(&name)?;
            cfg.save(&paths)?;
            println!("removed registry: {name}");
            Ok(())
        }
        RegistryCommand::Pin { name, rev } => {
            cfg.pin(&name, Some(rev.clone()))?;
            cfg.save(&paths)?;
            println!("pinned registry {name} to {rev}");
            Ok(())
        }
        RegistryCommand::Unpin { name } => {
            cfg.pin(&name, None)?;
            cfg.save(&paths)?;
            println!("unpinned registry {name}");
            Ok(())
        }
        RegistryCommand::Default { name } => {
            cfg.set_default(&name)?;
            cfg.save(&paths)?;
            println!("default registry set to: {name}");
            Ok(())
        }
        RegistryCommand::Sync { name } => {
            let targets: Vec<_> = cfg
                .ordered_enabled()
                .into_iter()
                .filter(|r| name.as_deref().is_none_or(|n| r.name == n))
                .cloned()
                .collect();
            if targets.is_empty() {
                return Err(ToolError::NotFound {
                    detail: format!(
                        "registry {:?} is not configured",
                        name.as_deref().unwrap_or("")
                    ),
                });
            }
            for entry in &targets {
                let dest = sync_registry(entry, &paths)?;
                println!("synced {} -> {}", entry.name, dest.display());
            }
            Ok(())
        }
    }
}

fn repo_or_current(repo: Option<PathBuf>) -> PathBuf {
    repo.unwrap_or_else(|| PathBuf::from("."))
}

fn legacy(result: Result<(), Vec<String>>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(problems) => {
            for p in &problems {
                eprintln!("{p}");
            }
            ExitCode::FAILURE
        }
    }
}

fn with_cx(
    temp_dir: Option<PathBuf>,
    f: impl FnOnce(&mut Cx) -> Result<Report, ToolError>,
) -> Result<Report, ToolError> {
    let cancelled = AtomicBool::new(false);
    let mut sink = |_: Progress| {};
    let mut cx = match temp_dir {
        Some(dir) => Cx::new_with_temp_dir(&cancelled, &mut sink, dir),
        None => Cx::new(&cancelled, &mut sink),
    };
    f(&mut cx)
}

fn run_op(label: &str, out: &Path, result: Result<Report, ToolError>) -> ExitCode {
    match result {
        Ok(report) => {
            println!(
                "{label}: {} ({} tensors, {} entries, {} bytes, sha256 {})",
                out.display(),
                report.tensors_changed,
                report.entries_changed,
                report.out_bytes,
                report.out_sha256
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn trees(repo: &Path) -> [PathBuf; 2] {
    [repo.to_path_buf(), repo.join("docs/example")]
}

fn models(records: &Records, tree: &Path) -> Vec<String> {
    let dir = tree.join("models");
    if dir.exists() {
        check_models(records, &dir)
    } else {
        Vec::new()
    }
}

fn check(repo: &Path) -> Result<(), Vec<String>> {
    let records = Records::load(&repo.join("schema/records.schema.json")).map_err(|e| vec![e])?;
    let mut problems = check_hygiene(repo);
    for tree in trees(repo) {
        let mut found = models(&records, &tree);
        found.extend(check_lookups(&records, &tree));
        if found.is_empty() {
            match generate::build(&tree) {
                Ok(output) => found.extend(generate::diff(&tree, &output)),
                Err(e) => found.push(e),
            }
        }
        let shown = tree.display().to_string();
        problems.extend(found.into_iter().map(|p| format!("{shown}: {p}")));
    }
    if problems.is_empty() {
        println!("checks passed");
        Ok(())
    } else {
        problems.push("from tool/, run `cargo run --locked -- generate ..` if only generated files are reported".into());
        Err(problems)
    }
}

fn regenerate(repo: &Path) -> Result<(), Vec<String>> {
    let records = Records::load(&repo.join("schema/records.schema.json")).map_err(|e| vec![e])?;
    for tree in trees(repo) {
        let errors = models(&records, &tree);
        if !errors.is_empty() {
            return Err(errors);
        }
        let output = generate::build(&tree).map_err(|e| vec![e])?;
        generate::write(&tree, &output).map_err(|e| vec![e])?;
        println!("{}: {} generated files", tree.display(), output.len());
    }
    Ok(())
}
