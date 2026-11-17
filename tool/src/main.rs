//! Thin binary: parse, dispatch, print, exit code.
//!
//! `weightspace check [repo]` runs every offline check and fails when a
//! generated file is missing, out of date or stale.
//! `weightspace generate [repo]` rewrites the generated files. Both act
//! on the repository root and on `docs/example/`, each a tree with its own
//! `models/` and generated files, and keep their original success
//! output.
//!
//! `weightspace replacement` and `weightspace apply` call the library
//! operations in `weightspace_tools::ops` with a no-op progress sink.

mod cli;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use clap::Parser;
use weightspace_tools::error::ToolError;
use weightspace_tools::generate;
use weightspace_tools::layout::{check_hygiene, check_lookups, check_models};
use weightspace_tools::ops::{
    ApplyReplacement, BuildReplacement, Cx, Progress, Report, apply_replacement, build_replacement,
};
use weightspace_tools::records::Records;

use cli::{Cli, Command};

fn main() -> ExitCode {
    match Cli::parse().command {
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
            temp_dir,
        } => {
            // Clap admits F32 only; the library also has no alternate storage path.
            debug_assert_eq!(factor_dtype, "f32");
            let result = cli::parse_expect(&expect_tensor_sha256).and_then(|expected| {
                with_cx(temp_dir, |cx| {
                    use weightspace_tools::derive_lora::*;
                    if method == "selector-exact" {
                        if rank.is_some() {
                            return Err(ToolError::InvalidArgument {
                                detail: "selector-exact chooses rank per tensor; omit --rank"
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
            println!("{}", weightspace_tools::derive_lora::DISCLAIMER);
            run_op("derived experimental LoRA written", &out, result)
        }
        Command::Apply {
            patch,
            model,
            out,
            expect_tensor_sha256,
            temp_dir,
        } => {
            let result = cli::parse_expect(&expect_tensor_sha256).and_then(|expected| {
                with_cx(temp_dir, |cx| {
                    apply_replacement(
                        &ApplyReplacement {
                            patch,
                            model,
                            out: out.clone(),
                            expect_tensor_sha256: expected,
                        },
                        cx,
                    )
                })
            });
            run_op("patched model written", &out, result)
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

/// Runs one library operation with the CLI's no-op progress sink.
/// Private input snapshots are made in `temp_dir` when given, else
/// in the system temporary directory.
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

/// Model checks for a tree; a tree with no `models/` has nothing to check.
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
