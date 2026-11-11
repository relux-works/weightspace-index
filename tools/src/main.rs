//! `weightspace check [repo]` runs every offline check and fails when a
//! generated file is missing, out of date or stale.
//! `weightspace generate [repo]` rewrites the generated files.
//!
//! Both act on the repository root and on `examples/`, each a tree with
//! its own `models/` and generated files.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use weightspace_tools::generate;
use weightspace_tools::layout::{check_hygiene, check_lookups, check_models};
use weightspace_tools::records::Records;

const USAGE: &str = "usage: weightspace <check|generate> [repo]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = PathBuf::from(args.get(1).map_or(".", String::as_str));
    let result = match (args.first().map(String::as_str), args.len()) {
        (Some("check"), 1 | 2) => check(&repo),
        (Some("generate"), 1 | 2) => regenerate(&repo),
        _ => Err(vec![USAGE.into()]),
    };
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

fn trees(repo: &Path) -> [PathBuf; 2] {
    [repo.to_path_buf(), repo.join("examples")]
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
        problems.push("run `cargo run -- generate` if only generated files are reported".into());
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
