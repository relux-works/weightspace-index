//! The generator: shipped generated files match their sources, generated
//! files satisfy the schema, and drift is reported and repaired.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::*;
use serde_json::json;
use weightspace_tools::generate::{build, diff, write};

/// A copy of the example tree that a test may change.
fn example_copy() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let tree = dir.path().join("docs/example");
    copy_dir(&repo().join("docs/example"), &tree);
    (dir, tree)
}

fn base_dir(tree: &Path) -> PathBuf {
    tree.join("models").join(BASE)
}

fn index_of(tree: &Path) -> serde_json::Value {
    let out = build(tree).unwrap();
    serde_json::from_slice(&out[&format!("models/{BASE}/index.json")]).unwrap()
}

fn readme_of(tree: &Path) -> String {
    let out = build(tree).unwrap();
    String::from_utf8(out[&format!("models/{BASE}/README.md")].clone()).unwrap()
}

// Both shipped trees match a fresh deterministic generation.
#[test]
fn shipped_trees_are_in_sync() {
    for tree in [repo(), repo().join("docs/example")] {
        let problems = diff(&tree, &build(&tree).unwrap());
        assert!(
            problems.is_empty(),
            "{}: {problems:#?}\nfrom tool/, run `cargo run --locked -- generate ..`",
            tree.display()
        );
    }
}

// Every generated JSON type in the example satisfies its mapped definition.
#[test]
fn generated_files_satisfy_the_schema() {
    let tree = repo().join("docs/example");
    let out = build(&tree).unwrap();
    let mut seen = Vec::new();
    for (rel, bytes) in &out {
        let def = match rel.as_str() {
            "models/index.json" => "modelsIndex",
            r if r.starts_with("lookup/by-digest/") => "digestLookup",
            r if r.starts_with("lookup/by-repo/") => "repoLookup",
            r if r.ends_with("/index.json") => "index",
            r if r.ends_with("/README.md") => continue,
            r => panic!("unexpected generated path {r}"),
        };
        let value = serde_json::from_slice(bytes).unwrap();
        records()
            .validate(def, &value)
            .unwrap_or_else(|e| panic!("{rel}: {e}"));
        seen.push(def);
    }
    seen.sort();
    seen.dedup();
    assert_eq!(seen, ["digestLookup", "index", "modelsIndex", "repoLookup"]);
}

// Weight digests produce lookups; config digests do not.
#[test]
fn lookups_cover_weight_files_only() {
    let out = build(&repo().join("docs/example")).unwrap();
    let digests: Vec<&String> = out
        .keys()
        .filter(|k| k.starts_with("lookup/by-digest/"))
        .collect();
    // model.safetensors (a…) and the GGUF (c…); config.json (b…) is not a weight file.
    assert_eq!(digests.len(), 2, "{digests:?}");
    assert!(digests.iter().all(|k| !k.contains("/bb/")));
    let repo_lookup: serde_json::Value =
        serde_json::from_slice(&out["lookup/by-repo/example-quants/Example-Model-1B-GGUF.json"])
            .unwrap();
    assert_eq!(
        repo_lookup["matches"],
        json!([{"base": BASE, "profiles": ["example-quants-q4-k-m"]}])
    );
}

#[test]
fn index_resolves_hosted_paths_from_its_own_folder() {
    let index = index_of(&repo().join("docs/example"));
    let file = &index["patches"][0]["versions"][0]["files"][0];
    assert_eq!(file["format"], "safetensors");
    assert_eq!(
        file["locators"][0]["path"],
        "better-math/v1/files/replacements.safetensors"
    );
    let base = base_dir(&repo().join("docs/example"));
    assert!(
        base.join(file["locators"][0]["path"].as_str().unwrap())
            .is_file()
    );
    let derived = &index["patches"][0]["versions"][0]["files"][1];
    assert_eq!(derived["format"], "gguf");
    assert!(derived.get("tool").is_none());
}

#[test]
fn latest_is_the_highest_active_version() {
    let (_dir, tree) = example_copy();
    assert_eq!(index_of(&tree)["patches"][1]["latest"], 2);

    let patch = base_dir(&tree).join("shorter-answers/patch.json");
    edit(&patch, |v| v["versions"][1]["status"] = json!("withdrawn"));
    let index = index_of(&tree);
    assert!(index["patches"][1].get("latest").is_none());
    assert!(readme_of(&tree).contains("| lora | none active |"));

    edit(&patch, |v| {
        v["versions"][0]["status"] = json!("active");
        v["versions"][1]["status"] = json!("deprecated");
    });
    assert_eq!(index_of(&tree)["patches"][1]["latest"], 1);
}

#[test]
fn readme_cells_stay_in_their_cell() {
    let (_dir, tree) = example_copy();
    edit(&base_dir(&tree).join("better-math/patch.json"), |v| {
        v["summary"] = json!("a | b\nc")
    });
    assert!(readme_of(&tree).contains("| a \\| b c |"));
}

#[test]
fn readme_reports_refused_and_unmeasured_versions() {
    let (_dir, tree) = example_copy();
    let readme = readme_of(&tree);
    assert!(readme.contains("| shorter-answers v2 | `bf16` | not measured |"));
    assert!(readme.contains("| better-math v1 | `bf16` | accuracy | 0.5 fraction (n=200) |"));

    let comparison = base_dir(&tree).join("better-math/v1/comparisons/bf16-accuracy.json");
    edit(&comparison, |v| {
        for key in ["denominator", "matched_outcomes", "uncertainty"] {
            remove(v, key);
        }
        v["state"] = json!("REFUSED");
        v["reason"] = json!("Different prompts.");
    });
    assert!(readme_of(&tree).contains("refused: Different prompts."));
}

// Missing and stale outputs are detected, then repaired without leaving stale lookup folders.
#[test]
fn drift_is_reported_and_repaired() {
    let (_dir, tree) = example_copy();
    edit(&base_dir(&tree).join("shorter-answers/patch.json"), |v| {
        v["versions"][0]["status"] = json!("withdrawn")
    });
    let stale = tree
        .join("lookup/by-digest/99")
        .join(format!("{}.json", "9".repeat(64)));
    fs::create_dir_all(stale.parent().unwrap()).unwrap();
    fs::write(&stale, "{}").unwrap();
    fs::remove_file(tree.join("models/index.json")).unwrap();

    let out = build(&tree).unwrap();
    let mut problems = diff(&tree, &out);
    problems.sort();
    assert_eq!(
        problems,
        [
            format!(
                "lookup/by-digest/99/{}.json: no longer generated",
                "9".repeat(64)
            ),
            format!("models/{BASE}/index.json: out of date"),
            "models/index.json: missing".to_string(),
        ]
    );

    write(&tree, &out).unwrap();
    assert!(diff(&tree, &out).is_empty());
    assert!(!tree.join("lookup/by-digest/99").exists());
}

// An empty tree generates an empty catalog and a table without rows.
#[test]
fn an_empty_tree_lists_no_bases() {
    let dir = tempfile::tempdir().unwrap();
    let out = build(dir.path()).unwrap();
    assert_eq!(out.len(), 2);
    let bases: serde_json::Value = serde_json::from_slice(&out["models/index.json"]).unwrap();
    assert_eq!(bases, json!({"contract_version": 1, "bases": []}));
}

// Every base gets a relative link and a count of patch names, including withdrawn patches.
#[test]
fn catalog_covers_all_bases_and_patch_statuses() {
    let (_dir, tree) = example_copy();
    let second = tree.join("models/another-org/Empty-Model");
    fs::create_dir_all(&second).unwrap();
    fs::copy(base_dir(&tree).join("base.json"), second.join("base.json")).unwrap();
    edit(&base_dir(&tree).join("shorter-answers/patch.json"), |v| {
        for version in v["versions"].as_array_mut().unwrap() {
            version["status"] = json!("withdrawn");
        }
    });
    let out = build(&tree).unwrap();
    let catalog: serde_json::Value = serde_json::from_slice(&out["models/index.json"]).unwrap();
    assert_eq!(
        catalog,
        json!({"contract_version": 1, "bases": ["another-org/Empty-Model", BASE]})
    );
    let readme = String::from_utf8(out["models/README.md"].clone()).unwrap();
    assert!(
        readme.contains("| [another-org/Empty-Model](another-org/Empty-Model/README.md) | 0 |")
    );
    assert!(readme.contains(&format!("| [{BASE}]({BASE}/README.md) | 2 |")));
    write(&tree, &out).unwrap();
    for address in catalog["bases"].as_array().unwrap() {
        assert!(
            tree.join("models")
                .join(address.as_str().unwrap())
                .join("README.md")
                .is_file()
        );
    }
}

// Catalog table drift and unrecognized lookup files cannot pass; regeneration repairs both.
#[test]
fn catalog_and_lookup_drift_are_repaired() {
    let (_dir, tree) = example_copy();
    fs::write(tree.join("models/README.md"), "wrong count").unwrap();
    fs::write(tree.join("lookup/extra.json"), "{}").unwrap();
    let out = build(&tree).unwrap();
    let problems = diff(&tree, &out);
    assert!(problems.contains(&"models/README.md: out of date".to_owned()));
    assert!(problems.contains(&"lookup/extra.json: no longer generated".to_owned()));
    write(&tree, &out).unwrap();
    assert!(!tree.join("lookup/extra.json").exists());
    assert!(diff(&tree, &out).is_empty());
}
