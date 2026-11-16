//! Layout checks on the example tree, the repository, and broken copies of
//! the example that must fail.

mod common;

use std::fs;
use std::path::Path;

use common::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use weightspace_tools::layout::{check_hygiene, check_lookups, check_models};

#[test]
fn example_tree() {
    let errors = check_models(records(), &example_models());
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn repository_tree() {
    let records = records();
    let mut errors = check_lookups(records, &repo());
    errors.extend(check_lookups(records, &repo().join("examples")));
    if repo().join("models").exists() {
        errors.extend(check_models(records, &repo().join("models")));
    }
    errors.extend(check_hygiene(&repo()));
    assert!(errors.is_empty(), "{errors:#?}");
}

fn safetensors(entries: &[(&str, &str, u64, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, len, bytes) in entries {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            json!({"dtype": dtype, "shape": [len], "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    out
}

fn i64s(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

const W: &str = "model.layers.0.mlp.down_proj.weight";

fn sparse(idx: &[i64], with_old: bool) -> Vec<u8> {
    let n = idx.len() as u64;
    let (i, new, old) = (format!("{W}#idx"), format!("{W}#new"), format!("{W}#old"));
    let mut entries = vec![
        (i.as_str(), "I64", n, i64s(idx)),
        (new.as_str(), "BF16", n, vec![0; 2 * idx.len()]),
    ];
    if with_old {
        entries.push((old.as_str(), "BF16", n, vec![0; 2 * idx.len()]));
    }
    safetensors(&entries)
}

/// Replaces better-math v1's hosted file and updates its record to match.
fn rehost(root: &Path, bytes: Vec<u8>) {
    let dir = root.join(BASE).join("better-math/v1");
    fs::write(dir.join("files/replacements.safetensors"), &bytes).unwrap();
    let sum: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    edit(&dir.join("version.json"), |v| {
        v["files"][0]["sha256"] = json!(sum);
        v["files"][0]["size_bytes"] = json!(bytes.len());
        v["files"][1]["derived_from"] = json!([sum]);
    });
}

#[test]
fn layout_controls() {
    type Edit = fn(&Path);
    let cases: Vec<(&str, bool, Edit)> = vec![
        ("unchanged", true, |_| {}),
        ("rebuilt-valid-sparse", true, |r| {
            rehost(r, sparse(&[1, 5, 9], true))
        }),
        ("withdrawn-files-removed", true, |r| {
            edit(&r.join(BASE).join("better-math/patch.json"), |v| {
                v["versions"][0]["status"] = json!("withdrawn")
            });
            fs::remove_dir_all(r.join(BASE).join("better-math/v1/files")).unwrap();
        }),
        ("version-folder-missing", false, |r| {
            fs::remove_dir_all(r.join(BASE).join("shorter-answers/v1")).unwrap()
        }),
        ("version-folder-unlisted", false, |r| {
            copy_dir(
                &r.join(BASE).join("shorter-answers/v2"),
                &r.join(BASE).join("shorter-answers/v3"),
            )
        }),
        ("version-gap", false, |r| {
            edit(&r.join(BASE).join("shorter-answers/patch.json"), |v| {
                v["versions"][1]["version"] = json!(3)
            })
        }),
        ("bad-slug-folder", false, |r| {
            fs::rename(
                r.join(BASE).join("better-math"),
                r.join(BASE).join("Better_Math"),
            )
            .unwrap()
        }),
        ("unknown-file-in-base", false, |r| {
            fs::write(r.join(BASE).join("notes.txt"), "x").unwrap()
        }),
        ("unknown-json-in-version", false, |r| {
            fs::write(r.join(BASE).join("better-math/v1/extra.json"), "{}").unwrap()
        }),
        ("missing-base-json", false, |r| {
            fs::remove_file(r.join(BASE).join("base.json")).unwrap()
        }),
        ("duplicate-profile", false, |r| {
            edit(&r.join(BASE).join("base.json"), |v| {
                v["profiles"][1]["id"] = json!("bf16")
            })
        }),
        ("unknown-target-profile", false, |r| {
            edit(&r.join(BASE).join("better-math/v1/version.json"), |v| {
                v["target_profile"] = json!("fp8")
            })
        }),
        ("run-unknown-profile", false, |r| {
            edit(
                &r.join(BASE)
                    .join("better-math/v1/runs/patched-bf16/run.json"),
                |v| v["profile"] = json!("fp8"),
            )
        }),
        ("hosted-bytes-changed", false, |r| {
            let f = r
                .join(BASE)
                .join("better-math/v1/files/replacements.safetensors");
            let mut b = fs::read(&f).unwrap();
            *b.last_mut().unwrap() ^= 1;
            fs::write(f, b).unwrap();
        }),
        ("hosted-file-missing", false, |r| {
            fs::remove_file(
                r.join(BASE)
                    .join("better-math/v1/files/replacements.safetensors"),
            )
            .unwrap()
        }),
        ("hosted-file-unlisted", false, |r| {
            fs::write(
                r.join(BASE).join("better-math/v1/files/extra.safetensors"),
                "x",
            )
            .unwrap()
        }),
        ("hosted-outside-files", false, |r| {
            edit(&r.join(BASE).join("better-math/v1/version.json"), |v| {
                v["files"][0]["locators"][0]["path"] = json!("replacements.safetensors")
            })
        }),
        ("duplicate-file-name", false, |r| {
            edit(&r.join(BASE).join("shorter-answers/v2/version.json"), |v| {
                v["files"][1]["name"] = json!("adapter_model.safetensors")
            })
        }),
        ("derived-from-foreign-digest", false, |r| {
            edit(&r.join(BASE).join("better-math/v1/version.json"), |v| {
                v["files"][1]["derived_from"] = json!(["9".repeat(64)])
            })
        }),
        ("replacement-gguf-source", false, |r| {
            edit(&r.join(BASE).join("better-math/v1/version.json"), |v| {
                v["files"][0]["name"] = json!("replacements.gguf")
            })
        }),
        ("lora-without-config", false, |r| {
            edit(&r.join(BASE).join("shorter-answers/v2/version.json"), |v| {
                v["files"][1]["name"] = json!("config.json")
            })
        }),
        ("comparison-unknown-run", false, |r| {
            edit(
                &r.join(BASE)
                    .join("better-math/v1/comparisons/bf16-accuracy.json"),
                |v| v["patched_run"] = json!("nope"),
            )
        }),
        ("comparison-unknown-baseline", false, |r| {
            edit(
                &r.join(BASE)
                    .join("better-math/v1/comparisons/bf16-accuracy.json"),
                |v| v["baseline_run"] = json!("nope"),
            )
        }),
        ("assessment-unknown-comparison", false, |r| {
            let cmp = read(
                &r.join(BASE)
                    .join("better-math/v1/comparisons/bf16-accuracy.json"),
            );
            edit(
                &r.join(BASE)
                    .join("better-math/v1/assessments/arithmetic-help.json"),
                |v| {
                    v["decision"] = json!("SUPPORTED");
                    remove(v, "owner");
                    v["policy"] = cmp["matched_outcomes"].clone();
                    v["comparisons"] = json!(["nope"]);
                },
            );
        }),
        ("sparse-without-old", false, |r| {
            rehost(r, sparse(&[3, 17], false))
        }),
        ("sparse-unsorted", false, |r| {
            rehost(r, sparse(&[17, 3], true))
        }),
        ("sparse-duplicate-index", false, |r| {
            rehost(r, sparse(&[3, 3], true))
        }),
        ("sparse-negative-index", false, |r| {
            rehost(r, sparse(&[-1, 3], true))
        }),
        ("sparse-unknown-suffix", false, |r| {
            rehost(r, safetensors(&[("w#delta", "BF16", 1, vec![0; 2])]))
        }),
        ("full-and-sparse", false, |r| {
            let (i, n, o) = (format!("{W}#idx"), format!("{W}#new"), format!("{W}#old"));
            rehost(
                r,
                safetensors(&[
                    (W, "BF16", 1, vec![0; 2]),
                    (&i, "I64", 1, i64s(&[0])),
                    (&n, "BF16", 1, vec![0; 2]),
                    (&o, "BF16", 1, vec![0; 2]),
                ]),
            )
        }),
        ("not-safetensors", false, |r| {
            rehost(r, b"not a tensor file".to_vec())
        }),
    ];
    let records = records();
    for (name, ok, edit_tree) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("models");
        copy_dir(&example_models(), &root);
        edit_tree(&root);
        let errors = check_models(records, &root);
        assert_eq!(errors.is_empty(), ok, "{name}: {errors:#?}");
    }
}

#[cfg(unix)]
#[test]
fn hygiene_controls() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("file"), "x").unwrap();
    assert!(check_hygiene(tmp.path()).is_empty());
    std::os::unix::fs::symlink("file", tmp.path().join("link")).unwrap();
    assert_eq!(check_hygiene(tmp.path()).len(), 1);
}

#[test]
fn lookup_controls() {
    let records = records();
    let sum = "ab".repeat(32);
    let good = r#"{"contract_version": 1, "matches": [{"base": "a/b", "profile": "bf16"}]}"#;
    let repo = r#"{"contract_version": 1, "matches": [{"base": "a/b", "profiles": ["bf16"]}]}"#;
    let cases = [
        (format!("by-digest/ab/{sum}.json"), good, true),
        (format!("by-digest/cd/{sum}.json"), good, false),
        (format!("by-digest/{sum}.json"), good, false),
        (
            format!("by-digest/ab/{sum}.json"),
            r#"{"contract_version": 1, "matches": []}"#,
            false,
        ),
        ("by-repo/a/b.json".to_owned(), repo, true),
        ("by-repo/a.json".to_owned(), repo, false),
    ];
    for (path, body, ok) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join(&path);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, body).unwrap();
        let errors = check_lookups(records, tmp.path());
        assert_eq!(errors.is_empty(), ok, "{path}: {errors:#?}");
    }
}
