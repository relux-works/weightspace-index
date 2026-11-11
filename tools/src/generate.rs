//! The generator: builds every generated file from the authored files of a
//! tree (a folder holding `models/`). It reads only the repository, so the
//! output is the same on every machine and never depends on a remote source.
//!
//! Generated paths, relative to the tree:
//! - `bases.json`
//! - `by-digest/<aa>/<sha256>.json` for each weight file of a profile
//! - `by-repo/<namespace>/<repo>.json` for each repository holding a profile
//! - `models/<namespace>/<model>/index.json` and `README.md`

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::{Value, json};

use crate::layout::{items, rel_parts, text, walk_files};
use crate::records::read_json;

/// Generated files by path relative to the tree, with their exact bytes.
pub type Output = BTreeMap<String, Vec<u8>>;

/// Builds every generated file. Run the layout checks first: the generator
/// assumes valid records and reports only what stops it from reading them.
pub fn build(tree: &Path) -> Result<Output, String> {
    let models = tree.join("models");
    let mut out = Output::new();
    let mut bases = Vec::new();
    let mut digests: BTreeMap<String, BTreeSet<(String, String)>> = BTreeMap::new();
    let mut repos: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();

    for ns in subdirs(&models)? {
        for model in subdirs(&models.join(&ns))? {
            let address = format!("{ns}/{model}");
            let base = Base::read(&models.join(&ns).join(&model), &address)?;
            for profile in items(&base.record, "profiles") {
                let id = text(profile, "id").to_owned();
                let repo = text(&profile["source"], "repo_id").to_owned();
                repos
                    .entry(repo)
                    .or_default()
                    .entry(address.clone())
                    .or_default()
                    .insert(id.clone());
                for file in items(profile, "files") {
                    if is_weights(text(file, "path")) {
                        let sum = text(file, "sha256").to_owned();
                        digests
                            .entry(sum)
                            .or_default()
                            .insert((address.clone(), id.clone()));
                    }
                }
            }
            out.insert(
                format!("models/{address}/index.json"),
                pretty(&base.index()),
            );
            out.insert(
                format!("models/{address}/README.md"),
                base.readme().into_bytes(),
            );
            bases.push(address);
        }
    }

    out.insert(
        "bases.json".into(),
        pretty(&json!({"contract_version": 1, "bases": bases})),
    );
    for (sum, matches) in digests {
        let matches: Vec<Value> = matches
            .iter()
            .map(|(b, p)| json!({"base": b, "profile": p}))
            .collect();
        out.insert(
            format!("by-digest/{}/{sum}.json", &sum[..2]),
            pretty(&json!({"contract_version": 1, "matches": matches})),
        );
    }
    for (repo, matches) in repos {
        let matches: Vec<Value> = matches
            .iter()
            .map(|(b, p)| json!({"base": b, "profiles": p}))
            .collect();
        out.insert(
            format!("by-repo/{repo}.json"),
            pretty(&json!({"contract_version": 1, "matches": matches})),
        );
    }
    Ok(out)
}

/// Compares generated files with the tree: each must exist with the same
/// bytes, and no other file may sit at a generated path.
pub fn diff(tree: &Path, output: &Output) -> Vec<String> {
    let mut problems = Vec::new();
    for (rel, bytes) in output {
        match fs::read(tree.join(rel)) {
            Ok(disk) if disk == *bytes => {}
            Ok(_) => problems.push(format!("{rel}: out of date")),
            Err(_) => problems.push(format!("{rel}: missing")),
        }
    }
    for rel in existing(tree) {
        if !output.contains_key(&rel) {
            problems.push(format!("{rel}: no longer generated"));
        }
    }
    problems
}

/// Writes generated files and removes ones no longer generated.
pub fn write(tree: &Path, output: &Output) -> Result<(), String> {
    for rel in existing(tree) {
        if !output.contains_key(&rel) {
            fs::remove_file(tree.join(&rel)).map_err(|e| format!("{rel}: {e}"))?;
        }
    }
    for dir in ["by-digest", "by-repo"] {
        remove_empty_dirs(&tree.join(dir));
    }
    for (rel, bytes) in output {
        let path = tree.join(rel);
        if fs::read(&path).is_ok_and(|disk| disk == *bytes) {
            continue;
        }
        fs::create_dir_all(path.parent().unwrap()).map_err(|e| format!("{rel}: {e}"))?;
        fs::write(&path, bytes).map_err(|e| format!("{rel}: {e}"))?;
    }
    Ok(())
}

/// Files at generated paths that exist in the tree now.
fn existing(tree: &Path) -> Vec<String> {
    let mut found: Vec<String> = ["by-digest", "by-repo"]
        .iter()
        .flat_map(|dir| walk_files(&tree.join(dir)))
        .map(|p| rel_parts(tree, &p).join("/"))
        .collect();
    if tree.join("bases.json").is_file() {
        found.push("bases.json".into());
    }
    let models = tree.join("models");
    for ns in subdirs(&models).unwrap_or_default() {
        for model in subdirs(&models.join(&ns)).unwrap_or_default() {
            for name in ["index.json", "README.md"] {
                if models.join(&ns).join(&model).join(name).is_file() {
                    found.push(format!("models/{ns}/{model}/{name}"));
                }
            }
        }
    }
    found
}

fn remove_empty_dirs(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    let mut empty = true;
    for entry in entries.flatten() {
        let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
        empty &= is_dir && remove_empty_dirs(&entry.path());
    }
    empty && fs::remove_dir(dir).is_ok()
}

/// Sorted names of the subfolders of `dir`; none when `dir` is absent.
fn subdirs(dir: &Path) -> Result<Vec<String>, String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(Vec::new());
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    Ok(names)
}

fn pretty(v: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(v).expect("JSON values serialize");
    bytes.push(b'\n');
    bytes
}

fn is_weights(path: &str) -> bool {
    path.ends_with(".safetensors") || path.ends_with(".gguf")
}

fn format_of(name: &str) -> &'static str {
    match name.rsplit_once('.').map(|(_, ext)| ext) {
        Some("gguf") => "gguf",
        Some("json") => "json",
        _ => "safetensors",
    }
}

/// Records of one folder: `<id>/run.json` files, or `<id>.json` files.
fn read_records(dir: &Path, nested: Option<&str>) -> Result<BTreeMap<String, Value>, String> {
    let mut out = BTreeMap::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(out);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        match nested {
            Some(file) => drop(out.insert(name, read_json(&entry.path().join(file))?)),
            None => {
                if let Some(id) = name.strip_suffix(".json") {
                    out.insert(id.to_owned(), read_json(&entry.path())?);
                }
            }
        }
    }
    Ok(out)
}

struct Base {
    address: String,
    record: Value,
    runs: BTreeMap<String, Value>,
    patches: Vec<Patch>,
}

struct Patch {
    slug: String,
    record: Value,
    versions: Vec<Version>,
}

struct Version {
    number: u64,
    status: String,
    record: Value,
    runs: BTreeMap<String, Value>,
    comparisons: BTreeMap<String, Value>,
    assessments: BTreeMap<String, Value>,
}

impl Base {
    fn read(dir: &Path, address: &str) -> Result<Self, String> {
        let mut patches = Vec::new();
        for slug in subdirs(dir)?.into_iter().filter(|s| s != "runs") {
            let pdir = dir.join(&slug);
            let record = read_json(&pdir.join("patch.json"))?;
            let mut versions = Vec::new();
            for v in items(&record, "versions") {
                let number = v["version"].as_u64().unwrap_or(0);
                let vdir = pdir.join(format!("v{number}"));
                versions.push(Version {
                    number,
                    status: text(v, "status").to_owned(),
                    record: read_json(&vdir.join("version.json"))?,
                    runs: read_records(&vdir.join("runs"), Some("run.json"))?,
                    comparisons: read_records(&vdir.join("comparisons"), None)?,
                    assessments: read_records(&vdir.join("assessments"), None)?,
                });
            }
            patches.push(Patch {
                slug,
                record,
                versions,
            });
        }
        Ok(Self {
            address: address.to_owned(),
            record: read_json(&dir.join("base.json"))?,
            runs: read_records(&dir.join("runs"), Some("run.json"))?,
            patches,
        })
    }

    fn index(&self) -> Value {
        let patches: Vec<Value> = self.patches.iter().map(Patch::index).collect();
        json!({
            "contract_version": 1,
            "base": self.address,
            "profiles": self.record["profiles"],
            "baseline_runs": self.runs.iter().map(|(id, r)| index_run(id, r)).collect::<Vec<_>>(),
            "patches": patches,
        })
    }

    fn readme(&self) -> String {
        let mut md = format!(
            "# {}\n\nGenerated from the files in this folder by `weightspace generate`; do not edit.\n\n## Profiles\n\n| Profile | Source | Files |\n|---|---|---|\n",
            self.address
        );
        for p in items(&self.record, "profiles") {
            let (repo, revision) = (
                text(&p["source"], "repo_id"),
                text(&p["source"], "revision"),
            );
            md += &format!(
                "| `{}` | [{repo}@{}](https://huggingface.co/{repo}/tree/{revision}) | {} |\n",
                text(p, "id"),
                &revision[..revision.len().min(7)],
                items(p, "files").len()
            );
        }

        md += "\n## Patches\n\n";
        if self.patches.is_empty() {
            md += "None yet.\n";
        } else {
            md += "| Patch | Kind | Latest | Summary |\n|---|---|---|---|\n";
            for p in &self.patches {
                let latest = p
                    .latest()
                    .map_or("none active".into(), |v| format!("v{}", v.number));
                md += &format!(
                    "| [{0}]({0}/) | {1} | {latest} | {2} |\n",
                    p.slug,
                    text(&p.record, "kind"),
                    cell(text(&p.record, "summary"))
                );
            }
        }

        let mut results = String::new();
        let mut decisions = String::new();
        for p in &self.patches {
            let Some(v) = p.latest() else { continue };
            let label = format!("{} v{}", p.slug, v.number);
            if v.comparisons.is_empty() {
                results += &format!(
                    "| {label} | `{}` | not measured | | | |\n",
                    text(&v.record, "target_profile")
                );
            }
            for c in v.comparisons.values() {
                let baseline = self.runs.get(text(c, "baseline_run"));
                let patched = v.runs.get(text(c, "patched_run"));
                let profile = baseline.map_or("", |r| text(r, "profile"));
                if text(c, "state") == "REFUSED" {
                    results += &format!(
                        "| {label} | `{profile}` | refused: {} | | | |\n",
                        cell(text(c, "reason"))
                    );
                    continue;
                }
                let score = |r: Option<&Value>| r.map_or("unknown".into(), |r| score(&r["score"]));
                let metric = patched.map_or("", |r| text(&r["score"], "metric"));
                let by = patched.map_or("", |r| text(r, "performed_by"));
                results += &format!(
                    "| {label} | `{profile}` | {} | {} | {} | {by} |\n",
                    cell(metric),
                    score(baseline),
                    score(patched)
                );
            }
            for (id, a) in &v.assessments {
                decisions += &format!(
                    "| {label} | {id} | {} | {} |\n",
                    cell(text(a, "intended_use")),
                    text(a, "decision")
                );
            }
        }
        if !results.is_empty() {
            md += "\n## Results\n\nLatest active version of each patch, against the baseline run on the same profile.\n\n| Patch | Profile | Metric | Baseline | Patched | Measured by |\n|---|---|---|---|---|---|\n";
            md += &results;
        }
        if !decisions.is_empty() {
            md += "\n## Assessments\n\n| Patch | Assessment | Intended use | Decision |\n|---|---|---|---|\n";
            md += &decisions;
        }
        md
    }
}

impl Patch {
    /// The highest-numbered active version.
    fn latest(&self) -> Option<&Version> {
        self.versions
            .iter()
            .filter(|v| v.status == "active")
            .max_by_key(|v| v.number)
    }

    fn index(&self) -> Value {
        let mut out = json!({ "slug": self.slug });
        for key in [
            "title",
            "summary",
            "kind",
            "authors",
            "submitter",
            "license",
            "tags",
        ] {
            out[key] = self.record[key].clone();
        }
        if let Some(v) = self.latest() {
            out["latest"] = json!(v.number);
        }
        out["versions"] = self
            .versions
            .iter()
            .map(|v| v.index(&format!("{}/v{}", self.slug, v.number)))
            .collect();
        out
    }
}

impl Version {
    /// The version's entry in index.json; `folder` is the version folder
    /// relative to the base folder.
    fn index(&self, folder: &str) -> Value {
        let files: Vec<Value> = items(&self.record, "files")
            .iter()
            .map(|f| {
                let mut f = f.clone();
                let format = format_of(text(&f, "name"));
                f.as_object_mut().unwrap().remove("tool");
                f["format"] = json!(format);
                // index-path is relative to the declaring file, which is now
                // index.json in the base folder.
                for l in f["locators"].as_array_mut().into_iter().flatten() {
                    if text(l, "kind") == "index-path" {
                        l["path"] = json!(format!("{folder}/{}", text(l, "path")));
                    }
                }
                f
            })
            .collect();
        let assessments: Vec<Value> = self
            .assessments
            .iter()
            .map(|(id, a)| json!({"id": id, "intended_use": a["intended_use"], "decision": a["decision"]}))
            .collect();
        json!({
            "version": self.number,
            "status": self.status,
            "changelog": self.record["changelog"],
            "target_profile": self.record["target_profile"],
            "files": files,
            "runs": self.runs.iter().map(|(id, r)| index_run(id, r)).collect::<Vec<_>>(),
            "assessments": assessments,
        })
    }
}

fn index_run(id: &str, run: &Value) -> Value {
    let mut out = json!({ "id": id });
    for key in ["profile", "performed_by", "protocol", "state", "score"] {
        out[key] = run[key].clone();
    }
    out
}

fn score(s: &Value) -> String {
    if text(s, "state") == "UNKNOWN" {
        return "unknown".into();
    }
    format!(
        "{} {} (n={})",
        s["value"],
        cell(text(s, "unit")),
        s["denominator"]
    )
}

/// Text safe inside one Markdown table cell.
fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace(['\n', '\r'], " ")
}
