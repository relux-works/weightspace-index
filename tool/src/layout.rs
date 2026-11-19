//! Layout rules: where files may live, how versions line up with their
//! folders, which records refer to which, and whether files hosted in the
//! repository match their recorded digests.

mod dual;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::patch::check_replacement_file;
use crate::records::{Records, read_json};

static SLUG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*\z").unwrap());
static ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\z").unwrap());
static VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^v[1-9][0-9]*\z").unwrap());
static SHA256: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-f0-9]{64}\z").unwrap());

/// Files uploaded to this repository must be under GitHub's per-file limit.
pub const HOSTED_FILE_LIMIT: u64 = 100_000_000;

type Runs = BTreeMap<String, Option<Value>>;

/// Checks a `models/` tree: `models/<namespace>/<model>/...`.
pub fn check_models(records: &Records, root: &Path) -> Vec<String> {
    let mut l = Layout {
        records,
        errors: Vec::new(),
    };
    for ns in l.read_dir(root) {
        if !ns.is_dir {
            match ns.name.as_str() {
                "index.json" => {
                    l.record(&ns.path, "modelsIndex");
                    continue;
                }
                "README.md" => continue,
                _ => {}
            }
        }
        if !ns.is_dir {
            l.fail(format!(
                "{}: only namespace folders belong here",
                ns.path.display()
            ));
            continue;
        }
        for model in l.read_dir(&ns.path) {
            if model.is_dir {
                l.base(&model.path);
            } else {
                l.fail(format!(
                    "{}: only model folders belong here",
                    model.path.display()
                ));
            }
        }
    }
    l.errors
}

/// Checks the generated `lookup/by-digest/` and `lookup/by-repo/` trees under `root`, when present.
pub fn check_lookups(records: &Records, root: &Path) -> Vec<String> {
    let mut l = Layout {
        records,
        errors: Vec::new(),
    };
    let digests = root.join("lookup/by-digest");
    for path in walk_files(&digests) {
        let parts = rel_parts(&digests, &path);
        let sum = parts
            .last()
            .and_then(|n| n.strip_suffix(".json"))
            .unwrap_or("");
        if parts.len() != 2 || !SHA256.is_match(sum) || parts[0] != sum[..2] {
            l.fail(format!(
                "{}: expected lookup/by-digest/<aa>/<sha256>.json",
                path.display()
            ));
            continue;
        }
        l.record(&path, "digestLookup");
    }
    let repos = root.join("lookup/by-repo");
    for path in walk_files(&repos) {
        let parts = rel_parts(&repos, &path);
        if parts.len() != 2 || !parts[1].ends_with(".json") {
            l.fail(format!(
                "{}: expected lookup/by-repo/<namespace>/<repo>.json",
                path.display()
            ));
            continue;
        }
        l.record(&path, "repoLookup");
    }
    l.errors
}

/// Rules 3 and 10: no symlinks, and no two names in a folder differing only
/// in case. Skips `.git` and the build output folder `target`.
pub fn check_hygiene(root: &Path) -> Vec<String> {
    let mut errors = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            errors.push(format!("{}: cannot read", dir.display()));
            continue;
        };
        let mut seen = BTreeMap::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if let Some(other) = seen.insert(name.to_lowercase(), name.clone()) {
                errors.push(format!(
                    "{}: {other} and {name} differ only in case",
                    dir.display()
                ));
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_symlink() {
                errors.push(format!("{}: symlinks are not allowed", path.display()));
            } else if kind.is_dir() && name != ".git" && name != "target" {
                stack.push(path);
            }
        }
    }
    errors
}

/// Enforces the two source forms of a patch version.
pub fn check_sources(kind: &str, sources: &[String]) -> Result<(), String> {
    let mut sources = sources.to_vec();
    sources.sort();
    match kind {
        "replacement" => {
            if sources.is_empty() {
                return Err("no source file".into());
            }
            if let Some(s) = sources.iter().find(|s| !s.ends_with(".safetensors")) {
                return Err(format!("replacement source {s} is not .safetensors"));
            }
        }
        "lora" => {
            let peft = sources == ["adapter_config.json", "adapter_model.safetensors"];
            let gguf = sources.len() == 1 && sources[0].ends_with(".gguf");
            if !peft && !gguf {
                return Err(format!(
                    "LoRA sources must be adapter_model.safetensors with adapter_config.json, or one .gguf; got {sources:?}"
                ));
            }
        }
        other => return Err(format!("unknown kind {other}")),
    }
    Ok(())
}

struct Entry {
    name: String,
    path: PathBuf,
    is_dir: bool,
}

struct Layout<'a> {
    records: &'a Records,
    errors: Vec<String>,
}

impl Layout<'_> {
    fn fail(&mut self, message: String) {
        self.errors.push(message);
    }

    fn read_dir(&mut self, dir: &Path) -> Vec<Entry> {
        let mut out = Vec::new();
        match fs::read_dir(dir) {
            Err(e) => self.fail(format!("{}: {e}", dir.display())),
            Ok(entries) => {
                for entry in entries.flatten() {
                    out.push(Entry {
                        name: entry.file_name().to_string_lossy().into_owned(),
                        path: entry.path(),
                        // file_type does not follow symlinks, so a symlink is neither.
                        is_dir: entry.file_type().is_ok_and(|t| t.is_dir()),
                    });
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn record(&mut self, path: &Path, def: &str) -> Option<Value> {
        let checked = read_json(path).and_then(|v| {
            self.records
                .validate(def, &v)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(v)
        });
        checked.map_err(|e| self.fail(e)).ok()
    }

    fn base(&mut self, dir: &Path) {
        let mut base = None;
        let mut base_runs = Runs::new();
        let mut slugs = Vec::new();
        for e in self.read_dir(dir) {
            match (e.name.as_str(), e.is_dir) {
                ("base.json", false) => base = self.record(&e.path, "base"),
                ("index.json", false) => drop(self.record(&e.path, "index")),
                ("README.md", false) => {}
                ("runs", true) => base_runs = self.runs(&e.path),
                (name, true) if SLUG.is_match(name) => slugs.push(e.path),
                _ => self.fail(format!("{}: unexpected path", e.path.display())),
            }
        }
        let Some(base) = base else {
            self.fail(format!("{}: missing or invalid base.json", dir.display()));
            return;
        };
        let mut profiles = BTreeSet::new();
        for p in items(&base, "profiles") {
            let id = text(p, "id");
            if !profiles.insert(id.to_owned()) {
                self.fail(format!("{}: duplicate profile {id}", dir.display()));
            }
            let mut paths = BTreeSet::new();
            for f in items(p, "files") {
                if !paths.insert(text(f, "path")) {
                    self.fail(format!(
                        "{}: profile {id} lists {} twice",
                        dir.display(),
                        text(f, "path")
                    ));
                }
            }
        }
        self.run_profiles(dir, &base_runs, &profiles);
        for slug in slugs {
            self.patch(&slug, &profiles, &base_runs);
        }
    }

    fn run_profiles(&mut self, dir: &Path, runs: &Runs, profiles: &BTreeSet<String>) {
        for (id, run) in runs {
            if let Some(run) = run {
                if !profiles.contains(text(run, "profile")) {
                    let p = text(run, "profile");
                    self.fail(format!(
                        "{}: run {id} names unknown profile {p}",
                        dir.display()
                    ));
                }
            }
        }
    }

    /// A run is a folder named by its ID holding run.json; outcome and
    /// resource files may sit beside it, other JSON may not.
    fn runs(&mut self, dir: &Path) -> Runs {
        let mut out = Runs::new();
        for e in self.read_dir(dir) {
            if !e.is_dir || !ID.is_match(&e.name) {
                self.fail(format!(
                    "{}: a run is a folder named by its ID",
                    e.path.display()
                ));
                continue;
            }
            let run = self.record(&e.path.join("run.json"), "run");
            for f in self.read_dir(&e.path) {
                if f.is_dir || (f.name != "run.json" && f.name.ends_with(".json")) {
                    self.fail(format!("{}: unexpected path", f.path.display()));
                }
            }
            if let Some(run) = &run {
                self.hosted(&e.path, run, "", false);
            }
            out.insert(e.name, run);
        }
        out
    }

    fn patch(&mut self, dir: &Path, profiles: &BTreeSet<String>, base_runs: &Runs) {
        let mut meta = None;
        let mut folders = BTreeSet::new();
        for e in self.read_dir(dir) {
            match (e.name.as_str(), e.is_dir) {
                ("patch.json", false) => meta = self.record(&e.path, "patch"),
                ("README.md", false) | ("notices", true) => {}
                (name, true) if VERSION.is_match(name) => drop(folders.insert(e.name)),
                _ => self.fail(format!("{}: unexpected path", e.path.display())),
            }
        }
        let Some(meta) = meta else {
            self.fail(format!("{}: missing or invalid patch.json", dir.display()));
            return;
        };
        for (i, v) in items(&meta, "versions").iter().enumerate() {
            let folder = format!("v{}", i + 1);
            if v["version"].as_u64() != Some(i as u64 + 1) {
                self.fail(format!(
                    "{}: versions must be listed as 1..N without gaps",
                    dir.display()
                ));
                return;
            }
            if !folders.remove(&folder) {
                self.fail(format!(
                    "{}: version {} has no folder",
                    dir.display(),
                    i + 1
                ));
                continue;
            }
            let withdrawn = text(v, "status") == "withdrawn";
            self.version(
                &dir.join(folder),
                text(&meta, "kind"),
                withdrawn,
                profiles,
                base_runs,
            );
        }
        for folder in folders {
            self.fail(format!(
                "{}: {folder} is not listed in patch.json",
                dir.display()
            ));
        }
    }

    fn version(
        &mut self,
        dir: &Path,
        kind: &str,
        withdrawn: bool,
        profiles: &BTreeSet<String>,
        base_runs: &Runs,
    ) {
        let mut version = None;
        let mut runs = Runs::new();
        let mut comparison_ids = Vec::new();
        let mut assessment_ids = Vec::new();
        let mut claim_ids = Vec::new();
        for e in self.read_dir(dir) {
            match (e.name.as_str(), e.is_dir) {
                ("version.json", false) => version = self.record(&e.path, "version"),
                ("files", true) => {}
                ("runs", true) => runs = self.runs(&e.path),
                (folder @ ("comparisons" | "assessments" | "claims"), true) => {
                    for f in self.read_dir(&e.path) {
                        match f.name.strip_suffix(".json") {
                            Some(id) if !f.is_dir && ID.is_match(id) => {
                                let ids = match folder {
                                    "comparisons" => &mut comparison_ids,
                                    "assessments" => &mut assessment_ids,
                                    _ => &mut claim_ids,
                                };
                                ids.push(id.to_owned());
                            }
                            _ => self.fail(format!("{}: expected <id>.json", f.path.display())),
                        }
                    }
                }
                _ => self.fail(format!("{}: unexpected path", e.path.display())),
            }
        }
        let Some(version) = version else {
            self.fail(format!(
                "{}: missing or invalid version.json",
                dir.display()
            ));
            return;
        };
        let at = dir.display();
        if !profiles.contains(text(&version, "target_profile")) {
            self.fail(format!(
                "{at}: unknown target profile {}",
                text(&version, "target_profile")
            ));
        }

        let files = items(&version, "files");
        let mut names = BTreeSet::new();
        let mut sources = Vec::new();
        let mut source_digests = BTreeSet::new();
        for f in files {
            if !names.insert(text(f, "name")) {
                self.fail(format!("{at}: file name {} used twice", text(f, "name")));
            }
            if text(f, "role") == "source" {
                sources.push(text(f, "name").to_owned());
                source_digests.insert(text(f, "sha256"));
            }
        }
        for f in files.iter().filter(|f| text(f, "role") == "derived") {
            let name = text(f, "name");
            if items(f, "derived_from")
                .iter()
                .any(|d| !source_digests.contains(d.as_str().unwrap_or("")))
            {
                self.fail(format!(
                    "{at}: {name} is derived from a file that is not a source of this version"
                ));
            }
            if ![".safetensors", ".gguf", ".json"]
                .iter()
                .any(|ext| name.ends_with(ext))
            {
                self.fail(format!(
                    "{at}: derived file {name} is not safetensors, GGUF or JSON"
                ));
            }
        }
        if let Err(e) = check_sources(kind, &sources) {
            self.fail(format!("{at}: {e}"));
        }

        let referenced = self.hosted(dir, &version, "files/", withdrawn);
        if !withdrawn {
            for path in walk_files(&dir.join("files")) {
                if !referenced.contains(&rel_parts(dir, &path).join("/")) {
                    self.fail(format!(
                        "{}: hosted file is not listed in version.json",
                        path.display()
                    ));
                }
            }
            if kind == "replacement" {
                for f in files.iter().filter(|f| text(f, "role") == "source") {
                    for loc in items(f, "locators")
                        .iter()
                        .filter(|l| text(l, "kind") == "index-path")
                    {
                        if let Err(e) = check_replacement_file(&dir.join(text(loc, "path"))) {
                            self.fail(format!("{at}: {}: {e}", text(f, "name")));
                        }
                    }
                }
            }
        }

        if !withdrawn {
            if let Err(e) = dual::check(dir, kind, files) {
                self.fail(e);
            }
        }

        self.run_profiles(dir, &runs, profiles);
        for id in &comparison_ids {
            let Some(cmp) = self.record(
                &dir.join("comparisons").join(format!("{id}.json")),
                "comparison",
            ) else {
                continue;
            };
            let baseline = base_runs.get(text(&cmp, "baseline_run"));
            let patched = runs.get(text(&cmp, "patched_run"));
            if baseline.is_none() {
                self.fail(format!(
                    "{at}: comparison {id} names unknown baseline run {}",
                    text(&cmp, "baseline_run")
                ));
            }
            if patched.is_none() {
                self.fail(format!(
                    "{at}: comparison {id} names unknown run {} of this version",
                    text(&cmp, "patched_run")
                ));
            }
            if let (Some(Some(b)), Some(Some(p))) = (baseline, patched) {
                if text(b, "profile") != text(p, "profile") {
                    self.fail(format!(
                        "{at}: comparison {id} compares runs on different profiles"
                    ));
                }
            }
            self.hosted(&dir.join("comparisons"), &cmp, "", false);
        }
        for id in &claim_ids {
            self.record(&dir.join("claims").join(format!("{id}.json")), "claim");
        }
        for id in &assessment_ids {
            let Some(a) = self.record(
                &dir.join("assessments").join(format!("{id}.json")),
                "assessment",
            ) else {
                continue;
            };
            for c in items(&a, "comparisons") {
                let c = c.as_str().unwrap_or("");
                if !comparison_ids.iter().any(|id| id == c) {
                    self.fail(format!(
                        "{at}: assessment {id} cites unknown comparison {c}"
                    ));
                }
            }
        }
    }

    /// Finds every artifact in `record` with an `index-path` locator. The path
    /// is relative to `dir`, the folder of the declaring file, and must start
    /// with `prefix`. Unless the version is withdrawn, the file must exist with
    /// the recorded size and SHA-256. Returns the referenced paths.
    fn hosted(
        &mut self,
        dir: &Path,
        record: &Value,
        prefix: &str,
        withdrawn: bool,
    ) -> BTreeSet<String> {
        let mut referenced = BTreeSet::new();
        let mut stack = vec![record];
        while let Some(v) = stack.pop() {
            match v {
                Value::Array(a) => stack.extend(a),
                Value::Object(o) if !o.contains_key("locators") => stack.extend(o.values()),
                Value::Object(_) => {
                    for loc in items(v, "locators")
                        .iter()
                        .filter(|l| text(l, "kind") == "index-path")
                    {
                        let rel = text(loc, "path");
                        referenced.insert(rel.to_owned());
                        if !rel.starts_with(prefix) {
                            self.fail(format!(
                                "{}: hosted path {rel} must be under {prefix}",
                                dir.display()
                            ));
                        } else if !withdrawn {
                            self.hosted_file(&dir.join(rel), v);
                        }
                    }
                }
                _ => {}
            }
        }
        referenced
    }

    fn hosted_file(&mut self, path: &Path, artifact: &Value) {
        let at = path.display();
        let regular = fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
        let Some(bytes) = regular.then(|| fs::read(path).ok()).flatten() else {
            self.fail(format!("{at}: hosted file missing"));
            return;
        };
        if hex(&Sha256::digest(&bytes)) != text(artifact, "sha256")
            || artifact["size_bytes"].as_u64() != Some(bytes.len() as u64)
        {
            self.fail(format!("{at}: size or SHA-256 differs from the record"));
        }
        if bytes.len() as u64 >= HOSTED_FILE_LIMIT {
            self.fail(format!("{at}: files hosted here must be under 100 MB"));
        }
    }
}

pub(crate) fn text<'v>(v: &'v Value, key: &str) -> &'v str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

pub(crate) fn items<'v>(v: &'v Value, key: &str) -> &'v [Value] {
    v.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every regular file under `dir`, recursively; empty when `dir` is absent.
pub(crate) fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).into_iter().flatten().flatten() {
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(entry.path()),
                Ok(_) => out.push(entry.path()),
                Err(_) => {}
            }
        }
    }
    out.sort();
    out
}

pub(crate) fn rel_parts(root: &Path, path: &Path) -> Vec<String> {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect()
}
