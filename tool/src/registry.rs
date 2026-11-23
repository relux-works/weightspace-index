//! Multi-registry configuration, synchronization, and merged catalog loading.
//!
//! Configuration lives in `~/.config/weightspace/registries.toml` (overridable
//! via `WEIGHTSPACE_CONFIG`). Registries can be:
//! - `git`: any Git host (GitHub, corporate GitLab, Gitea, SSH URLs, local Git
//!   repos), synced via the system `git` binary so existing SSH keys and
//!   credential helpers work with zero authentication inside `wsp`.
//! - `http`: a static HTTP(S) base URL serving a Weightspace Index directory
//!   layout (`models/index.json` and `models/<org>/<model>/index.json`).
//! - `local`: a local directory containing `models/index.json`.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::ToolError;

pub const OFFICIAL_REGISTRY_NAME: &str = "official";
pub const OFFICIAL_REGISTRY_URL: &str = "https://github.com/relux-works/weightspace-index.git";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegistryKind {
    Git,
    Http,
    Local,
}

impl RegistryKind {
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        match s {
            "git" => Ok(Self::Git),
            "http" | "https" => Ok(Self::Http),
            "local" | "dir" | "path" => Ok(Self::Local),
            _ => Err(ToolError::InvalidArgument {
                detail: format!("unknown registry kind {s:?}; expected git, http, or local"),
            }),
        }
    }

    pub fn infer(url: &str) -> Self {
        let p = Path::new(url);
        if p.is_dir() && !p.join(".git").exists() && !url.ends_with(".git") {
            return Self::Local;
        }
        if url.starts_with("git@")
            || url.starts_with("ssh://")
            || url.ends_with(".git")
            || url.contains("github.com/")
            || url.contains("gitlab.com/")
            || (p.is_dir() && p.join(".git").exists())
        {
            return Self::Git;
        }
        if url.starts_with("http://") || url.starts_with("https://") {
            return Self::Http;
        }
        Self::Local
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Git => "git",
            Self::Http => "http",
            Self::Local => "local",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub kind: RegistryKind,
    pub url: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_priority() -> i32 {
    100
}

fn default_enabled() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub default: String,
    pub registries: Vec<RegistryEntry>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            default: OFFICIAL_REGISTRY_NAME.to_owned(),
            registries: vec![RegistryEntry {
                name: OFFICIAL_REGISTRY_NAME.to_owned(),
                kind: RegistryKind::Git,
                url: OFFICIAL_REGISTRY_URL.to_owned(),
                priority: 100,
                pin: None,
                enabled: true,
            }],
        }
    }
}

/// Cross-platform config, cache, and data directories with environment overrides.
#[derive(Clone, Debug)]
pub struct AppPaths {
    pub config_file: PathBuf,
    pub cache_dir: PathBuf,
    pub data_dir: PathBuf,
}

fn resolve_base_dir(xdg_var: &str, #[allow(unused_variables)] win_var: &str) -> Option<PathBuf> {
    if let Some(p) = env::var_os(xdg_var) {
        return Some(PathBuf::from(p));
    }
    #[cfg(windows)]
    if let Some(p) = env::var_os(win_var) {
        return Some(PathBuf::from(p));
    }
    None
}

impl AppPaths {
    pub fn resolve() -> Self {
        let home = home_dir();
        let config_file = env::var_os("WEIGHTSPACE_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                resolve_base_dir("XDG_CONFIG_HOME", "APPDATA")
                    .unwrap_or_else(|| home.join(".config"))
                    .join("weightspace")
                    .join("registries.toml")
            });

        let cache_dir = env::var_os("WEIGHTSPACE_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                resolve_base_dir("XDG_CACHE_HOME", "LOCALAPPDATA")
                    .unwrap_or_else(|| home.join(".cache"))
                    .join("weightspace")
            });

        let data_dir = env::var_os("WEIGHTSPACE_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                resolve_base_dir("XDG_DATA_HOME", "LOCALAPPDATA")
                    .unwrap_or_else(|| home.join(".local").join("share"))
                    .join("weightspace")
            });

        Self {
            config_file,
            cache_dir,
            data_dir,
        }
    }

    pub fn registry_cache_dir(&self, name: &str) -> PathBuf {
        self.cache_dir.join("registries").join(name)
    }

    pub fn patches_dir(&self) -> PathBuf {
        self.data_dir.join("patches")
    }

    pub fn bases_dir(&self) -> PathBuf {
        self.data_dir.join("bases")
    }

    pub fn patched_models_dir(&self) -> PathBuf {
        self.data_dir.join("patched")
    }
}

pub fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn validate_registry_name(name: &str) -> Result<(), ToolError> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(ToolError::InvalidArgument {
            detail: format!(
                "invalid registry name {name:?}; use ASCII letters, digits, '-' or '_'"
            ),
        });
    }
    Ok(())
}

impl RegistryConfig {
    pub fn load(paths: &AppPaths) -> Result<Self, ToolError> {
        if !paths.config_file.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(&paths.config_file)
            .map_err(|e| ToolError::io(&paths.config_file, e))?;
        toml::from_str(&raw).map_err(|e| ToolError::Registry {
            name: paths.config_file.display().to_string(),
            detail: format!("invalid registries.toml: {e}"),
        })
    }

    pub fn save(&self, paths: &AppPaths) -> Result<(), ToolError> {
        if let Some(parent) = paths.config_file.parent() {
            fs::create_dir_all(parent).map_err(|e| ToolError::io(parent, e))?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| ToolError::Registry {
            name: "config".into(),
            detail: format!("serialize registries.toml: {e}"),
        })?;
        fs::write(&paths.config_file, text).map_err(|e| ToolError::io(&paths.config_file, e))
    }

    pub fn add(
        &mut self,
        name: String,
        url: String,
        kind: Option<RegistryKind>,
        pin: Option<String>,
        priority: Option<i32>,
        make_default: bool,
    ) -> Result<(), ToolError> {
        validate_registry_name(&name)?;
        if self.registries.iter().any(|r| r.name == name) {
            return Err(ToolError::Registry {
                name,
                detail: "registry already exists; remove it first or pick another name".into(),
            });
        }
        let kind = kind.unwrap_or_else(|| RegistryKind::infer(&url));
        let priority = priority.unwrap_or(if make_default { 200 } else { 150 });
        self.registries.push(RegistryEntry {
            name: name.clone(),
            kind,
            url,
            priority,
            pin,
            enabled: true,
        });
        if make_default {
            self.default = name;
        }
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<(), ToolError> {
        let before = self.registries.len();
        self.registries.retain(|r| r.name != name);
        if self.registries.len() == before {
            return Err(ToolError::NotFound {
                detail: format!("registry {name:?} is not configured"),
            });
        }
        if self.default == name {
            self.default = self
                .registries
                .first()
                .map(|r| r.name.clone())
                .unwrap_or_else(|| OFFICIAL_REGISTRY_NAME.to_owned());
        }
        Ok(())
    }

    pub fn pin(&mut self, name: &str, rev: Option<String>) -> Result<(), ToolError> {
        let entry = self
            .registries
            .iter_mut()
            .find(|r| r.name == name)
            .ok_or_else(|| ToolError::NotFound {
                detail: format!("registry {name:?} is not configured"),
            })?;
        entry.pin = rev;
        Ok(())
    }

    pub fn set_default(&mut self, name: &str) -> Result<(), ToolError> {
        if !self.registries.iter().any(|r| r.name == name) {
            return Err(ToolError::NotFound {
                detail: format!("registry {name:?} is not configured"),
            });
        }
        self.default = name.to_owned();
        Ok(())
    }

    /// Ordered highest-precedence first: default registry wins ties, otherwise higher priority wins.
    pub fn ordered_enabled(&self) -> Vec<&RegistryEntry> {
        let mut list: Vec<&RegistryEntry> = self.registries.iter().filter(|r| r.enabled).collect();
        list.sort_by(|a, b| {
            let a_def = a.name == self.default;
            let b_def = b.name == self.default;
            b.priority
                .cmp(&a.priority)
                .then_with(|| b_def.cmp(&a_def))
                .then_with(|| a.name.cmp(&b.name))
        });
        list
    }
}

/// Synchronizes one registry into the local cache.
pub fn sync_registry(entry: &RegistryEntry, paths: &AppPaths) -> Result<PathBuf, ToolError> {
    match entry.kind {
        RegistryKind::Local => {
            let root = PathBuf::from(&entry.url);
            if !root.join("models").join("index.json").is_file() {
                return Err(ToolError::Registry {
                    name: entry.name.clone(),
                    detail: format!(
                        "local directory {} does not contain models/index.json",
                        root.display()
                    ),
                });
            }
            Ok(root)
        }
        RegistryKind::Git => sync_git_registry(entry, paths),
        RegistryKind::Http => sync_http_registry(entry, paths),
    }
}

fn sync_git_registry(entry: &RegistryEntry, paths: &AppPaths) -> Result<PathBuf, ToolError> {
    let dest = paths.registry_cache_dir(&entry.name);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| ToolError::io(parent, e))?;
    }

    if dest.join(".git").exists() {
        run_git(
            &entry.name,
            Some(&dest),
            &["remote", "set-url", "origin", &entry.url],
        )?;
        if let Some(pin) = &entry.pin {
            // Fetch specifically the pinned ref/commit (works for branches, tags, and SHAs).
            let _ = run_git(
                &entry.name,
                Some(&dest),
                &["fetch", "--tags", "origin", pin],
            );
            run_git(&entry.name, Some(&dest), &["fetch", "--tags", "origin"])?;
            run_git(&entry.name, Some(&dest), &["checkout", "--detach", pin])?;
        } else {
            run_git(&entry.name, Some(&dest), &["fetch", "origin"])?;
            // Resolve remote HEAD cleanly.
            let target = git_stdout(&dest, &["symbolic-ref", "refs/remotes/origin/HEAD"])
                .unwrap_or_else(|| "origin/HEAD".to_owned());
            run_git(
                &entry.name,
                Some(&dest),
                &["checkout", "--detach", target.trim()],
            )?;
        }
    } else {
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(|e| ToolError::io(&dest, e))?;
        }
        let dest_str = dest.to_string_lossy().into_owned();
        if entry.pin.is_some() {
            run_git(&entry.name, None, &["clone", &entry.url, &dest_str])?;
        } else {
            run_git(
                &entry.name,
                None,
                &["clone", "--depth", "1", &entry.url, &dest_str],
            )?;
        }
        if let Some(pin) = &entry.pin {
            run_git(&entry.name, Some(&dest), &["checkout", "--detach", pin])?;
        }
    }

    if !dest.join("models").join("index.json").is_file() {
        return Err(ToolError::Registry {
            name: entry.name.clone(),
            detail: "synced repository does not contain models/index.json".into(),
        });
    }
    Ok(dest)
}

fn sync_http_registry(entry: &RegistryEntry, paths: &AppPaths) -> Result<PathBuf, ToolError> {
    let dest = paths.registry_cache_dir(&entry.name);
    fs::create_dir_all(dest.join("models")).map_err(|e| ToolError::io(&dest, e))?;
    let base_url = entry.url.trim_end_matches('/');
    let root_url = match &entry.pin {
        Some(pin) => format!("{base_url}/{pin}/models/index.json"),
        None => format!("{base_url}/models/index.json"),
    };
    let root_json = http_get_text(&root_url)?;
    let root_path = dest.join("models").join("index.json");
    fs::write(&root_path, &root_json).map_err(|e| ToolError::io(&root_path, e))?;

    let root_idx: RootModelsIndex =
        serde_json::from_str(&root_json).map_err(|e| ToolError::Registry {
            name: entry.name.clone(),
            detail: format!("invalid models/index.json from {root_url}: {e}"),
        })?;

    for base_id in &root_idx.bases {
        let model_url = match &entry.pin {
            Some(pin) => format!("{base_url}/{pin}/models/{base_id}/index.json"),
            None => format!("{base_url}/models/{base_id}/index.json"),
        };
        let model_json = http_get_text(&model_url)?;
        let model_dir = dest.join("models").join(base_id);
        fs::create_dir_all(&model_dir).map_err(|e| ToolError::io(&model_dir, e))?;
        let model_path = model_dir.join("index.json");
        fs::write(&model_path, &model_json).map_err(|e| ToolError::io(&model_path, e))?;
    }
    Ok(dest)
}

fn http_get_text(url: &str) -> Result<String, ToolError> {
    let resp = ureq::get(url)
        .set("User-Agent", "weightspace-cli/0.2")
        .call()
        .map_err(|e| ToolError::Network {
            url: url.to_owned(),
            detail: e.to_string(),
        })?;
    resp.into_string().map_err(|e| ToolError::Network {
        url: url.to_owned(),
        detail: e.to_string(),
    })
}

fn run_git(reg_name: &str, cwd: Option<&Path>, args: &[&str]) -> Result<(), ToolError> {
    let mut cmd = Command::new("git");
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.args(args);
    // Avoid blocking non-interactively on missing credentials.
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    let out = cmd.output().map_err(|e| ToolError::Registry {
        name: reg_name.to_owned(),
        detail: format!("failed to run git {}: {e}", args.join(" ")),
    })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(ToolError::Registry {
            name: reg_name.to_owned(),
            detail: format!("git {} failed: {}", args.join(" "), stderr.trim()),
        });
    }
    Ok(())
}

fn git_stdout(cwd: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

// ---------------------------------------------------------------------------
// Catalog types loaded from `models/index.json` + `models/<org>/<model>/index.json`
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize)]
struct RootModelsIndex {
    bases: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct BaseModelIndex {
    base: String,
    #[serde(default)]
    profiles: Vec<BaseProfile>,
    #[serde(default)]
    patches: Vec<RawPatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseProfile {
    pub id: String,
    pub source: BaseSource,
    #[serde(default)]
    pub files: Vec<BaseProfileFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseSource {
    pub repo_id: String,
    pub revision: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseProfileFile {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct RawPatch {
    slug: String,
    title: String,
    summary: String,
    kind: String,
    license: String,
    #[serde(default)]
    tags: Vec<String>,
    latest: Option<u32>,
    #[serde(default)]
    versions: Vec<RawPatchVersion>,
}

#[derive(Clone, Debug, Deserialize)]
struct RawPatchVersion {
    version: u32,
    status: String,
    #[serde(default)]
    target_profile: Option<String>,
    #[serde(default)]
    changelog: Option<String>,
    #[serde(default)]
    files: Vec<PatchArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchArtifact {
    pub name: String,
    pub role: String,
    pub sha256: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub locators: Vec<ArtifactLocator>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ArtifactLocator {
    Hf {
        repo_id: String,
        revision: String,
        path: String,
        #[serde(default)]
        repo_type: Option<String>,
    },
    IndexPath {
        path: String,
    },
    Url {
        url: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, Serialize)]
pub struct CatalogBase {
    pub id: String,
    pub registry: String,
    pub profiles: Vec<BaseProfile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CatalogPatch {
    pub registry: String,
    pub registry_root: PathBuf,
    pub registry_kind: RegistryKind,
    pub registry_url: String,
    pub registry_pin: Option<String>,
    pub base: String,
    pub slug: String,
    pub title: String,
    pub summary: String,
    pub kind: String,
    pub license: String,
    pub tags: Vec<String>,
    pub version: u32,
    pub target_profile: String,
    pub changelog: String,
    pub files: Vec<PatchArtifact>,
    /// Unix timestamp when this patch was added to its registry (for newest-first sorting).
    pub added_unix: i64,
    /// Optional concise headline metric from author claims or measured runs.
    pub highlight: Option<String>,
}

impl CatalogPatch {
    pub fn address(&self) -> String {
        format!("{}/{}", self.base, self.slug)
    }

    pub fn has_gguf(&self) -> bool {
        self.files
            .iter()
            .any(|f| f.name.to_ascii_lowercase().ends_with(".gguf"))
    }

    pub fn has_peft(&self) -> bool {
        self.files
            .iter()
            .any(|f| f.name == "adapter_model.safetensors" || f.name == "adapter.safetensors")
    }

    pub fn has_replacement(&self) -> bool {
        self.kind == "replacement"
            && self
                .files
                .iter()
                .any(|f| f.role == "source" && f.name.ends_with(".safetensors"))
    }

    pub fn format_badges(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.has_replacement() {
            out.push("replacement");
        }
        if self.has_peft() {
            out.push("PEFT");
        }
        if self.has_gguf() {
            out.push("GGUF");
        }
        out
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct MergedCatalog {
    pub bases: BTreeMap<String, CatalogBase>,
    pub patches: Vec<CatalogPatch>,
    pub loaded_registries: Vec<String>,
}

impl MergedCatalog {
    /// Finds a patch by `<org>/<model>/<slug>` (with optional `@vN`) or unique `<slug>`.
    pub fn resolve_patch(&self, query: &str) -> Result<&CatalogPatch, ToolError> {
        let (addr, _ver) = query.split_once('@').unwrap_or((query, ""));
        let q = addr.trim().trim_matches('/');
        let exact: Vec<&CatalogPatch> = self
            .patches
            .iter()
            .filter(|p| p.address().eq_ignore_ascii_case(q))
            .collect();
        if exact.len() == 1 {
            return Ok(exact[0]);
        }
        let by_slug: Vec<&CatalogPatch> = self
            .patches
            .iter()
            .filter(|p| p.slug.eq_ignore_ascii_case(q))
            .collect();
        match by_slug.as_slice() {
            [one] => Ok(one),
            [] => Err(ToolError::NotFound {
                detail: format!(
                    "no patch matches {query:?}; run `wsp search` to list available patches"
                ),
            }),
            many => {
                let choices: Vec<String> = many.iter().map(|p| p.address()).collect();
                Err(ToolError::InvalidArgument {
                    detail: format!(
                        "ambiguous patch slug {query:?}; specify full address: {}",
                        choices.join(", ")
                    ),
                })
            }
        }
    }

    pub fn search(
        &self,
        query: Option<&str>,
        kind: Option<&str>,
        base: Option<&str>,
        registry: Option<&str>,
    ) -> Vec<&CatalogPatch> {
        let needle = query.map(|q| q.to_ascii_lowercase());
        let mut results: Vec<&CatalogPatch> = self
            .patches
            .iter()
            .filter(|p| {
                if let Some(k) = kind
                    && !p.kind.eq_ignore_ascii_case(k)
                {
                    return false;
                }
                if let Some(b) = base
                    && !p
                        .base
                        .to_ascii_lowercase()
                        .contains(&b.to_ascii_lowercase())
                {
                    return false;
                }
                if let Some(r) = registry
                    && !p.registry.eq_ignore_ascii_case(r)
                {
                    return false;
                }
                if let Some(ref q) = needle {
                    let hay = format!(
                        "{} {} {} {} {}",
                        p.base,
                        p.slug,
                        p.title,
                        p.summary,
                        p.tags.join(" ")
                    )
                    .to_ascii_lowercase();
                    return q.split_whitespace().all(|tok| hay.contains(tok));
                }
                true
            })
            .collect();
        results.sort_by(|a, b| {
            b.added_unix
                .cmp(&a.added_unix)
                .then_with(|| a.base.cmp(&b.base))
                .then_with(|| a.slug.cmp(&b.slug))
        });
        results
    }
}

/// Loads and merges all enabled registries. If `sync_missing` is true, any
/// uncached registry is synced first (unless a local checkout of the index is
/// found in the current directory for the `official` registry).
pub fn load_merged_catalog(
    cfg: &RegistryConfig,
    paths: &AppPaths,
    registry_filter: Option<&str>,
    sync_missing: bool,
) -> Result<MergedCatalog, ToolError> {
    let mut merged = MergedCatalog::default();
    let mut seen_patches: HashMap<(String, String), usize> = HashMap::new();

    for entry in cfg.ordered_enabled() {
        if let Some(filter) = registry_filter
            && entry.name != filter
        {
            continue;
        }
        let root = resolve_registry_root(entry, paths, sync_missing)?;
        let Some(root) = root else {
            continue;
        };
        load_one_registry(entry, &root, &mut merged, &mut seen_patches)?;
        merged.loaded_registries.push(entry.name.clone());
    }

    // Sort newest added first by default (used directly by TUI and search).
    merged.patches.sort_by(|a, b| {
        b.added_unix
            .cmp(&a.added_unix)
            .then_with(|| a.base.cmp(&b.base))
            .then_with(|| a.slug.cmp(&b.slug))
    });
    Ok(merged)
}

fn resolve_registry_root(
    entry: &RegistryEntry,
    paths: &AppPaths,
    sync_missing: bool,
) -> Result<Option<PathBuf>, ToolError> {
    match entry.kind {
        RegistryKind::Local => {
            let p = PathBuf::from(&entry.url);
            if p.join("models").join("index.json").is_file() {
                Ok(Some(p))
            } else {
                Err(ToolError::Registry {
                    name: entry.name.clone(),
                    detail: format!("missing models/index.json under {}", p.display()),
                })
            }
        }
        RegistryKind::Git | RegistryKind::Http => {
            let cached = paths.registry_cache_dir(&entry.name);
            if cached.join("models").join("index.json").is_file() {
                return Ok(Some(cached));
            }
            // Developer convenience: if running inside a local Weightspace Index checkout
            // and `official` has never been synced yet, read the local repo seamlessly.
            if entry.name == OFFICIAL_REGISTRY_NAME
                && entry.pin.is_none()
                && let Some(local_repo) = detect_cwd_index_repo()
            {
                return Ok(Some(local_repo));
            }
            if sync_missing {
                Ok(Some(sync_registry(entry, paths)?))
            } else {
                Ok(None)
            }
        }
    }
}

fn detect_cwd_index_repo() -> Option<PathBuf> {
    let mut cur = env::current_dir().ok()?;
    for _ in 0..4 {
        if cur.join("models").join("index.json").is_file()
            && cur.join("schema").join("records.schema.json").is_file()
        {
            return Some(cur);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

fn load_one_registry(
    entry: &RegistryEntry,
    root: &Path,
    merged: &mut MergedCatalog,
    seen_patches: &mut HashMap<(String, String), usize>,
) -> Result<(), ToolError> {
    let root_idx_path = root.join("models").join("index.json");
    let raw = fs::read_to_string(&root_idx_path).map_err(|e| ToolError::io(&root_idx_path, e))?;
    let root_idx: RootModelsIndex =
        serde_json::from_str(&raw).map_err(|e| ToolError::Registry {
            name: entry.name.clone(),
            detail: format!("invalid {}: {e}", root_idx_path.display()),
        })?;

    let git_dates = if root.join(".git").exists() {
        collect_git_added_times(root)
    } else {
        HashMap::new()
    };

    for (order_idx, base_id) in root_idx.bases.iter().enumerate() {
        let model_idx_path = root.join("models").join(base_id).join("index.json");
        if !model_idx_path.is_file() {
            continue;
        }
        let mraw =
            fs::read_to_string(&model_idx_path).map_err(|e| ToolError::io(&model_idx_path, e))?;
        let model_idx: BaseModelIndex =
            serde_json::from_str(&mraw).map_err(|e| ToolError::Registry {
                name: entry.name.clone(),
                detail: format!("invalid {}: {e}", model_idx_path.display()),
            })?;

        merged
            .bases
            .entry(model_idx.base.clone())
            .or_insert_with(|| CatalogBase {
                id: model_idx.base.clone(),
                registry: entry.name.clone(),
                profiles: model_idx.profiles.clone(),
            });

        for (patch_idx, p) in model_idx.patches.into_iter().enumerate() {
            let chosen = p
                .latest
                .and_then(|v| p.versions.iter().find(|rv| rv.version == v))
                .or_else(|| p.versions.iter().find(|rv| rv.status == "active"))
                .or_else(|| p.versions.last());
            let Some(ver) = chosen else {
                continue;
            };
            let key = (model_idx.base.clone(), p.slug.clone());
            if seen_patches.contains_key(&key) {
                continue;
            }

            let patch_rel = format!("models/{}/{}/patch.json", model_idx.base, p.slug);
            let added_unix = git_dates
                .get(&patch_rel)
                .copied()
                .or_else(|| {
                    fs::metadata(root.join(&patch_rel))
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                })
                .unwrap_or((order_idx * 100 + patch_idx) as i64);

            let highlight = load_claim_highlight(root, &model_idx.base, &p.slug, ver.version);

            seen_patches.insert(key, merged.patches.len());
            merged.patches.push(CatalogPatch {
                registry: entry.name.clone(),
                registry_root: root.to_path_buf(),
                registry_kind: entry.kind,
                registry_url: entry.url.clone(),
                registry_pin: entry.pin.clone(),
                base: model_idx.base.clone(),
                slug: p.slug,
                title: p.title,
                summary: p.summary,
                kind: p.kind,
                license: p.license,
                tags: p.tags,
                version: ver.version,
                target_profile: ver
                    .target_profile
                    .clone()
                    .unwrap_or_else(|| "bf16".to_owned()),
                changelog: ver.changelog.clone().unwrap_or_default(),
                files: ver.files.clone(),
                added_unix,
                highlight,
            });
        }
    }
    Ok(())
}

fn collect_git_added_times(repo_root: &Path) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let Some(out) = git_stdout(
        repo_root,
        &[
            "log",
            "--diff-filter=A",
            "--format=TS:%ct",
            "--name-only",
            "--",
            "models",
        ],
    ) else {
        return map;
    };
    let mut current_ts = 0i64;
    for line in out.lines() {
        if let Some(ts) = line.strip_prefix("TS:") {
            current_ts = ts.trim().parse::<i64>().unwrap_or(0);
        } else if line.ends_with("/patch.json") {
            map.entry(line.trim().to_owned()).or_insert(current_ts);
        }
    }
    map
}

fn load_claim_highlight(root: &Path, base: &str, slug: &str, version: u32) -> Option<String> {
    let claims_dir = root
        .join("models")
        .join(base)
        .join(slug)
        .join(format!("v{version}"))
        .join("claims");
    let entries = fs::read_dir(claims_dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let raw = fs::read_to_string(&p).ok()?;
        let val: serde_json::Value = serde_json::from_str(&raw).ok()?;
        if let Some(bench) = val.get("benchmark").and_then(|b| b.as_str())
            && let Some(m) = val.get("metric").and_then(|m| m.as_str())
            && let Some(after) = val.get("after").and_then(|a| a.as_f64())
        {
            if let Some(before) = val.get("before").and_then(|b| b.as_f64()) {
                return Some(format!("{bench} ({m}): {before:.1} -> {after:.1}"));
            }
            return Some(format!("{bench} ({m}): {after:.1}"));
        }
    }
    None
}
