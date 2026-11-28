//! Local inference engine detection, model-directory discovery (Hugging Face,
//! llama.cpp, Ollama, LM Studio, mlx-serve, Strata, DwarfStar/ds4, Weightspace
//! managed), and launch command planning.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::ToolError;
use crate::registry::{AppPaths, CatalogPatch, home_dir};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    LlamaCpp,
    Vllm,
    Transformers,
    Ollama,
    LmStudio,
    MlxLm,
    MlxServe,
    Strata,
    DwarfStar,
}

impl EngineKind {
    pub const ALL: [Self; 9] = [
        Self::LlamaCpp,
        Self::Vllm,
        Self::Transformers,
        Self::Ollama,
        Self::LmStudio,
        Self::MlxLm,
        Self::MlxServe,
        Self::Strata,
        Self::DwarfStar,
    ];

    pub fn parse(s: &str) -> Result<Self, ToolError> {
        match s.to_ascii_lowercase().as_str() {
            "llama.cpp" | "llamacpp" | "llama" | "llama-cpp" | "llama-server" => Ok(Self::LlamaCpp),
            "vllm" => Ok(Self::Vllm),
            "transformers" | "peft" | "hf" | "huggingface" => Ok(Self::Transformers),
            "ollama" => Ok(Self::Ollama),
            "lm-studio" | "lmstudio" | "lms" => Ok(Self::LmStudio),
            "mlx" | "mlx-lm" | "mlx_lm" => Ok(Self::MlxLm),
            "mlx-serve" | "mlxserve" | "mlx_serve" => Ok(Self::MlxServe),
            "strata" | "strata-server" | "strata-engine" => Ok(Self::Strata),
            "ds4" | "dwarfstar" | "dwarf-star" => Ok(Self::DwarfStar),
            _ => Err(ToolError::InvalidArgument {
                detail: format!(
                    "unknown engine {s:?}; expected llama.cpp, vllm, transformers, ollama, lm-studio, mlx-lm, mlx-serve, strata, or ds4"
                ),
            }),
        }
    }

    pub fn id(&self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Vllm => "vllm",
            Self::Transformers => "transformers",
            Self::Ollama => "ollama",
            Self::LmStudio => "lm-studio",
            Self::MlxLm => "mlx-lm",
            Self::MlxServe => "mlx-serve",
            Self::Strata => "strata",
            Self::DwarfStar => "ds4",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Vllm => "vLLM",
            Self::Transformers => "Transformers / PEFT",
            Self::Ollama => "Ollama",
            Self::LmStudio => "LM Studio",
            Self::MlxLm => "MLX-LM",
            Self::MlxServe => "mlx-serve",
            Self::Strata => "Strata",
            Self::DwarfStar => "DwarfStar (ds4)",
        }
    }

    pub fn prefers_gguf(&self) -> bool {
        matches!(
            self,
            Self::LlamaCpp | Self::Ollama | Self::LmStudio | Self::Strata | Self::DwarfStar
        )
    }

    pub fn candidate_binaries(&self) -> &'static [&'static str] {
        match self {
            Self::LlamaCpp => &["llama-server", "llama-cli"],
            Self::Vllm => &["vllm"],
            Self::Transformers => &["python3", "python"],
            Self::Ollama => &["ollama"],
            Self::LmStudio => &["lms"],
            Self::MlxLm => &["mlx_lm.server", "mlx_lm.generate"],
            Self::MlxServe => &["mlx-serve"],
            Self::Strata => &["strata-server", "strata-engine", "strata"],
            Self::DwarfStar => &["ds4-server", "ds4", "ds4-agent"],
        }
    }

    pub fn install_hint(&self) -> &'static str {
        match self {
            Self::LlamaCpp => {
                "macOS/Linux: brew install llama.cpp  (or build from https://github.com/ggml-org/llama.cpp)"
            }
            Self::Vllm => "pip install vllm  (CUDA/ROCm/Metal build: https://docs.vllm.ai/)",
            Self::Transformers => "pip install transformers peft safetensors torch",
            Self::Ollama => "https://ollama.com/download  (macOS: brew install ollama)",
            Self::LmStudio => "https://lmstudio.ai/  (then run `lms bootstrap` for CLI)",
            Self::MlxLm => "Apple Silicon: pip install mlx-lm",
            Self::MlxServe => {
                "Apple Silicon: https://mlxserve.com/  (or https://github.com/ddalcu/mlx-serve)"
            }
            Self::Strata => "Build from source: https://github.com/Niko1221/Strata",
            Self::DwarfStar => "Build from source: https://github.com/antirez/ds4",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::LlamaCpp => "GGUF inference + native `--lora <file.gguf>` runtime stacking",
            Self::Vllm => "Production server for safetensors bases + `--enable-lora` PEFT adapters",
            Self::Transformers => {
                "Python `AutoModelForCausalLM` + `PeftModel` or baked replacements"
            }
            Self::Ollama => "Local GGUF runner using `Modelfile` (`FROM` + `ADAPTER`)",
            Self::LmStudio => "Desktop & CLI (`lms`) runner for local GGUF and MLX weights",
            Self::MlxLm => "Apple Silicon native MLX runner with `--adapter-path`",
            Self::MlxServe => {
                "Native Zig/Swift Apple Silicon server (OpenAI/Anthropic/Ollama APIs, no Python)"
            }
            Self::Strata => "C++ local inference server for GGUF & Strata SQ quantized packs",
            Self::DwarfStar => "Minimal C Metal/CUDA/ROCm runner (`ds4-server -m <model.gguf>`)",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineStatus {
    pub kind: EngineKind,
    pub id: &'static str,
    pub label: &'static str,
    pub installed: bool,
    pub binary_path: Option<PathBuf>,
    pub install_hint: &'static str,
    pub description: &'static str,
}

pub fn detect_engines() -> Vec<EngineStatus> {
    EngineKind::ALL
        .iter()
        .map(|&kind| {
            let found = kind
                .candidate_binaries()
                .iter()
                .find_map(|bin| find_in_path(bin));
            EngineStatus {
                kind,
                id: kind.id(),
                label: kind.label(),
                installed: found.is_some(),
                binary_path: found,
                install_hint: kind.install_hint(),
                description: kind.description(),
            }
        })
        .collect()
}

pub fn detect_engine(kind: EngineKind) -> EngineStatus {
    let found = kind
        .candidate_binaries()
        .iter()
        .find_map(|bin| find_in_path(bin));
    EngineStatus {
        kind,
        id: kind.id(),
        label: kind.label(),
        installed: found.is_some(),
        binary_path: found,
        install_hint: kind.install_hint(),
        description: kind.description(),
    }
}

pub fn find_in_path(binary: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if is_executable(&candidate) {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            for ext in [".exe", ".cmd", ".bat"] {
                let c = dir.join(format!("{binary}{ext}"));
                if c.is_file() {
                    return Some(c);
                }
            }
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.is_file() && (meta.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        meta.is_file()
    }
}

/// Source directory kind for a discovered local base model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelSourceKind {
    HuggingFace,
    LlamaCpp,
    Ollama,
    LmStudio,
    MlxServe,
    Strata,
    DwarfStar,
    Weightspace,
}

impl ModelSourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HuggingFace => "hf-cache",
            Self::LlamaCpp => "llama.cpp",
            Self::Ollama => "ollama",
            Self::LmStudio => "lm-studio",
            Self::MlxServe => "mlx-serve",
            Self::Strata => "strata",
            Self::DwarfStar => "ds4",
            Self::Weightspace => "weightspace",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelWeightFormat {
    Safetensors,
    Gguf,
    OllamaManifest,
    Unknown,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalModel {
    /// Best-effort canonical repo ID (e.g. `Qwen/Qwen3.8-27B`) or filename stem.
    pub model_id: String,
    pub source: ModelSourceKind,
    pub format: ModelWeightFormat,
    /// Path to model directory (HF/safetensors) or concrete `.gguf` file.
    pub path: PathBuf,
    pub revision: Option<String>,
    pub has_single_safetensors: bool,
}

/// Search roots for local model discovery (injectable in tests).
#[derive(Clone, Debug, Default)]
pub struct DiscoveryRoots {
    pub hf_hub_dirs: Vec<PathBuf>,
    pub llama_cpp_dirs: Vec<PathBuf>,
    pub ollama_dirs: Vec<PathBuf>,
    pub lm_studio_dirs: Vec<PathBuf>,
    pub mlx_serve_dirs: Vec<PathBuf>,
    pub strata_dirs: Vec<PathBuf>,
    pub ds4_dirs: Vec<PathBuf>,
    pub weightspace_bases_dir: Option<PathBuf>,
}

impl DiscoveryRoots {
    pub fn from_env(paths: &AppPaths) -> Self {
        let home = home_dir();

        let mut hf_hub_dirs = Vec::new();
        if let Some(p) = env::var_os("HUGGINGFACE_HUB_CACHE") {
            hf_hub_dirs.push(PathBuf::from(p));
        }
        if let Some(p) = env::var_os("HF_HOME") {
            hf_hub_dirs.push(PathBuf::from(p).join("hub"));
        }
        hf_hub_dirs.push(home.join(".cache").join("huggingface").join("hub"));

        let mut llama_cpp_dirs = Vec::new();
        if let Some(p) = env::var_os("LLAMA_CACHE") {
            llama_cpp_dirs.push(PathBuf::from(p));
        }
        llama_cpp_dirs.push(home.join("Library").join("Caches").join("llama.cpp"));
        llama_cpp_dirs.push(home.join(".cache").join("llama.cpp"));

        let mut ollama_dirs = Vec::new();
        if let Some(p) = env::var_os("OLLAMA_MODELS") {
            ollama_dirs.push(PathBuf::from(p));
        }
        ollama_dirs.push(home.join(".ollama").join("models"));

        let lm_studio_dirs = vec![
            home.join(".cache").join("lm-studio").join("models"),
            home.join(".lmstudio").join("models"),
        ];

        let mlx_serve_dirs = vec![home.join(".mlx-serve").join("models")];

        let strata_dirs = vec![
            home.join("Strata").join("models"),
            home.join("Strata"),
            home.join(".local").join("share").join("strata"),
            home.join("Library")
                .join("Application Support")
                .join("Strata"),
        ];

        let mut ds4_dirs = vec![home.join("ds4"), home.join("Developer").join("ds4")];
        if let Ok(cwd) = env::current_dir() {
            ds4_dirs.push(cwd);
        }

        Self {
            hf_hub_dirs: dedup_paths(hf_hub_dirs),
            llama_cpp_dirs: dedup_paths(llama_cpp_dirs),
            ollama_dirs: dedup_paths(ollama_dirs),
            lm_studio_dirs: dedup_paths(lm_studio_dirs),
            mlx_serve_dirs: dedup_paths(mlx_serve_dirs),
            strata_dirs: dedup_paths(strata_dirs),
            ds4_dirs: dedup_paths(ds4_dirs),
            weightspace_bases_dir: Some(paths.bases_dir()),
        }
    }
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for p in paths {
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }
    out
}

pub fn discover_local_models(roots: &DiscoveryRoots) -> Vec<LocalModel> {
    let mut found = Vec::new();
    let mut seen_paths = BTreeSet::new();

    for dir in &roots.hf_hub_dirs {
        scan_hf_hub(dir, &mut found, &mut seen_paths);
    }
    if let Some(ws_dir) = &roots.weightspace_bases_dir {
        scan_weightspace_bases(ws_dir, &mut found, &mut seen_paths);
    }
    for dir in &roots.lm_studio_dirs {
        scan_lm_studio(dir, &mut found, &mut seen_paths);
    }
    for dir in &roots.mlx_serve_dirs {
        scan_mlx_serve(dir, &mut found, &mut seen_paths);
    }
    for dir in &roots.llama_cpp_dirs {
        scan_gguf_dir(
            dir,
            ModelSourceKind::LlamaCpp,
            2,
            &mut found,
            &mut seen_paths,
        );
    }
    for dir in &roots.strata_dirs {
        scan_gguf_dir(dir, ModelSourceKind::Strata, 2, &mut found, &mut seen_paths);
    }
    for dir in &roots.ds4_dirs {
        scan_gguf_dir(
            dir,
            ModelSourceKind::DwarfStar,
            1,
            &mut found,
            &mut seen_paths,
        );
    }
    for dir in &roots.ollama_dirs {
        scan_ollama(dir, &mut found, &mut seen_paths);
    }

    found.sort_by(|a, b| {
        a.model_id
            .cmp(&b.model_id)
            .then_with(|| a.path.cmp(&b.path))
    });
    found
}

fn scan_hf_hub(hub_dir: &Path, out: &mut Vec<LocalModel>, seen: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = fs::read_dir(hub_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let parts: Vec<&str> = rest.split("--").collect();
        if parts.len() < 2 {
            continue;
        }
        let repo_id = format!("{}/{}", parts[0], parts[1..].join("--"));
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snaps) = fs::read_dir(&snapshots_dir) else {
            continue;
        };
        let mut snap_entries: Vec<PathBuf> = snaps
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        snap_entries.sort();
        for snap in snap_entries {
            if !seen.insert(snap.clone()) {
                continue;
            }
            let (format, single_st) = inspect_model_dir(&snap);
            if format == ModelWeightFormat::Unknown {
                continue;
            }
            let revision = snap.file_name().map(|n| n.to_string_lossy().into_owned());
            out.push(LocalModel {
                model_id: repo_id.clone(),
                source: ModelSourceKind::HuggingFace,
                format,
                path: snap,
                revision,
                has_single_safetensors: single_st,
            });
        }
    }
}

fn scan_weightspace_bases(
    bases_dir: &Path,
    out: &mut Vec<LocalModel>,
    seen: &mut BTreeSet<PathBuf>,
) {
    let Ok(orgs) = fs::read_dir(bases_dir) else {
        return;
    };
    for org in orgs.flatten() {
        if !org.path().is_dir() {
            continue;
        }
        let org_name = org.file_name().to_string_lossy().into_owned();
        let Ok(models) = fs::read_dir(org.path()) else {
            continue;
        };
        for model in models.flatten() {
            if !model.path().is_dir() {
                continue;
            }
            let model_name = model.file_name().to_string_lossy().into_owned();
            let repo_id = format!("{org_name}/{model_name}");
            let Ok(profiles) = fs::read_dir(model.path()) else {
                continue;
            };
            for prof in profiles.flatten() {
                let p = prof.path();
                if !p.is_dir() || !seen.insert(p.clone()) {
                    continue;
                }
                let (format, single_st) = inspect_model_dir(&p);
                if format == ModelWeightFormat::Unknown {
                    continue;
                }
                out.push(LocalModel {
                    model_id: repo_id.clone(),
                    source: ModelSourceKind::Weightspace,
                    format,
                    path: p,
                    revision: prof.file_name().to_str().map(str::to_owned),
                    has_single_safetensors: single_st,
                });
            }
        }
    }
}

fn scan_lm_studio(root: &Path, out: &mut Vec<LocalModel>, seen: &mut BTreeSet<PathBuf>) {
    let Ok(orgs) = fs::read_dir(root) else {
        return;
    };
    for org in orgs.flatten() {
        if !org.path().is_dir() {
            continue;
        }
        let org_name = org.file_name().to_string_lossy().into_owned();
        let Ok(repos) = fs::read_dir(org.path()) else {
            continue;
        };
        for repo in repos.flatten() {
            let repo_path = repo.path();
            if !repo_path.is_dir() {
                continue;
            }
            let repo_name = repo.file_name().to_string_lossy().into_owned();
            let repo_id = format!("{org_name}/{repo_name}");
            let Ok(files) = fs::read_dir(&repo_path) else {
                continue;
            };
            let mut found_gguf = false;
            for f in files.flatten() {
                let fp = f.path();
                if fp.is_file()
                    && fp
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                    && seen.insert(fp.clone())
                {
                    found_gguf = true;
                    out.push(LocalModel {
                        model_id: repo_id.clone(),
                        source: ModelSourceKind::LmStudio,
                        format: ModelWeightFormat::Gguf,
                        path: fp,
                        revision: None,
                        has_single_safetensors: false,
                    });
                }
            }
            if !found_gguf && seen.insert(repo_path.clone()) {
                let (format, single_st) = inspect_model_dir(&repo_path);
                if format != ModelWeightFormat::Unknown {
                    out.push(LocalModel {
                        model_id: repo_id,
                        source: ModelSourceKind::LmStudio,
                        format,
                        path: repo_path,
                        revision: None,
                        has_single_safetensors: single_st,
                    });
                }
            }
        }
    }
}

fn scan_mlx_serve(root: &Path, out: &mut Vec<LocalModel>, seen: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let first_name = entry.file_name().to_string_lossy().into_owned();
        let (fmt, single_st) = inspect_model_dir(&p);
        if fmt != ModelWeightFormat::Unknown {
            if seen.insert(p.clone()) {
                out.push(LocalModel {
                    model_id: first_name,
                    source: ModelSourceKind::MlxServe,
                    format: fmt,
                    path: p,
                    revision: None,
                    has_single_safetensors: single_st,
                });
            }
            continue;
        }
        // Nested `<org>/<model>` layout under `~/.mlx-serve/models/`.
        let Ok(children) = fs::read_dir(&p) else {
            continue;
        };
        for child in children.flatten() {
            let cp = child.path();
            if !cp.is_dir() || !seen.insert(cp.clone()) {
                continue;
            }
            let (cfmt, csingle) = inspect_model_dir(&cp);
            if cfmt != ModelWeightFormat::Unknown {
                let child_name = child.file_name().to_string_lossy().into_owned();
                out.push(LocalModel {
                    model_id: format!("{first_name}/{child_name}"),
                    source: ModelSourceKind::MlxServe,
                    format: cfmt,
                    path: cp,
                    revision: None,
                    has_single_safetensors: csingle,
                });
            }
        }
    }
}

fn scan_gguf_dir(
    root: &Path,
    source: ModelSourceKind,
    max_depth: usize,
    out: &mut Vec<LocalModel>,
    seen: &mut BTreeSet<PathBuf>,
) {
    if !root.is_dir() {
        return;
    }
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if depth < max_depth {
                    stack.push((p, depth + 1));
                }
            } else if p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                && seen.insert(p.clone())
            {
                let stem = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".to_owned());
                out.push(LocalModel {
                    model_id: stem,
                    source,
                    format: ModelWeightFormat::Gguf,
                    path: p,
                    revision: None,
                    has_single_safetensors: false,
                });
            }
        }
    }
}

fn scan_ollama(models_dir: &Path, out: &mut Vec<LocalModel>, seen: &mut BTreeSet<PathBuf>) {
    let manifests = models_dir.join("manifests").join("registry.ollama.ai");
    if !manifests.is_dir() {
        return;
    }
    let Ok(namespaces) = fs::read_dir(&manifests) else {
        return;
    };
    for ns in namespaces.flatten() {
        if !ns.path().is_dir() {
            continue;
        }
        let ns_name = ns.file_name().to_string_lossy().into_owned();
        let Ok(models) = fs::read_dir(ns.path()) else {
            continue;
        };
        for model in models.flatten() {
            if !model.path().is_dir() {
                continue;
            }
            let model_name = model.file_name().to_string_lossy().into_owned();
            let Ok(tags) = fs::read_dir(model.path()) else {
                continue;
            };
            for tag in tags.flatten() {
                let tp = tag.path();
                if tp.is_file() && seen.insert(tp.clone()) {
                    let tag_name = tag.file_name().to_string_lossy().into_owned();
                    let id = if ns_name == "library" {
                        format!("{model_name}:{tag_name}")
                    } else {
                        format!("{ns_name}/{model_name}:{tag_name}")
                    };
                    out.push(LocalModel {
                        model_id: id,
                        source: ModelSourceKind::Ollama,
                        format: ModelWeightFormat::OllamaManifest,
                        path: tp,
                        revision: Some(tag_name),
                        has_single_safetensors: false,
                    });
                }
            }
        }
    }
}

fn inspect_model_dir(dir: &Path) -> (ModelWeightFormat, bool) {
    let Ok(entries) = fs::read_dir(dir) else {
        return (ModelWeightFormat::Unknown, false);
    };
    let mut st_count = 0usize;
    let mut has_gguf = false;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".safetensors") {
            st_count += 1;
        } else if name.ends_with(".gguf") {
            has_gguf = true;
        }
    }
    if st_count > 0 {
        (ModelWeightFormat::Safetensors, st_count == 1)
    } else if has_gguf {
        (ModelWeightFormat::Gguf, false)
    } else {
        (ModelWeightFormat::Unknown, false)
    }
}

/// Checks whether a discovered local model corresponds to a catalog base ID
/// (e.g., `Qwen/Qwen3.8-27B` matches HF snapshot `Qwen/Qwen3.8-27B`, LM Studio
/// `Qwen/Qwen3.8-27B-GGUF`, or GGUF filename `Qwen3.8-27B-Q4_K_M.gguf`).
pub fn matches_base(local: &LocalModel, base_id: &str) -> bool {
    let local_norm = normalize_name(&local.model_id);
    let base_norm = normalize_name(base_id);
    if local_norm == base_norm {
        return true;
    }
    let short_base = base_id
        .split_once('/')
        .map(|(_, m)| normalize_name(m))
        .unwrap_or(base_norm);
    // Avoid trivial false positives on very short strings.
    short_base.len() >= 5 && local_norm.contains(&short_base)
}

fn normalize_name(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Finds the best local base model for a catalog base ID, preferring the
/// format native to `engine`.
pub fn best_local_base<'a>(
    locals: &'a [LocalModel],
    base_id: &str,
    engine: EngineKind,
) -> Option<&'a LocalModel> {
    let mut candidates: Vec<&LocalModel> =
        locals.iter().filter(|m| matches_base(m, base_id)).collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by_key(|m| {
        let format_score = match (engine.prefers_gguf(), m.format) {
            (true, ModelWeightFormat::Gguf) => 0,
            (true, ModelWeightFormat::OllamaManifest) if engine == EngineKind::Ollama => 0,
            (false, ModelWeightFormat::Safetensors) => 0,
            _ => 1,
        };
        let exact_score = if m.model_id.eq_ignore_ascii_case(base_id) {
            0
        } else {
            1
        };
        (format_score, exact_score)
    });
    Some(candidates[0])
}

/// Planned launch command and absolute artifact paths after preparing a patch.
#[derive(Clone, Debug, Serialize)]
pub struct LaunchPlan {
    pub engine: EngineKind,
    pub engine_installed: bool,
    pub install_hint: String,
    pub executable: Option<PathBuf>,
    pub args: Vec<String>,
    pub display_command: String,
    pub base_path: Option<PathBuf>,
    pub patched_model_path: Option<PathBuf>,
    pub adapter_path: Option<PathBuf>,
    pub notes: Vec<String>,
}

pub fn plan_launch(
    engine: EngineKind,
    patch: &CatalogPatch,
    base_path: Option<&Path>,
    patched_model_path: Option<&Path>,
    adapter_path: Option<&Path>,
) -> LaunchPlan {
    let status = detect_engine(engine);
    let mut notes = Vec::new();
    let base_str = base_path
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("<path-to-{}>", patch.base));
    let patched_str = patched_model_path.map(|p| p.display().to_string());
    let adapter_str = adapter_path.map(|p| p.display().to_string());

    if !status.installed {
        notes.push(format!(
            "{} was not found in PATH. Install it via: {}",
            engine.label(),
            engine.install_hint()
        ));
    }

    let (executable, args, display_command) = match engine {
        EngineKind::LlamaCpp => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("llama-server")));
            let mut a = vec!["-m".into()];
            if let Some(ref pm) = patched_str {
                a.push(pm.clone());
                notes.push("Using baked replacement model.".into());
            } else {
                a.push(base_str.clone());
            }
            if let Some(ref ad) = adapter_str {
                if ad.to_ascii_lowercase().ends_with(".gguf") {
                    a.push("--lora".into());
                    a.push(ad.clone());
                } else {
                    notes.push(format!(
                        "Downloaded adapter ({ad}) is safetensors/PEFT; llama.cpp requires a GGUF LoRA (`convert_lora_to_gguf.py`)."
                    ));
                }
            }
            a.extend(["--port".into(), "8080".into(), "-ngl".into(), "99".into()]);
            let cmd = format_cmd("llama-server", &a);
            (exe, a, cmd)
        }
        EngineKind::Vllm => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("vllm")));
            let mut a = vec!["serve".into()];
            if let Some(ref pm) = patched_str {
                a.push(pm.clone());
            } else {
                a.push(base_str.clone());
                if let Some(ref ad) = adapter_str {
                    let ad_dir = adapter_path
                        .and_then(|p| if p.is_file() { p.parent() } else { Some(p) })
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| ad.clone());
                    a.push("--enable-lora".into());
                    a.push("--lora-modules".into());
                    a.push(format!("{}={ad_dir}", patch.slug));
                }
            }
            let cmd = format_cmd("vllm", &a);
            (exe, a, cmd)
        }
        EngineKind::Transformers => {
            let target = patched_str.as_deref().unwrap_or(&base_str);
            let snippet = if let Some(ref ad) = adapter_str
                && patched_str.is_none()
            {
                let ad_dir = adapter_path
                    .and_then(|p| if p.is_file() { p.parent() } else { Some(p) })
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| ad.clone());
                format!(
                    "python3 -c \"from transformers import AutoModelForCausalLM; from peft import PeftModel; m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained({target:?}), {ad_dir:?}); print(m)\""
                )
            } else {
                format!(
                    "python3 -c \"from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained({target:?}); print(m)\""
                )
            };
            (None, Vec::new(), snippet)
        }
        EngineKind::Ollama => {
            let model_name = format!("wsp-{}", patch.slug);
            let cmd = if let Some(ref ad) = adapter_str {
                format!(
                    "printf 'FROM {base_str}\\nADAPTER {ad}\\n' > Modelfile && ollama create {model_name} -f Modelfile && ollama run {model_name}"
                )
            } else {
                format!(
                    "printf 'FROM {}\\n' > Modelfile && ollama create {model_name} -f Modelfile && ollama run {model_name}",
                    patched_str.as_deref().unwrap_or(&base_str)
                )
            };
            notes.push("Ollama loads GGUF LoRA adapters via `ADAPTER` inside a Modelfile.".into());
            (None, Vec::new(), cmd)
        }
        EngineKind::LmStudio => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("lms")));
            let target = patched_str.as_deref().unwrap_or(&base_str).to_owned();
            let a = vec!["load".into(), target];
            if let Some(ref ad) = adapter_str {
                notes.push(format!(
                    "Attach adapter path inside LM Studio runtime settings: {ad}"
                ));
            }
            let cmd = format_cmd("lms", &a);
            (exe, a, cmd)
        }
        EngineKind::MlxLm => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("mlx_lm.server")));
            let mut a = vec![
                "--model".into(),
                patched_str.as_deref().unwrap_or(&base_str).to_owned(),
            ];
            if let Some(ref ad) = adapter_str
                && patched_str.is_none()
            {
                let ad_dir = adapter_path
                    .and_then(|p| if p.is_file() { p.parent() } else { Some(p) })
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| ad.clone());
                a.push("--adapter-path".into());
                a.push(ad_dir);
            }
            let cmd = format_cmd("mlx_lm.server", &a);
            (exe, a, cmd)
        }
        EngineKind::MlxServe => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("mlx-serve")));
            let target = patched_str.as_deref().unwrap_or(&base_str).to_owned();
            let a = vec![
                "serve".into(),
                "--model-dir".into(),
                target,
                "--port".into(),
                "11234".into(),
            ];
            if let Some(ref ad) = adapter_str
                && patched_str.is_none()
            {
                notes.push(format!(
                    "mlx-serve serves baked/merged MLX safetensors directories directly; adapter downloaded at: {ad} (or run dynamically via `mlx-lm --adapter-path`)."
                ));
            }
            notes.push(
                "mlx-serve site/repo: https://mlxserve.com/ · https://github.com/ddalcu/mlx-serve"
                    .into(),
            );
            let cmd = format_cmd("mlx-serve", &a);
            (exe, a, cmd)
        }
        EngineKind::Strata => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("strata-server")));
            let target = patched_str.as_deref().unwrap_or(&base_str).to_owned();
            let a = vec!["--model".into(), target];
            if let Some(ref ad) = adapter_str {
                notes.push(format!("Downloaded adapter ready at: {ad}"));
            }
            notes.push("Strata repo: https://github.com/Niko1221/Strata".into());
            let cmd = format_cmd("strata-server", &a);
            (exe, a, cmd)
        }
        EngineKind::DwarfStar => {
            let exe = status
                .binary_path
                .clone()
                .or_else(|| Some(PathBuf::from("ds4-server")));
            let target = patched_str.as_deref().unwrap_or(&base_str).to_owned();
            let a = vec!["-m".into(), target];
            if let Some(ref ad) = adapter_str {
                notes.push(format!(
                    "ds4 takes a single GGUF via `-m`; adapter downloaded at: {ad}"
                ));
            }
            notes.push("DwarfStar (ds4) repo: https://github.com/antirez/ds4".into());
            let cmd = format_cmd("ds4-server", &a);
            (exe, a, cmd)
        }
    };

    LaunchPlan {
        engine,
        engine_installed: status.installed,
        install_hint: engine.install_hint().to_owned(),
        executable,
        args,
        display_command,
        base_path: base_path.map(Path::to_path_buf),
        patched_model_path: patched_model_path.map(Path::to_path_buf),
        adapter_path: adapter_path.map(Path::to_path_buf),
        notes,
    }
}

fn format_cmd(bin: &str, args: &[String]) -> String {
    let mut parts = vec![bin.to_owned()];
    for arg in args {
        if arg.contains(' ') || arg.contains('"') || arg.contains('\'') {
            parts.push(format!("{arg:?}"));
        } else {
            parts.push(arg.clone());
        }
    }
    parts.join(" ")
}
