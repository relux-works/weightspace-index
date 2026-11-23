//! Zero-authentication streaming downloader for Weightspace patches, LoRA
//! adapters, and pinned Hugging Face base models, with SHA-256 verification.

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::ToolError;
use crate::files::{Sha256Hex, hash_file_with_progress};
use crate::registry::{
    AppPaths, ArtifactLocator, BaseProfile, CatalogPatch, PatchArtifact, RegistryKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PullFormat {
    Auto,
    Peft,
    Gguf,
    Replacement,
    All,
}

impl PullFormat {
    pub fn parse(s: &str) -> Result<Self, ToolError> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "peft" | "safetensors" => Ok(Self::Peft),
            "gguf" => Ok(Self::Gguf),
            "replacement" => Ok(Self::Replacement),
            "all" => Ok(Self::All),
            _ => Err(ToolError::InvalidArgument {
                detail: format!(
                    "unknown --format {s:?}; expected auto, peft, gguf, replacement, or all"
                ),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PulledPatch {
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    pub replacement_file: Option<PathBuf>,
    pub gguf_adapter: Option<PathBuf>,
    pub peft_adapter: Option<PathBuf>,
}

pub fn hf_endpoint() -> String {
    env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_owned())
        .trim_end_matches('/')
        .to_owned()
}

/// Selects which artifacts from a catalog patch to download for the requested format.
pub fn select_artifacts<'a>(
    patch: &'a CatalogPatch,
    format: PullFormat,
    prefer_gguf: bool,
) -> Vec<&'a PatchArtifact> {
    let pick_gguf = || -> Vec<&'a PatchArtifact> {
        patch
            .files
            .iter()
            .filter(|f| f.name.to_ascii_lowercase().ends_with(".gguf"))
            .collect()
    };
    let pick_peft = || -> Vec<&'a PatchArtifact> {
        patch
            .files
            .iter()
            .filter(|f| {
                f.name == "adapter_model.safetensors"
                    || f.name == "adapter_config.json"
                    || f.name == "adapter.safetensors"
            })
            .collect()
    };
    let pick_replacement = || -> Vec<&'a PatchArtifact> {
        patch
            .files
            .iter()
            .filter(|f| f.role == "source" && f.name.ends_with(".safetensors"))
            .collect()
    };

    match format {
        PullFormat::All => patch.files.iter().collect(),
        PullFormat::Gguf => {
            let v = pick_gguf();
            if v.is_empty() {
                patch.files.iter().collect()
            } else {
                v
            }
        }
        PullFormat::Peft => {
            let v = pick_peft();
            if v.is_empty() {
                patch.files.iter().collect()
            } else {
                v
            }
        }
        PullFormat::Replacement => {
            let v = pick_replacement();
            if v.is_empty() {
                patch.files.iter().collect()
            } else {
                v
            }
        }
        PullFormat::Auto => {
            if prefer_gguf {
                let g = pick_gguf();
                if !g.is_empty() {
                    return g;
                }
            }
            if patch.kind == "replacement" {
                let r = pick_replacement();
                if !r.is_empty() {
                    return r;
                }
            }
            let p = pick_peft();
            if !p.is_empty() {
                return p;
            }
            patch.files.iter().collect()
        }
    }
}

/// Downloads and SHA-256-verifies the selected files for `patch`.
pub fn pull_patch(
    patch: &CatalogPatch,
    paths: &AppPaths,
    dest_override: Option<&Path>,
    format: PullFormat,
    prefer_gguf: bool,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<PulledPatch, ToolError> {
    let dest_dir = dest_override.map(Path::to_path_buf).unwrap_or_else(|| {
        paths
            .patches_dir()
            .join(&patch.base)
            .join(&patch.slug)
            .join(format!("v{}", patch.version))
    });
    fs::create_dir_all(&dest_dir).map_err(|e| ToolError::io(&dest_dir, e))?;

    let artifacts = select_artifacts(patch, format, prefer_gguf);
    let mut files = Vec::new();
    let mut replacement_file = None;
    let mut gguf_adapter = None;
    let mut peft_adapter = None;

    for art in artifacts {
        let out_path = dest_dir.join(&art.name);
        fetch_verified_artifact(patch, art, &out_path, on_progress)?;
        if patch.kind == "replacement" && art.role == "source" && art.name.ends_with(".safetensors")
        {
            replacement_file = Some(out_path.clone());
        } else if art.name.to_ascii_lowercase().ends_with(".gguf") {
            gguf_adapter = Some(out_path.clone());
        } else if art.name == "adapter_model.safetensors" || art.name == "adapter.safetensors" {
            peft_adapter = Some(out_path.clone());
        }
        files.push(out_path);
    }

    Ok(PulledPatch {
        dir: dest_dir,
        files,
        replacement_file,
        gguf_adapter,
        peft_adapter,
    })
}

fn fetch_verified_artifact(
    patch: &CatalogPatch,
    art: &PatchArtifact,
    dest: &Path,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<(), ToolError> {
    // Reuse already-verified local file.
    if dest.is_file() && file_matches_sha256(dest, &art.sha256)? {
        on_progress(&art.name, art.size_bytes, art.size_bytes);
        return Ok(());
    }

    let mut last_err: Option<ToolError> = None;
    for locator in &art.locators {
        let res = match locator {
            ArtifactLocator::IndexPath { path } => {
                fetch_index_path(patch, path, dest, &art.name, art.size_bytes, on_progress)
            }
            ArtifactLocator::Hf {
                repo_id,
                revision,
                path,
                ..
            } => {
                let url = format!("{}/{repo_id}/resolve/{revision}/{path}", hf_endpoint());
                download_url_to_file(&url, dest, &art.name, Some(art.size_bytes), on_progress)
            }
            ArtifactLocator::Url { url } => {
                download_url_to_file(url, dest, &art.name, Some(art.size_bytes), on_progress)
            }
            ArtifactLocator::Other => continue,
        };
        match res {
            Ok((got_sha, got_bytes)) => {
                if got_sha != art.sha256 {
                    let _ = fs::remove_file(dest);
                    last_err = Some(ToolError::DownloadIntegrity {
                        name: art.name.clone(),
                        detail: format!("expected sha256 {}, got {got_sha}", art.sha256),
                    });
                    continue;
                }
                if art.size_bytes > 0 && got_bytes != art.size_bytes {
                    let _ = fs::remove_file(dest);
                    last_err = Some(ToolError::DownloadIntegrity {
                        name: art.name.clone(),
                        detail: format!("expected {} bytes, got {got_bytes}", art.size_bytes),
                    });
                    continue;
                }
                return Ok(());
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| ToolError::NotFound {
        detail: format!("no supported locator for artifact {}", art.name),
    }))
}

fn fetch_index_path(
    patch: &CatalogPatch,
    rel_path: &str,
    dest: &Path,
    display_name: &str,
    expected_bytes: u64,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<(String, u64), ToolError> {
    let local_src = patch
        .registry_root
        .join("models")
        .join(&patch.base)
        .join(rel_path);
    if local_src.is_file() {
        return copy_and_hash(&local_src, dest, display_name, expected_bytes, on_progress);
    }
    if patch.registry_kind == RegistryKind::Http {
        let base_url = patch.registry_url.trim_end_matches('/');
        let url = match &patch.registry_pin {
            Some(pin) => format!("{base_url}/{pin}/models/{}/{rel_path}", patch.base),
            None => format!("{base_url}/models/{}/{rel_path}", patch.base),
        };
        return download_url_to_file(&url, dest, display_name, Some(expected_bytes), on_progress);
    }
    Err(ToolError::NotFound {
        detail: format!(
            "index-path artifact missing in registry cache: {}",
            local_src.display()
        ),
    })
}

/// Downloads a pinned base model profile (weight files from `base.json` plus
/// companion config/tokenizer JSON files from Hugging Face) into
/// `~/.local/share/weightspace/bases/<org>/<model>/<profile>/`.
pub fn download_base_profile(
    base_id: &str,
    profile: &BaseProfile,
    paths: &AppPaths,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<PathBuf, ToolError> {
    let dest_dir = paths.bases_dir().join(base_id).join(&profile.id);
    fs::create_dir_all(&dest_dir).map_err(|e| ToolError::io(&dest_dir, e))?;

    // 1. Download every pinned weight file and verify exact SHA-256.
    for wf in &profile.files {
        let dest = dest_dir.join(&wf.path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| ToolError::io(parent, e))?;
        }
        if dest.is_file() && file_matches_sha256(&dest, &wf.sha256)? {
            on_progress(&wf.path, wf.size_bytes, wf.size_bytes);
            continue;
        }
        let url = format!(
            "{}/{}/resolve/{}/{}",
            hf_endpoint(),
            profile.source.repo_id,
            profile.source.revision,
            wf.path
        );
        let (got_sha, got_bytes) =
            download_url_to_file(&url, &dest, &wf.path, Some(wf.size_bytes), on_progress)?;
        if got_sha != wf.sha256 {
            let _ = fs::remove_file(&dest);
            return Err(ToolError::DownloadIntegrity {
                name: wf.path.clone(),
                detail: format!("expected sha256 {}, got {got_sha}", wf.sha256),
            });
        }
        if wf.size_bytes > 0 && got_bytes != wf.size_bytes {
            let _ = fs::remove_file(&dest);
            return Err(ToolError::DownloadIntegrity {
                name: wf.path.clone(),
                detail: format!("expected {} bytes, got {got_bytes}", wf.size_bytes),
            });
        }
    }

    // 2. Best-effort fetch of companion tokenizer/config files from public HF tree API.
    let companion_files =
        list_hf_companion_files(&profile.source.repo_id, &profile.source.revision);
    for rel in companion_files {
        let dest = dest_dir.join(&rel);
        if dest.is_file() {
            continue;
        }
        let url = format!(
            "{}/{}/resolve/{}/{rel}",
            hf_endpoint(),
            profile.source.repo_id,
            profile.source.revision
        );
        let _ = download_url_to_file(&url, &dest, &rel, None, on_progress);
    }

    Ok(dest_dir)
}

fn list_hf_companion_files(repo_id: &str, revision: &str) -> Vec<String> {
    let api_url = format!("{}/api/models/{repo_id}/tree/{revision}", hf_endpoint());
    let Ok(resp) = ureq::get(&api_url)
        .set("User-Agent", "weightspace-cli/0.2")
        .call()
    else {
        return Vec::new();
    };
    let Ok(val) = resp.into_json::<serde_json::Value>() else {
        return Vec::new();
    };
    let Some(items) = val.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let Some(path) = item.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if path.contains('/') {
            continue;
        }
        let keep = matches!(
            path,
            "config.json"
                | "generation_config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "vocab.json"
                | "merges.txt"
                | "tokenizer.model"
                | "chat_template.jinja"
                | "model.safetensors.index.json"
        );
        if keep {
            out.push(path.to_owned());
        }
    }
    out
}

fn copy_and_hash(
    src: &Path,
    dest: &Path,
    display_name: &str,
    expected_bytes: u64,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<(String, u64), ToolError> {
    let mut reader = File::open(src).map_err(|e| ToolError::io(src, e))?;
    let total = reader.metadata().map(|m| m.len()).unwrap_or(expected_bytes);
    stream_to_dest(&mut reader, dest, display_name, total, on_progress)
}

pub fn download_url_to_file(
    url: &str,
    dest: &Path,
    display_name: &str,
    expected_bytes: Option<u64>,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<(String, u64), ToolError> {
    let resp = ureq::get(url)
        .set("User-Agent", "weightspace-cli/0.2")
        .call()
        .map_err(|e| ToolError::Network {
            url: url.to_owned(),
            detail: e.to_string(),
        })?;
    let total = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .or(expected_bytes)
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    stream_to_dest(&mut reader, dest, display_name, total, on_progress)
}

fn stream_to_dest(
    reader: &mut dyn Read,
    dest: &Path,
    display_name: &str,
    total: u64,
    on_progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<(String, u64), ToolError> {
    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
    ));
    let mut out = File::create(&part).map_err(|e| ToolError::io(&part, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut done = 0u64;
    on_progress(display_name, 0, total);
    loop {
        let n = reader.read(&mut buf).map_err(|e| ToolError::io(&part, e))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| ToolError::io(&part, e))?;
        hasher.update(&buf[..n]);
        done += n as u64;
        on_progress(display_name, done, total);
    }
    out.flush().map_err(|e| ToolError::io(&part, e))?;
    out.sync_all().map_err(|e| ToolError::io(&part, e))?;
    drop(out);

    #[cfg(windows)]
    {
        if dest.exists() {
            fs::remove_file(dest).map_err(|e| ToolError::io(dest, e))?;
        }
    }
    fs::rename(&part, dest).map_err(|e| ToolError::io(dest, e))?;
    let sha = Sha256Hex::from_digest(&hasher.finalize()).to_string();
    Ok((sha, done))
}

fn file_matches_sha256(path: &Path, expected: &str) -> Result<bool, ToolError> {
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let got = hash_file_with_progress(path, &cancelled, &mut |_, _| {})?;
    Ok(got.as_str().eq_ignore_ascii_case(expected))
}
