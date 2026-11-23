//! Integration tests for multi-registry management, local engine & model cache
//! discovery, zero-auth `wsp` CLI commands, and the headless TUI state machine.

mod common;

use std::fs;
use std::path::Path;
use std::process::Command;

use weightspace_tools::download::{PullFormat, pull_patch};
use weightspace_tools::engines::{
    DiscoveryRoots, EngineKind, ModelSourceKind, ModelWeightFormat, best_local_base,
    detect_engines, discover_local_models, matches_base, plan_launch,
};
use weightspace_tools::registry::{
    AppPaths, RegistryConfig, RegistryKind, load_merged_catalog, sync_registry,
};
use weightspace_tools::tui::{Screen, TuiApp};

fn test_paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/registries.toml"),
        cache_dir: root.join("cache"),
        data_dir: root.join("data"),
    }
}

#[test]
fn registry_config_crud_priority_and_kind_inference() {
    assert_eq!(
        RegistryKind::infer("git@gitlab.internal.corp:ml/weightspace-index.git"),
        RegistryKind::Git
    );
    assert_eq!(
        RegistryKind::infer("ssh://git@gitea.example.org/team/patches.git"),
        RegistryKind::Git
    );
    assert_eq!(
        RegistryKind::infer("https://github.com/org/custom-index"),
        RegistryKind::Git
    );
    assert_eq!(
        RegistryKind::infer("https://patches.example.org/static-index"),
        RegistryKind::Http
    );

    let tmp = tempfile::tempdir().unwrap();
    let paths = test_paths(tmp.path());

    let mut cfg = RegistryConfig::load(&paths).unwrap();
    assert_eq!(cfg.default, "official");
    assert_eq!(cfg.registries.len(), 1);

    cfg.add(
        "corp".into(),
        "git@gitlab.corp.example:ai/index.git".into(),
        None,
        Some("v1.2.0".into()),
        Some(200),
        true,
    )
    .unwrap();
    cfg.add(
        "community".into(),
        "https://community.example/index".into(),
        Some(RegistryKind::Http),
        None,
        Some(150),
        false,
    )
    .unwrap();
    cfg.save(&paths).unwrap();

    let reloaded = RegistryConfig::load(&paths).unwrap();
    assert_eq!(reloaded.default, "corp");
    let ordered: Vec<&str> = reloaded
        .ordered_enabled()
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    assert_eq!(ordered, vec!["corp", "community", "official"]);

    let mut mutated = reloaded;
    mutated
        .pin("community", Some("release-2026".into()))
        .unwrap();
    assert_eq!(
        mutated
            .registries
            .iter()
            .find(|r| r.name == "community")
            .unwrap()
            .pin
            .as_deref(),
        Some("release-2026")
    );
    mutated.pin("community", None).unwrap();
    mutated.set_default("official").unwrap();
    mutated.remove("corp").unwrap();
    assert_eq!(mutated.registries.len(), 2);
}

#[test]
fn multi_registry_merge_and_git_pin_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = test_paths(tmp.path());

    // Build a tiny local Git registry with two tagged commits to verify pinning.
    let git_repo = tmp.path().join("custom-git-reg");
    fs::create_dir_all(git_repo.join("models/TestOrg/Tiny-1B")).unwrap();
    fs::write(
        git_repo.join("models/index.json"),
        r#"{"contract_version":1,"bases":["TestOrg/Tiny-1B"]}"#,
    )
    .unwrap();
    fs::write(
        git_repo.join("models/TestOrg/Tiny-1B/index.json"),
        r#"{
          "contract_version": 1,
          "base": "TestOrg/Tiny-1B",
          "profiles": [],
          "patches": [
            {
              "slug": "pinned-lora",
              "title": "Pinned v1 LoRA",
              "summary": "From v1 tag",
              "kind": "lora",
              "license": "MIT",
              "tags": ["coding"],
              "latest": 1,
              "versions": [{"version": 1, "status": "active", "files": []}]
            }
          ]
        }"#,
    )
    .unwrap();

    let run_git = |args: &[&str]| {
        let out = Command::new("git")
            .current_dir(&git_repo)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    };
    run_git(&["init", "-b", "main"]);
    run_git(&["add", "."]);
    run_git(&["-c", "commit.gpgsign=false", "commit", "-m", "v1"]);
    run_git(&["-c", "tag.gpgsign=false", "tag", "v1.0"]);

    // Commit a v2 update on main so pinning to v1.0 proves pin enforcement.
    fs::write(
        git_repo.join("models/TestOrg/Tiny-1B/index.json"),
        r#"{
          "contract_version": 1,
          "base": "TestOrg/Tiny-1B",
          "profiles": [],
          "patches": [
            {
              "slug": "pinned-lora",
              "title": "Unpinned v2 LoRA",
              "summary": "From main HEAD",
              "kind": "lora",
              "license": "MIT",
              "tags": ["coding"],
              "latest": 1,
              "versions": [{"version": 1, "status": "active", "files": []}]
            }
          ]
        }"#,
    )
    .unwrap();
    run_git(&["add", "."]);
    run_git(&["-c", "commit.gpgsign=false", "commit", "-m", "v2"]);

    // Configure both the live repo (as local registry) and the pinned git registry.
    let mut cfg = RegistryConfig {
        default: "custom-git".into(),
        registries: Vec::new(),
    };
    cfg.add(
        "local-official".into(),
        common::repo().to_string_lossy().into_owned(),
        Some(RegistryKind::Local),
        None,
        Some(100),
        false,
    )
    .unwrap();
    cfg.add(
        "custom-git".into(),
        git_repo.to_string_lossy().into_owned(),
        Some(RegistryKind::Git),
        Some("v1.0".into()),
        Some(200),
        true,
    )
    .unwrap();

    let git_entry = cfg
        .registries
        .iter()
        .find(|r| r.name == "custom-git")
        .unwrap();
    sync_registry(git_entry, &paths).unwrap();

    let merged = load_merged_catalog(&cfg, &paths, None, false).unwrap();
    assert_eq!(
        merged.loaded_registries,
        vec!["custom-git", "local-official"]
    );

    // Verify pinned v1.0 tag was checked out rather than HEAD v2.
    let custom = merged.resolve_patch("TestOrg/Tiny-1B/pinned-lora").unwrap();
    assert_eq!(custom.title, "Pinned v1 LoRA");
    assert_eq!(custom.registry, "custom-git");

    // Verify official index patches are merged alongside custom registry.
    let official_patch = merged
        .resolve_patch("Qwen/Qwen3.8-27B/harness-hub-pi")
        .unwrap();
    assert_eq!(official_patch.registry, "local-official");
    assert!(official_patch.has_peft());
    assert!(official_patch.has_gguf());
}

#[test]
fn discovers_local_models_across_all_supported_engine_caches() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // 1. Hugging Face cache
    let hf_snap = root.join("hf/models--Qwen--Qwen3.8-27B/snapshots/abcdef1234");
    fs::create_dir_all(&hf_snap).unwrap();
    fs::write(hf_snap.join("model.safetensors"), b"stub").unwrap();

    // 2. llama.cpp GGUF cache
    let llama_dir = root.join("llama");
    fs::create_dir_all(&llama_dir).unwrap();
    fs::write(llama_dir.join("Qwen3.8-27B-Q4_K_M.gguf"), b"gguf").unwrap();

    // 3. Ollama manifest
    let ollama_manifest = root.join("ollama/manifests/registry.ollama.ai/library/qwen3.8/27b");
    fs::create_dir_all(ollama_manifest.parent().unwrap()).unwrap();
    fs::write(&ollama_manifest, b"{}").unwrap();

    // 4. LM Studio GGUF cache
    let lms_dir = root.join("lms/LiquidAI/LFM2.5-24B-A2B-GGUF");
    fs::create_dir_all(&lms_dir).unwrap();
    fs::write(lms_dir.join("LFM2.5-24B-A2B-Q8_0.gguf"), b"gguf").unwrap();

    // 5. Strata cache
    let strata_dir = root.join("strata");
    fs::create_dir_all(&strata_dir).unwrap();
    fs::write(strata_dir.join("TwIL-8B-Q4.gguf"), b"gguf").unwrap();

    // 6. DwarfStar (ds4) dir
    let ds4_dir = root.join("ds4");
    fs::create_dir_all(&ds4_dir).unwrap();
    fs::write(ds4_dir.join("ds4flash.gguf"), b"gguf").unwrap();

    // 7. Weightspace managed base
    let ws_base = root.join("ws-bases/Qwen/Qwen3.5-0.8B/bf16");
    fs::create_dir_all(&ws_base).unwrap();
    fs::write(ws_base.join("model.safetensors"), b"stub").unwrap();

    let roots = DiscoveryRoots {
        hf_hub_dirs: vec![root.join("hf")],
        llama_cpp_dirs: vec![llama_dir],
        ollama_dirs: vec![root.join("ollama")],
        lm_studio_dirs: vec![root.join("lms")],
        strata_dirs: vec![strata_dir],
        ds4_dirs: vec![ds4_dir],
        weightspace_bases_dir: Some(root.join("ws-bases")),
    };

    let found = discover_local_models(&roots);
    assert_eq!(found.len(), 7, "discovered: {found:?}");

    assert!(found.iter().any(|m| {
        m.source == ModelSourceKind::HuggingFace
            && m.model_id == "Qwen/Qwen3.8-27B"
            && m.has_single_safetensors
    }));
    assert!(
        found
            .iter()
            .any(|m| m.source == ModelSourceKind::LlamaCpp && m.format == ModelWeightFormat::Gguf)
    );
    assert!(
        found.iter().any(|m| m.source == ModelSourceKind::Ollama
            && m.format == ModelWeightFormat::OllamaManifest)
    );
    assert!(found.iter().any(|m| m.source == ModelSourceKind::LmStudio));
    assert!(found.iter().any(|m| m.source == ModelSourceKind::Strata));
    assert!(found.iter().any(|m| m.source == ModelSourceKind::DwarfStar));
    assert!(
        found
            .iter()
            .any(|m| m.source == ModelSourceKind::Weightspace
                && matches_base(m, "Qwen/Qwen3.5-0.8B"))
    );

    // For llama.cpp, GGUF base wins over HF safetensors for Qwen/Qwen3.8-27B.
    let best_llama = best_local_base(&found, "Qwen/Qwen3.8-27B", EngineKind::LlamaCpp).unwrap();
    assert_eq!(best_llama.format, ModelWeightFormat::Gguf);

    // For vLLM, safetensors HF base wins over GGUF for Qwen/Qwen3.8-27B.
    let best_vllm = best_local_base(&found, "Qwen/Qwen3.8-27B", EngineKind::Vllm).unwrap();
    assert_eq!(best_vllm.format, ModelWeightFormat::Safetensors);
}

#[test]
fn tui_state_machine_orders_local_bases_first_and_plans_launch() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = test_paths(tmp.path());

    let mut cfg = RegistryConfig {
        default: "local".into(),
        registries: Vec::new(),
    };
    cfg.add(
        "local".into(),
        common::repo().to_string_lossy().into_owned(),
        Some(RegistryKind::Local),
        None,
        Some(100),
        true,
    )
    .unwrap();
    let catalog = load_merged_catalog(&cfg, &paths, None, false).unwrap();

    // Only Qwen/Qwen3.5-0.8B is downloaded locally.
    let local_base = tmp
        .path()
        .join("hf/models--Qwen--Qwen3.5-0.8B/snapshots/abc");
    fs::create_dir_all(&local_base).unwrap();
    fs::write(local_base.join("model.safetensors"), b"stub").unwrap();
    let roots = DiscoveryRoots {
        hf_hub_dirs: vec![tmp.path().join("hf")],
        ..Default::default()
    };
    let locals = discover_local_models(&roots);

    let mut app = TuiApp::new(detect_engines(), catalog, locals);
    assert_eq!(app.screen, Screen::SelectEngine);

    // Step 1 -> Step 2 (choose llama.cpp)
    app.confirm_engine();
    assert_eq!(app.screen, Screen::SelectPatch);
    assert!(!app.rows.is_empty());

    // First row MUST be the locally installed Qwen/Qwen3.5-0.8B patch.
    let first_patch = &app.catalog.patches[app.rows[0].patch_index];
    assert_eq!(first_patch.base, "Qwen/Qwen3.5-0.8B");
    assert!(app.rows[0].local_base.is_some());

    // Remaining rows without local base are marked None (`[not downloaded]`).
    assert!(app.rows.iter().skip(1).any(|r| r.local_base.is_none()));

    // Filter narrows list cleanly.
    app.filter = "superqwen".into();
    app.rebuild_rows();
    assert_eq!(app.rows.len(), 2);

    // Step 2 -> Step 3 -> Step 4
    app.confirm_patch();
    assert_eq!(app.screen, Screen::Prepare);
    app.build_ready_plan();
    assert_eq!(app.screen, Screen::Ready);
    let plan = app.launch_plan.as_ref().unwrap();
    assert!(plan.display_command.contains("llama-server"));

    // Verify launch planning across additional engines (vLLM, Strata, ds4).
    let pi_patch = app
        .catalog
        .resolve_patch("Qwen/Qwen3.8-27B/harness-hub-pi")
        .unwrap();
    let vllm_plan = plan_launch(
        EngineKind::Vllm,
        pi_patch,
        Some(Path::new("/models/qwen")),
        None,
        Some(Path::new("/adapters/pi/adapter_model.safetensors")),
    );
    assert!(vllm_plan.display_command.contains("--enable-lora"));
    assert!(
        vllm_plan
            .display_command
            .contains("harness-hub-pi=/adapters/pi")
    );

    let ds4_plan = plan_launch(
        EngineKind::DwarfStar,
        pi_patch,
        Some(Path::new("/models/qwen.gguf")),
        None,
        None,
    );
    assert!(
        ds4_plan
            .display_command
            .contains("ds4-server -m /models/qwen.gguf")
    );
}

#[test]
fn pull_index_hosted_replacement_verifies_sha256_and_wsp_cli_works() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = test_paths(tmp.path());

    // Test library pull_patch against local repo's index-path hosted replacement.
    let mut cfg = RegistryConfig {
        default: "local".into(),
        registries: Vec::new(),
    };
    cfg.add(
        "local".into(),
        common::repo().to_string_lossy().into_owned(),
        Some(RegistryKind::Local),
        None,
        Some(100),
        true,
    )
    .unwrap();
    let catalog = load_merged_catalog(&cfg, &paths, None, false).unwrap();
    let patch = catalog
        .resolve_patch("Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops")
        .unwrap();
    let pulled = pull_patch(
        patch,
        &paths,
        None,
        PullFormat::Replacement,
        false,
        &mut |_, _, _| {},
    )
    .unwrap();
    let rep = pulled.replacement_file.expect("replacement pulled");
    assert!(rep.is_file());

    // Test the `wsp` binary end-to-end with isolated config/cache/data env vars.
    let wsp = env!("CARGO_BIN_EXE_wsp");
    let run_wsp = |args: &[&str]| {
        Command::new(wsp)
            .env("WEIGHTSPACE_CONFIG", &paths.config_file)
            .env("WEIGHTSPACE_CACHE", &paths.cache_dir)
            .env("WEIGHTSPACE_DATA", &paths.data_dir)
            .args(args)
            .output()
            .unwrap()
    };

    let add = run_wsp(&[
        "registry",
        "add",
        "local",
        common::repo().to_str().unwrap(),
        "--kind",
        "local",
        "--default",
    ]);
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let search = run_wsp(&["search", "harness-hub-pi", "--registry", "local"]);
    assert!(
        search.status.success(),
        "{}",
        String::from_utf8_lossy(&search.stderr)
    );
    let search_out = String::from_utf8_lossy(&search.stdout);
    assert!(
        search_out.contains("Qwen/Qwen3.8-27B/harness-hub-pi"),
        "{search_out}"
    );

    let info = run_wsp(&[
        "info",
        "Qwen/Qwen3.8-27B/harness-hub-pi",
        "--registry",
        "local",
        "--json",
    ]);
    assert!(
        info.status.success(),
        "{}",
        String::from_utf8_lossy(&info.stderr)
    );
    let info_json: serde_json::Value = serde_json::from_slice(&info.stdout).unwrap();
    assert_eq!(info_json["slug"], "harness-hub-pi");

    let engines = run_wsp(&["engines", "--json"]);
    assert!(engines.status.success());
    let eng_json: serde_json::Value = serde_json::from_slice(&engines.stdout).unwrap();
    assert_eq!(eng_json.as_array().unwrap().len(), 8);
}
