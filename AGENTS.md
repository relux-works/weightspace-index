# Weightspace Index — Agent Guide (`AGENTS.md`)

Single-file reference for AI coding agents and maintainers working on `relux-works/weightspace-index`.

## 1. Philosophy & Scope

Weightspace Index is an open catalog of **weight patches** for open-weight models, pinned by exact base checkpoint revision, byte size, and SHA-256 digest.

- **What belongs in the index:** targeted **bit-exact replacement patches** (`kind: "replacement"`) and **native LoRA adapters** (`kind: "lora"`, PEFT safetensors and/or GGUF LoRA).
- **What does NOT belong:** full-weight fine-tunes where ~100% of weights changed (those are separate models, not patches), and **abliterated / uncensored** patches in the default Relux Works registry. Third-party / community registries are supported by `weightspace` (`wsp`) from day one for any policy an operator chooses.
- **Evidence integrity:** Author-reported scores (`claims/*.json`) are always labeled **Reported** and kept strictly separate from independent **Measured** runs (`runs/*/run.json`). Study claims **never** transfer from a replacement patch to a derived LoRA or vice versa without independent evaluation.

## 2. Repository Layout

- **Full CLI & TUI walkthrough (install, registries, downloading bases/patches, stacking `--on-conflict`, `--in-place` backups & `restore`, all 9 engines including `mlx-serve`):** see [docs/cli.md](docs/cli.md) or run `wsp --help` / `wsp <command> --help`.

```text
models/                 Authored catalog records + generated index.json / README.md
  <ns>/<model>/
    base.json           Exact base model profiles (repo_id, 40-char revision, files + SHA-256)
    <patch-slug>/
      patch.json        Title, summary, kind ("replacement" | "lora"), authors, license, tags, versions
      README.md         Patch card (human prose)
      notices/          Required license texts and attributions
      v<N>/
        version.json    target_profile, changelog, files[] (role: "source" | "derived", locators, sha256)
        files/          Optional in-repo artifacts (< 100 MB)
        claims/         Author-reported benchmark measures (optional)
        runs/           Independent measured benchmark runs (optional)
lookup/                 GENERATED client lookups (by-digest/<aa>/<sha256>.json, by-repo/<ns>/<repo>.json)
schema/                 JSON Schema Draft 2020-12 definitions (records.schema.json)
tool/                   Standalone Rust crate (`weightspace`, alias `wsp`): check, generate, replacement, derive-lora, apply, client CLI & TUI
site/                   Static Astro site deployed to Cloudflare Workers Static Assets (weightspaceindex.com)
bench/                  Python paired-seed benchmark runner (MATH-500, IFEval, etc.)
docs/                   Architecture, structure, patch-formats, workflows, benchmarking, synthetic example (docs/example/)
.github/workflows/
  checks.yml            Rust fmt, clippy, cargo test --locked, weightspace check across Linux/macOS/Windows
  deploy-site.yml       Node 24 npm ci, npm test, wrangler deploy to Cloudflare Workers on push to main & release
  release.yml           Cross-platform release binaries + site/src/data/tool-release.json update
```

## 3. Record Contract (v1) & Generated Files

Every authored JSON record carries `"contract_version": 1` and validates against `schema/records.schema.json`.

- **Allowed `locator` kinds:**
  - `{"kind": "index-path", "path": "files/..."}` (for files hosted in the repo, `< 100 MB`)
  - `{"kind": "hf", "repo_type": "model"|"dataset", "repo_id": "<ns>/<repo>", "revision": "<40-hex>", "path": "..."}`
- **Allowed `source` shapes (`tool/src/layout.rs`):**
  - `kind: "replacement"`: one or more `*.safetensors` source files.
  - `kind: "lora"`: either `["adapter_config.json", "adapter_model.safetensors"]` (PEFT) **or** a single `*.gguf` adapter. A companion GGUF LoRA derived from the PEFT source may be listed alongside with `"role": "derived"`, `"derived_from": ["<adapter_model_sha256>"]`, and `"tool": {"name": "...", "version": "..."}`.
- **Never hand-edit generated files:** `models/index.json`, `models/README.md`, `models/<ns>/<model>/index.json`, `models/<ns>/<model>/README.md`, `lookup/**`, and their mirrors under `docs/example/`. Always regenerate them with `cargo run --locked -- generate ..`.
- **Rule 5 (adoption policy):** Never delete or change file types in already-merged `v<N>/` directories; add `v<N+1>/` after adoption.
- **Hygiene rules:** No symlinks anywhere in the repository; no case-only filename collisions.
- **Note on `.gitignore`:** `.gitignore` ignores `.*/`; use `git add -f .github/...` when adding new GitHub workflow files.

## 4. Standard Verification Commands

### Rust tool & index validation (`tool/`)

```sh
cd tool
cargo run --locked -- generate ..
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- check ..
```

### Static site (`site/`)

Node `>= 22.23.4` (CI uses Node 24). Note that `site/.npmrc` sets `bin-links=false` and `ignore-scripts=true` for supply-chain safety; install or run `wrangler` from the repo root rather than via `npx` inside `site/`.

```sh
cd site
npm ci --no-audit --no-fund
npm test
npm run build            # builds site/dist/ (Node-only by default; set WS_RUN_INDEX_CHECK=1 to also run Cargo check)
npm run build:example    # builds synthetic fixture in site/dist-example/
```

Production builds refuse dirty working trees on index paths (`models`, `lookup`, `bench`, `schema`) because every download link is pinned to `HEAD`. For local preview of uncommitted edits, pass `WS_ALLOW_DIRTY=1`.

## 5. Site Design & Agent Endpoints

- Design system contract: `site/DESIGN.md` and `PRODUCT.md` (Relux product-family tokens, Inter Tight / Inter / Roboto Mono, English LTR, light & dark modes, readable without JavaScript, respects `prefers-reduced-motion`).
- Machine-readable endpoints built directly from the live index on every deploy:
  - `/llms.txt` — concise directory for LLMs & agents (`site/src/pages/llms.txt.ts`).
  - `/agent.md` — single self-contained Markdown guide with the live catalog, SHA-256 digests, CLI recipes, registry config, and PR workflow (`site/src/pages/agent.md.ts`).
  - `/search.json` — structured JSON index (`site/src/pages/search.json.ts`).

## 6. Deployment & Git Workflow

- **Production host:** `https://weightspaceindex.com/` on Cloudflare Workers Static Assets (`site/wrangler.jsonc`, worker `weightspace-index`), with `www.weightspaceindex.com` redirecting `301` → apex.
- **Auto-deploy:** `.github/workflows/deploy-site.yml` runs on every push to `main`, on successful `release` workflow completion, and via `workflow_dispatch`.
- **Landing changes:** Open a PR from a feature branch, wait for `checks` to pass, rebase onto `main` preserving original SSH-signed commits, and fast-forward push to `main`. **No merge commits.**

## 7. Quick Checklist for Adding a Patch / LoRA

1. Ensure `models/<ns>/<model>/base.json` lists the exact pinned base profile (`repo_id`, 40-char `revision`, exact weight filenames, sizes, and SHA-256 digests).
2. Create `models/<ns>/<model>/<patch-slug>/` with `patch.json`, `README.md`, `notices/<license-file>`, and `v1/version.json` (plus `v1/claims/*.json` if citing author-published benchmarks).
3. Verify every SHA-256, byte size, and pinned revision against upstream Hugging Face API metadata (`lfs.oid` for LFS blobs or direct hash for non-LFS JSON files).
4. Run `cargo run --locked -- generate .. && cargo run --locked -- check .. && cargo test --locked` from `tool/`.
5. Run `npm test && npm run build` from `site/`.
