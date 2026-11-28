# Weightspace CLI & TUI Guide (`wsp` / `weightspace`)

`weightspace` — and its short alias **`wsp`** — is the official command-line tool and interactive TUI for the [Weightspace Index](../README.md). No account, API token, or authentication is required.

> Every command includes built-in examples and formulas via `wsp --help` and `wsp <command> --help` (for example, `wsp apply --help` or `wsp registry --help`).

## Contents

1. [Install](#1-install)
2. [Quick Start (Interactive TUI)](#2-quick-start-interactive-tui)
3. [Browse the Default Registry & Local Caches](#3-browse-the-default-registry--local-caches)
4. [Add Third-Party, Private, or Community Registries](#4-add-third-party-private-or-community-registries)
5. [Download a Base Model & Patches](#5-download-a-base-model--patches)
6. [Apply, Stack, and Restore Bit-Exact Replacements](#6-apply-stack-and-restore-bit-exact-replacements)
7. [Run on Your Inference Engine](#7-run-on-your-inference-engine)
8. [Authoring & Index Maintenance (Offline)](#8-authoring--index-maintenance-offline)
9. [Reference for AI Agents & Scripts](#9-reference-for-ai-agents--scripts)

---

## 1. Install

### macOS & Linux (Homebrew)

macOS binaries are Developer ID signed and Apple notarized (`source=Notarized Developer ID`).

```sh
brew install relux-works/tap/weightspace
```

### Debian & Ubuntu (Signed APT — `amd64` and `arm64`)

```sh
curl -fsSL https://apt.relux.works/relux-works.gpg \
  | sudo tee /usr/share/keyrings/relux-works.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/relux-works.gpg] https://apt.relux.works stable main" \
  | sudo tee /etc/apt/sources.list.d/relux-works.list

sudo apt update && sudo apt install weightspace
```

### Windows, Standalone Binaries, or From Source

- **Direct verified downloads** (Linux static musl, macOS ARM64/Intel, Windows x86_64, `.deb`, `SHA256SUMS` + `cosign`): [weightspaceindex.com/weightspace/](https://weightspaceindex.com/weightspace/)
- **From source (Rust 1.85+):**
  ```sh
  git clone https://github.com/relux-works/weightspace-index
  cd weightspace-index/tool
  cargo install --path . --locked
  ```

Verify the installation:

```sh
wsp --version
wsp --help
```

---

## 2. Quick Start (Interactive TUI)

Run `wsp` with no arguments inside a terminal to launch the 4-step TUI:

```sh
wsp
```

1. **Choose an inference engine** — shows which engines are on your `PATH` (`[installed]` vs `[not in PATH]` with install hints) and how many matching local base models exist for each.
2. **Pick a model & patch** — shows locally cached base models that match indexed patches **first**, followed by the rest of the catalog sorted newest-added first with `[not downloaded]`. Press `/` to filter or `r` to sync registries.
3. **Prepare & apply** — if the base model isn't downloaded yet, press `d` to fetch the pinned base profile; press `Enter` to pull and SHA-256-verify the patch/LoRA (and bake replacements into a safe managed copy, or press `i` for `--in-place` with backup).
4. **Launch or print paths** — press `1` to launch your engine with recommended flags, or `2` to print absolute paths (`BASE_PATH`, `PATCHED_MODEL`, `ADAPTER_PATH`, `COMMAND`) for your own scripts.

---

## 3. Browse the Default Registry & Local Caches

Out of the box, `wsp` uses the official `relux-works/weightspace-index` registry (`official`).

```sh
# List all patches (newest added first)
wsp search

# Filter by query, kind (`lora` | `replacement`), or base model
wsp search coding --kind lora
wsp search --base Qwen3.8-27B

# Inspect files, SHA-256 digests, license, and reported benchmarks
wsp info Qwen/Qwen3.8-27B/harness-hub-pi

# Detect installed inference engines and install hints
wsp engines

# Scan local model caches and match them to indexed patches
wsp models
wsp models --engine llama.cpp
```

### Local Model Caches Scanned Automatically

| Runtime / Cache | Default Locations |
|---|---|
| Hugging Face Hub | `$HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub` |
| `llama.cpp` | `$LLAMA_CACHE`, `~/Library/Caches/llama.cpp`, `~/.cache/llama.cpp` |
| Ollama | `$OLLAMA_MODELS`, `~/.ollama/models` |
| LM Studio | `~/.cache/lm-studio/models`, `~/.lmstudio/models` |
| `mlx-serve` | `~/.mlx-serve/models` |
| Strata | `~/Strata`, `~/.local/share/strata`, `~/Library/Application Support/Strata` |
| DwarfStar (`ds4`) | `~/ds4`, `~/Developer/ds4`, `./*.gguf` |
| Weightspace managed | `~/.local/share/weightspace/bases` |

---

## 4. Add Third-Party, Private, or Community Registries

Weightspace Index is pure Git + static JSON — never locked to GitHub. Any Git host (GitHub, corporate GitLab, Gitea, Forgejo, SSH URLs), static HTTPS directory, or local folder serving `models/index.json` works as a registry.

Git registries sync through your system `git` binary, automatically inheriting SSH keys and corporate credential helpers with zero auth stored inside `wsp`.

```sh
# Add a third-party community index (for example, an external abliterated/uncensored index)
wsp registry add community https://git.example.org/community/weightspace-index.git --sync

# Add a private corporate GitLab index pinned to a specific release tag with higher priority
wsp registry add corp ssh://git@gitlab.internal.example/ml/weightspace-index.git \
  --pin v2026.11 --priority 200 --sync

# Add a static HTTP or local directory registry
wsp registry add mirror https://patches.internal.example/index --kind http --sync
wsp registry add local-dev ./my-local-index --kind local

# Pin, unpin, switch default, inspect, or sync
wsp registry list
wsp registry pin community 8a1f3c0
wsp registry unpin community
wsp registry default community
wsp registry sync
```

Registries are stored in `~/.config/weightspace/registries.toml` (override with `WEIGHTSPACE_CONFIG`):

```toml
default = "official"

[[registries]]
name = "official"
kind = "git"
url = "https://github.com/relux-works/weightspace-index.git"
priority = 100
enabled = true

[[registries]]
name = "corp"
kind = "git"
url = "ssh://git@gitlab.internal.example/ml/weightspace-index.git"
priority = 200
pin = "v2026.11"
enabled = true
```

When two enabled registries define the same `<org>/<model>/<slug>`, higher `priority` wins (ties break in favor of `default`). Every search/info entry keeps its source registry tag.

---

## 5. Download a Base Model & Patches

Every downloaded file is streamed to a `.part` file, verified against its pinned SHA-256 digest and byte size from the index, and published atomically. Already-verified local files are reused.

```sh
# Pull only the GGUF LoRA adapter (for llama.cpp / Ollama / LM Studio)
wsp pull Qwen/Qwen3.8-27B/harness-hub-pi --format gguf

# Pull the PEFT adapter + also download the pinned base model profile from Hugging Face
wsp pull Qwen/Qwen3.8-27B/superqwen-swe-lora --format peft --with-base

# Pull a bit-exact replacement file into an explicit folder
wsp pull Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops --format replacement --out ./patches

# Pull every artifact recorded for the patch
wsp pull Qwen/Qwen3.8-27B/harness-hub-pi --format all
```

- Supported `--format` values: `auto` (default), `peft`, `gguf`, `replacement`, `all`.
- Set `HF_ENDPOINT` to use a Hugging Face mirror without modifying any index records.
- Default storage: `~/.local/share/weightspace/patches/` and `~/.local/share/weightspace/bases/` (override with `WEIGHTSPACE_DATA`).

---

## 6. Apply, Stack, and Restore Bit-Exact Replacements

`wsp apply` (and `weightspace apply`) runs **100% offline**. Before writing a single byte, it takes private input snapshots and verifies every sparse `#old` value (and any full-tensor `--expect-tensor-sha256 NAME=HEX`) against your base weights.

### Single Patch into a Fresh Copy (Default — Original Untouched)

```sh
wsp apply \
  --patch ./rikiyat-thinking-loops.safetensors \
  --model ./Qwen3.5-0.8B-bf16 \
  --out ./Qwen3.5-0.8B-patched
```

### Stacking Multiple Patches (`--patch` Repeated + `--on-conflict`)

Non-overlapping weight coordinates are always applied **bit-exactly**. When two or more replacement patches touch the same coordinate (relative to unpatched base $W_0$ with per-patch deltas $\Delta_k = W_k - W_0$), choose an explicit `--on-conflict` policy:

| `--on-conflict` | Formula on overlapping coordinates | When to use |
|---|---|---|
| `error` *(default)* | Refuses and reports overlapping entry count | Safest default; guarantees zero unintended interference |
| `add` | $W_0 + \sum_k \Delta_k$ (computed in `f64`, rounded to target dtype) | Additive delta composition — matches multi-LoRA merge math |
| `average` | $\frac{1}{m}\sum_k W_k = W_0 + \frac{1}{m}\sum_k \Delta_k$ | Balanced midpoint when two patches pull the same weights |
| `weighted` | $W_0 + \sum_k \alpha_k \Delta_k$ via `PATH:WEIGHT` | Fine-grained blending (e.g. `0.6` coding + `0.4` math) |
| `last-wins` | Bit-exact value from the last `--patch` in stack order | Explicit override of an earlier patch |
| `first-wins` | Bit-exact value from the first `--patch` in stack order | Preserve priority of the first patch |

```sh
# Stack two patches with additive delta merge into a new directory
wsp apply \
  --patch ./patch-a.safetensors \
  --patch ./patch-b.safetensors \
  --on-conflict add \
  --model ./base-model \
  --out ./base-model-stacked

# Weighted blend (0.6 * delta_A + 0.4 * delta_B)
wsp apply \
  --patch ./patch-a.safetensors:0.6 \
  --patch ./patch-b.safetensors:0.4 \
  --on-conflict weighted \
  --model ./base-model \
  --out ./base-model-blended
```

### In-Place Patching, Incremental Stacking & Verified Rollback (`--in-place` / `restore`)

Pass `--in-place` when you do not want a duplicate multi-GB model directory:

```sh
# 1. Patch in place (writes verified backup to .weightspace-backup/<base_sha256>/
#    and state to .weightspace-state.json)
wsp apply --patch ./patch-a.safetensors --model ./my-model --in-place

# 2. Stack a second patch directly on top (re-composes cleanly from the verified backup!)
wsp apply --patch ./patch-b.safetensors --on-conflict average --model ./my-model --in-place

# 3. Roll back to byte-identical original weights (verifies backup SHA-256 before restoring)
wsp restore --model ./my-model
```

> If `.weightspace-backup/` was deleted or modified on disk, `wsp restore` refuses rather than writing corrupted weights.

---

## 7. Run on Your Inference Engine

`wsp run` pulls the patch in the right format for your engine, applies replacements if a local base is ready, and either launches the engine directly or prints absolute paths (`--print-paths`).

```sh
# Launch directly (auto-detects local base model)
wsp run Qwen/Qwen3.8-27B/harness-hub-pi --engine llama.cpp

# Print absolute paths and the exact command without spawning anything
wsp run Qwen/Qwen3.8-27B/superqwen-swe-lora --engine vllm --print-paths

# Download missing pinned base automatically, bake replacement, and print paths
wsp run Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops \
  --engine mlx-serve --download-base --print-paths
```

### Supported Engines (`--engine`)

| Engine ID | Binary / Runtime | Native Format | Launch Pattern |
|---|---|---|---|
| `llama.cpp` | `llama-server`, `llama-cli` | GGUF + GGUF LoRA | `llama-server -m <base.gguf> --lora <adapter.gguf> --port 8080 -ngl 99` |
| `vllm` | `vllm` | Safetensors + PEFT LoRA | `vllm serve <base> --enable-lora --lora-modules <slug>=<dir>` |
| `transformers` | `python3` | Safetensors + PEFT | `AutoModelForCausalLM` / `PeftModel.from_pretrained(...)` |
| `ollama` | `ollama` | GGUF + Modelfile `ADAPTER` | `Modelfile` (`FROM` + `ADAPTER`) → `ollama create` / `ollama run` |
| `lm-studio` | `lms` | GGUF / MLX | `lms load <model>` + adapter path |
| `mlx-lm` | `mlx_lm.server` | Safetensors / MLX | `mlx_lm.server --model <base> --adapter-path <dir>` |
| `mlx-serve` | `mlx-serve` ([mlxserve.com](https://mlxserve.com/)) | Safetensors / MLX (no Python) | `mlx-serve serve --model-dir <dir> --port 11234` |
| `strata` | `strata-server` ([Niko1221/Strata](https://github.com/Niko1221/Strata)) | GGUF / Strata SQ | `strata-server --model <model>` |
| `ds4` | `ds4-server` ([antirez/ds4](https://github.com/antirez/ds4)) | GGUF | `ds4-server -m <model.gguf>` |

If a chosen engine is not found on `PATH`, `wsp` still downloads/applies the patch, prints the install command for that engine, and outputs the exact paths and launch command.

---

## 8. Authoring & Index Maintenance (Offline)

These commands are strictly offline and never touch the network:

```sh
# Build a minimal replacement file from base + edited safetensors checkpoints
weightspace replacement --base ./base.safetensors --edited ./edited.safetensors --out ./patch.safetensors

# Derive experimental selector-exact LoRA factors (study claims do NOT transfer to derived LoRA)
weightspace derive-lora --patch ./patch.safetensors --tensor-shapes ./shapes.json \
  --method selector-exact --factor-dtype f32 --out ./derived-lora

# Regenerate and verify every index/lookup file in a registry checkout
weightspace generate .
weightspace check .
```

See [tool/README.md](../tool/README.md) for the full low-level specification of snapshots, manifest gates, no-clobber guarantees, and derived-LoRA numerical reports.

---

## 9. Reference for AI Agents & Scripts

- **Machine-readable outputs:** pass `--json` to `wsp search`, `wsp info`, `wsp models`, `wsp engines`, and `wsp registry list`.
- **Non-interactive execution:** when stdout is not a TTY, running `wsp` without a subcommand prints help and exits `0` instead of entering raw terminal mode. Use `--print-paths` with `wsp run` to capture `BASE_PATH=`, `PATCHED_MODEL=`, `ADAPTER_PATH=`, and `COMMAND=` lines cleanly.
- **Built-in help:** `wsp --help` and `wsp <command> --help` always match the exact installed binary version.
- **Environment overrides** (useful for sandboxes & CI):
  - `WEIGHTSPACE_CONFIG` — path to `registries.toml`
  - `WEIGHTSPACE_CACHE` — registry cache root
  - `WEIGHTSPACE_DATA` — downloaded patches, bases, and managed patched models root
  - `HF_ENDPOINT` — custom Hugging Face mirror base URL
- **Live catalog endpoints:**
  - Single-file agent manual: https://weightspaceindex.com/agent.md
  - Compact directory: https://weightspaceindex.com/llms.txt
  - JSON search index: https://weightspaceindex.com/search.json
