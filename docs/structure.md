# Repository structure and change rules

**Status: layout in place, no patches yet.** The schema, a synthetic example and the Rust checks follow this document. The `models/` tree is empty, and the tools, CI and most generated files do not exist yet; [Migration](#migration-from-the-scaffold-layout) lists what remains.

## Purpose and goals

The index collects open-source weight patches, identifies each one exactly, and benchmarks them as far as the budget allows. Maintainers import patches they find; anyone may submit a patch through a pull request. A client tool reads the index to find patches for models already installed on a user's machine, downloads them, and either installs them as adapters or bakes them into a copy of the model.

- **Humans** open the folder of the base model they run and see every patch for it, with a generated results table.
- **Client tools** fetch only the small index of the one model a user selected, never the whole repository.
- **Each patch has one stable name.** Updates are new versions under that name; "latest" is computed.
- **Baselines are measured once per profile** and reused by every patch on that base.
- **Paths are permanent** and **each fact is recorded once**: anything that can be read from a file or a path is generated, not authored.
- **No new conventions where one exists.** Formats and adapter behavior follow what Hugging Face, PEFT and llama.cpp already do.

## Target shape

```text
bases.json                            GENERATED list of base addresses
by-digest/<aa>/<sha256>.json          GENERATED known base file digest -> base and profile
by-repo/<namespace>/<repo>.json       GENERATED known repository -> base
schema/  tools/  docs/  NOTICE.md  DISCLAIMER.md
models/
  <hf-namespace>/                     base model namespace, exact casing
    <hf-model>/                       base model repository name, exact casing
      README.md                       GENERATED results table for this base
      index.json                      GENERATED single-model index for clients
      base.json                       profiles of this base
      runs/<run-id>/                  baseline runs of unpatched profiles
      <patch-slug>/                   one patch; the slug is its permanent name
        README.md                     the card: what it does, how to use it, caveats
        patch.json                    name-level metadata and version statuses
        notices/                      license texts and attribution
        v<N>/                         one version; merged files never change
          version.json                files of this version
          files/                      optional, patch files hosted in this repository
          runs/<run-id>/              runs of this version
          comparisons/<id>.json       patched run against baseline run
          assessments/<id>.json       decision for a stated use, citing comparisons
```

`<aa>` is the first two hex characters of the digest, which keeps every directory small enough for GitHub's file browser.

| Object | Path pattern | Address |
|---|---|---|
| Base | `models/*/*/base.json` | `<hf-namespace>/<hf-model>` |
| Patch | `models/*/*/*/patch.json` | `<hf-namespace>/<hf-model>/<patch-slug>` |
| Version | `models/*/*/*/v*/version.json` | patch address + `@v<N>` |
| Baseline run | `models/*/*/runs/*/run.json` | base address + `/runs/<run-id>` |
| Version run | `models/*/*/*/v*/runs/*/run.json` | version address + `/runs/<run-id>` |

Addresses are the identities. Records refer to each other by address; a pinned index commit makes the referenced bytes immutable.

## Authored files

| File | Contains |
|---|---|
| `base.json` | Profiles of the base. Each profile has an ID, its source (`repo_id` and full revision of a Hugging Face model repository), and every file with path, SHA-256 and size. Every distinct file set is its own profile: another quantization, another publisher's file with the same quantization name, or a later upstream revision with changed weights. |
| `patch.json` | Title, summary, kind (`replacement` or `lora`, see [Patch forms](#patch-forms)), `authors[]`, `submitter`, license, tags, and the status of each version (`active`, `deprecated`, `withdrawn`). |
| `version.json` | One-line changelog, target base profile, and every file with its `name`, locators, SHA-256, size and `role` (`source` or `derived`). A derived file also records `derived_from` (file digests) and the tool name and version that built it. |
| `run.json` | Profile and version (none for a baseline), protocol and dataset identities, settings, completion state, score, who performed it, and the outcome and resource files beside it. |
| `comparisons/<id>.json` | The baseline and patched runs compared, estimand, matched outcomes, uncertainty, or the reason the comparison is refused. |
| `assessments/<id>.json` | Intended use, decision policy, supporting comparisons, decision and reason. |
| Patch `README.md` | The card. Prose for people, linking to machine facts instead of restating them. |
| `notices/` | The patch's license and attribution, plus the obligations the base model's license passes on to derivatives. |

Every JSON file, authored or generated, carries a `contract_version`. [`schema/records.schema.json`](../schema/records.schema.json) defines each one; the file's path decides which definition applies.

## Patch forms

Every version has exactly one **source**: the files the author produced, in one of two forms. Nothing else is accepted as a source.

| Form | `kind` | Files | Applying it |
|---|---|---|---|
| Replacement | `replacement` | One or more `.safetensors` files keyed by the base's own tensor names, in the target profile's dtype. Each changed tensor is either the full new tensor under its own name, or sparse: `<name>#idx` (sorted, unique flat `int64` indices), `<name>#new` and `<name>#old` (values in the tensor's dtype). | Assignment into a copy of the target profile, which is always unquantized. Bit-exact. |
| LoRA | `lora` | As published: PEFT (`adapter_model.safetensors` and `adapter_config.json`) or a GGUF LoRA. | As the runtime does it: loaded as an adapter, or merged with the runtime's own tool. |

- **A complete model is never a patch.** A fine-tune or merge publishing every tensor is a base of its own.
- **`#old` records what the patch replaced.** It lets CI build and check derived files without downloading the base.
- **Derived files are optional conveniences** built by the index's own tool, never submitted: a LoRA built from a replacement and GGUF conversions of a PEFT LoRA, made with llama.cpp's own converter. A LoRA built from a replacement is exact up to runtime rounding and stays small only when the edits touch few rows or columns; a replacement of whole tensors gets none. CI rebuilds every derived file and compares digests.
- **A new form is a change to this document**, not something a submission can introduce.

**Combining patches follows the runtimes.** Several LoRAs combine the way llama.cpp loads and merges them: the scaled sum of their outputs, independent of order. The client passes the adapters and their scales to the runtime and adds no rule of its own. Replacements have no runtime convention, so a baked copy takes at most one replacement; the copy is then an ordinary model, and LoRAs go on top of it.

## Generated files

| File | Contains |
|---|---|
| `models/<ns>/<model>/index.json` | Everything a client needs for one base: its profiles, every patch with metadata, every version with its number, files and status, the tensors each version changes and each file's format (read from the files), the latest version, and the results summary. |
| `models/<ns>/<model>/README.md` | Table of every patch: kind, latest version, results per profile against that profile's baseline, regressions, assessment. Profiles without a run show "not measured". |
| `by-digest/<aa>/<sha256>.json` | For each known base file digest: base address and profile ID. |
| `by-repo/<namespace>/<repo>.json` | For each repository that holds any profile of a base, upstream or not, and for each former name of a renamed repository: base address and profile IDs. |
| `bases.json` | Every base address. Its `index.json` and `README.md` sit at `models/<address>/`. |

Generated files are committed and never edited by hand. CI fails when they differ from a fresh build.

The latest version of a patch is its highest-numbered `active` version. Deprecating or withdrawing a version moves "latest" back to the previous active one.

## Rules

1. **Fixed depth.** No optional or variable path levels under `models/`.
2. **Base folder choice.** A patch is filed under the model it was made against, which must have a Hugging Face repository. Same weights in another encoding are a profile of that model, never a folder: GGUF quants, FP8 or AWQ made after training, converted formats, whoever publishes them. A later revision of the same repository is another profile too. Different weights in another repository are a model of their own with their own base folder: fine-tunes, merges and quantization-aware trained (QAT) releases. A QAT folder is named after the unquantized QAT repository when one is published, otherwise after the quantized one.
3. **Casing and renames.** Namespace and model folders keep the source repository's exact spelling and case at the time of filing. Paths that differ only in case are rejected. A repository renamed later keeps its folder; `by-repo/` maps the new name.
4. **Names are for patches, versions are for updates.** A slug is lowercase ASCII and hyphen-separated, describes the effect, contains no author, model name or version, is unique within its base (first merged keeps it) and is never renamed. Improved bytes for the same patch are a new version under the same slug; a new slug is only for a genuinely different patch.
5. **Versions are immutable.** Versions are numbered `v1`, `v2`, … with no gaps. Merged files under `v<N>/` are never modified or deleted; runs, comparisons and assessments may be added. The one exception is a legal takedown: the affected files are removed, the version is marked `withdrawn`, and `version.json` stays. The card, `notices/` and `patch.json` stay editable, including version statuses.
6. **Existing patches change under human review.** A new version, a status change or a card edit for an existing patch is always reviewed by a human maintainer. Authors and submitter are recorded separately and may differ.
7. **Hosting is the submitter's choice.** A patch file may be hosted in this repository (under `files/`) or on Hugging Face. Files uploaded to this repository must each be under 100 MB, GitHub's per-file limit; the cap does not apply to Hugging Face-hosted files. Git LFS is not used.
8. **Safe formats only.** Weights are safetensors or GGUF; metadata is JSON. Pickle-based formats (`.bin`, `.pt`, `.pth`, `.pkl`) and any code from a source repository are refused, because loading them can execute code.
9. **Results say who measured them.** Every run records whether the index maintainers performed it under an index protocol or it is reported by the author or a third party. Generated tables show the two separately.
10. **No symlinks** anywhere in the repository.

## Where files live

| Locator | Fields | Download URL |
|---|---|---|
| In repository | `kind: index-path`, `path` relative to the folder of the declaring file; under `files/` for a version | `https://raw.githubusercontent.com/<org>/<repo>/<index-commit>/<folder>/<path>` |
| Hugging Face | `kind: hf`, `repo_type`, case-preserving `repo_id`, full 40-hex `revision`, relative `path` | `https://huggingface.co/<repo_id>/resolve/<revision>/<path>` |

A file may have several locators for identical bytes; clients try Hugging Face first. Paths are ASCII segments of letters, digits, `_`, `-` and `.`, not starting with `.`; absolute paths, traversal, backslashes, URL encoding and arbitrary URLs are refused. Never execute locator strings, repository hooks or remote code from the index.

**Every digest is our own.** The intake tool downloads every file, checks that it is valid safetensors, GGUF or JSON, reads its tensor names, and computes its SHA-256 and size. Nothing is copied from Hugging Face's API or from the submitter. CI repeats this for every file a pull request adds.

| Attack | What stops it |
|---|---|
| A Hugging Face owner replaces a file | Pinned revision, plus our own SHA-256: a mismatch is refused |
| A pull request claims a false digest | CI recomputes it from the downloaded bytes |
| A later pull request points a merged version at new bytes | Rule 5, enforced by CI |
| This repository's history is rewritten | `main` is protected against force pushes; clients record the index commit they used |

The worst a source owner can do is make a file unavailable: delete the repository, make it private or gated, or squash its history. A scheduled job requests every Hugging Face locator at its pinned revision and records the result and date; an unavailable source never removes an entry. Where the license allows, maintainers mirror files to a maintainer-owned Hugging Face repository as a second locator. Later upstream commits never move a pin; changed upstream weights are a new version.

**Hugging Face model cards prefill placement.** `base_model` and `base_model_relation` (`adapter`, `merge`, `quantized`, `finetune`) suggest the base folder and kind; a reviewer confirms them. `quantized` usually means a profile of the parent, except for QAT releases. Private repositories are refused. A gated base is recorded by its revision and file digests; obtaining it is left to the user.

## Submissions and review

Maintainers import patches they find; anyone else opens a pull request. Both use the same files and checks:

1. Create the patch folder or a new `v<N>/`, usually with the intake tool, and complete the card and `notices/`.
2. CI checks:
   - schema validity of every authored file;
   - every file downloaded, its format, SHA-256 and size, and the 100 MB cap for uploads here;
   - exactly one source form per version, and every derived file rebuilt to the same digest;
   - slug, casing, fixed depth and no symlinks;
   - no merged file under any `v<N>/` modified or deleted;
   - license and notices present;
   - generated files in sync.
3. The review agent approves a new patch only when every CI check has passed, and queues it for benchmarking. Changes to an existing patch, and anything the agent cannot decide, go to a human maintainer; an approved new version is queued for benchmarking too. Results supplied by a submitter are recorded as reported.

**The review agent reads untrusted text.** Cards, metadata, file contents and commit messages in a pull request are data, never instructions. The agent holds no secrets and can do nothing beyond approving, and its approval cannot override a failed check.

## Reading the index from a client

The client tool lists models installed locally, lets the user pick one, then reads only that model's index.

**1. Find installed models without the network.**

| Source | Location | What it gives |
|---|---|---|
| Ollama | `~/.ollama/models/manifests/…` (or `$OLLAMA_MODELS`) | Model name and the model layer's `sha256:` digest, which is the SHA-256 of the GGUF file. The blob is itself a GGUF, so its header keys can be read too; a model pulled as `hf.co/<ns>/<repo>` also gives the repository ID. |
| Hugging Face | `~/.cache/huggingface/hub/models--<ns>--<model>/` (or `$HF_HUB_CACHE`) | Repository ID and revision from the path; large files in `blobs/` are named by their SHA-256. |
| llama.cpp | its download cache (`$LLAMA_CACHE`) and folders the user adds | GGUF files: `general.base_model.*` header keys, or a computed SHA-256. |
| LM Studio | `~/.lmstudio/models/<publisher>/<repo>/` | Repository ID from the path. |

**2. Pin the index.** Resolve the index repository's `main` branch to a commit SHA once per session and read every index file at that commit. Generated files hold repository paths, not finished URLs, because a file cannot contain the SHA of the commit that adds it.

**3. Find the base**, stopping at the first hit; a 404 means no match at that step:

1. `by-digest/<aa>/<sha256>.json` for an exact profile match;
2. `by-repo/<namespace>/<repo>.json` from the cache path or LM Studio path;
3. the GGUF `general.base_model.0.repo_url` key, then step 2;
4. manual selection from `bases.json`, shown as unverified.

**4. Fetch `models/<ns>/<model>/index.json`** and show its patches, latest versions and results for the user's profile.

**5. Download and check.** Every file's size and SHA-256 must match before use.

**6. Apply.** The original model is never modified.

- **Bake a replacement:** only into a copy of a safetensors model whose files match the target profile's digests. The copy is bit-exact.
- **Add LoRAs:** on any profile, including GGUF quants in Ollama, llama.cpp or LM Studio, use a LoRA: the source, or a derived one. The client installs it for the runtime, or merges it with the runtime's own tool, such as llama.cpp's `llama-export-lora`. The client never quantizes, dequantizes or converts weights itself.
- A patch with no route for the user's profile is shown as unavailable for it.

The client records the base profile, every patch version applied and the index commit next to the result, which also lets it report newer versions later. A `contract_version` the client does not support stops it instead of being guessed. A combination has no index results unless a run for that exact combination exists.

## Evidence states

| State | Meaning |
|---|---|
| `UNKNOWN` | Not measured or not established; give the reason and owner. Stays visible. |
| `UNSUPPORTED` | Outside a declared capability (version, route, profile, transport); refused with a reason. |
| `REFUSED` | A requested comparison cannot meet its prerequisites; the request and reason are kept. |
| `INCONCLUSIVE` | Evidence cannot settle the decision. |
| `REJECT` | A declared criterion or veto failed on adequate evidence. Never used for an untested case. |
| `SUPPORTED` | A versioned policy accepts a stated use and profile on retained comparisons. Not a general quality claim. |

Negative, refused, inconclusive and unassessed results stay visible. A later decision adds a new assessment; it never rewrites a run. Results for one profile or version never fill in for another. Evidence hosted outside this repository, such as a Hugging Face dataset, is referenced by locator plus SHA-256.

## Migration from the scaffold layout

The scaffold described `models/<publisher>/<model>/patches/<fix-id>/` pages, a top-level `payloads/` tree and a single hand-edited `catalog.json` holding every record.

Done:

1. **This document.**
2. **`schema/`.** One schema, `records.schema.json`, with a definition per file type; bundles, JSON pointers and digest status are gone.
3. **`examples/`.** A synthetic base with a replacement patch and a LoRA patch replaces `TEMPLATE.json`; generated files never read it.
4. **Checks, in the Rust crate under `tools/`.** Schema validation, the layout rules a single file cannot express, hosted-file digests and the replacement file format, tested against the example and broken copies of it.
5. **Docs.** `README.md`, `docs/patch-formats.md`, `docs/workflows.md`, `schema/README.md`.
6. **`catalog.json`.** Removed; `bases.json` (empty) and the per-base indexes replace it.

Remaining:

7. **`tools/`, built on those checks:** intake tool that downloads, checks and hashes every file and drafts a patch folder or version, without credentials or running repository code; generator with a check mode, covering every generated file; builder for derived patch files; availability checker.
8. **CI (new).** Checks on every pull request, including that no merged file under `v<N>/` changed; the availability checker on a schedule.
9. **Generated files.** Built by the generator once `models/` has a base.

## Open decisions

- **Acceptance policy:** what the index refuses, for example refusal-removal patches, hidden behavior, or cards that misdescribe a patch.
- **Gated and restricted bases:** `#old` values and replacement tensors are excerpts of the base weights; whether replacements are accepted for gated bases or bases whose license restricts redistribution, and whether patch files may be hosted in gated repositories.
- **Run and profile IDs:** naming scheme for runs (for example benchmark, profile and date) and for profiles (for example publisher and quantization, such as `bartowski-q4-k-m`).
- **Index protocols:** which benchmarks maintainers run by default within the budget.
- **Patch ownership:** whether changes to an existing patch can be approved without a human, for example when they come from its original submitter or from a list of owners.
- **CI platform:** presumably GitHub Actions.
