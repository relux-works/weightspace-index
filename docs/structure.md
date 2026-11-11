# Repository structure and change rules

**Status: target layout.** This snapshot is still the scaffold: no entry, payload or run is present, and the `models/` tree, generated indexes, tools and CI described below do not exist yet. [Migration from the scaffold layout](#migration-from-the-scaffold-layout) lists the changes that bring the repository to this shape.

## Purpose and goals

The index collects open-source weight patches, identifies each one exactly, and benchmarks them as far as the budget allows. Maintainers import patches they find; anyone may submit a patch through a pull request. A client tool reads the index to find patches for models already installed on a user's machine, downloads them, and either installs them as adapters or bakes them into a copy of the model.

- **Humans** open the folder of the base model they run and see every patch for it, with a generated results table.
- **Client tools** fetch only the small index of the one model a user selected, never the whole repository.
- **Each patch has one stable name.** Updates are new versions under that name; "latest" is computed.
- **Baselines are measured once per base** and reused by every patch on that base.
- **Paths are permanent** and **each fact is authored once**; indexes and tables are generated.

## Target shape

```text
catalog.json                          GENERATED full index of everything
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
          version.json                exact identity of this version
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

Addresses are the identities. Within this repository, records refer to each other by address; a pinned index commit makes the referenced bytes immutable.

## Authored files

| File | Contains |
|---|---|
| `base.json` | Profiles of the base. Each profile has an ID, its source (`repo_type`, `repo_id`, full revision), format, precision or quantization, every file with path, SHA-256 and size, and the context fields the record contract requires. Quantized or converted forms published in other repositories are further profiles here, with their own source repository. |
| `patch.json` | Title, summary, kind (`sparse-exact`: stored values for selected weights; `lora`: a low-rank adapter; `whole`: complete replacement model files), `authors[]`, `submitter`, license, tags, and the status of each version (`active`, `deprecated`, `withdrawn`). |
| `version.json` | Version number, one-line changelog, target base profile, treatment details, `targets` (every tensor the patch changes), optional base fingerprint, and every file with locator, SHA-256 and size, `format` (`safetensors`, `gguf`, `json`) and `role` (for example `adapter`, `adapter-config`, `replacements`, `model-file`). A version may carry the same patch in several formats, such as a GGUF adapter and a PEFT adapter. |
| `run.json` | Profile and version (none for a baseline), protocol and dataset identities, settings, completion state, score, who performed it, and the outcome and resource files beside it. |
| `comparisons/<id>.json` | The baseline and patched runs compared, estimand, matched outcomes, uncertainty, or the reason the comparison is refused. |
| `assessments/<id>.json` | Intended use, decision policy, supporting comparisons, decision and reason. |
| Patch `README.md` | The card. Prose for people, linking to machine facts instead of restating them. |

Every JSON file, authored or generated, carries a `contract_version`.

## Generated files

| File | Contains |
|---|---|
| `models/<ns>/<model>/index.json` | Everything a client needs for one base: its profiles, every patch with metadata, every version with files, targets and status, the latest version, and the results summary. |
| `models/<ns>/<model>/README.md` | Table of every patch: kind, latest version, results against the shared baseline, regressions, assessment. |
| `by-digest/<aa>/<sha256>.json` | For each known base file digest: base address and profile ID. |
| `by-repo/<namespace>/<repo>.json` | For each repository that holds any profile of a base, upstream or not: base address and profile IDs. |
| `bases.json` | Every base address with its display name, for manual selection. |
| `catalog.json` | Every record in the repository with its address and computed fields: latest version, availability, results. |

Generated files are committed and never edited by hand. CI fails when they differ from a fresh build.

The latest version of a patch is its highest-numbered `active` version. Deprecating or withdrawing a version moves "latest" back to the previous active one.

## Rules

1. **Fixed depth.** No optional or variable path levels under `models/`.
2. **Base folder choice.** A patch is filed under the model it was made against. A quantized or converted form of a model is a profile of that model, not a separate base folder. A fine-tune is a model of its own and gets its own base folder.
3. **Casing.** Namespace and model folders keep the source repository's exact spelling and case. Paths that differ only in case are rejected.
4. **Names are for patches, versions are for updates.** A slug is lowercase ASCII and hyphen-separated, describes the effect, contains no author, model name or version, is unique within its base (first merged keeps it) and is never renamed. Improved bytes for the same patch are a new version under the same slug; a new slug is only for a genuinely different patch.
5. **Versions are immutable.** Versions are numbered `v1`, `v2`, … with no gaps. Merged files under `v<N>/` are never modified or deleted; runs, comparisons and assessments may be added. The one exception is a legal takedown: the affected files are removed, the version is marked `withdrawn`, and `version.json` stays. The card, `notices/` and `patch.json` stay editable, including version statuses.
6. **Existing patches change under human review.** A new version, a status change or a card edit for an existing patch is always reviewed by a human maintainer. Authors and submitter are recorded separately and may differ.
7. **Hosting is the submitter's choice.** A patch file may be hosted in this repository (under `files/`) or on Hugging Face. Files uploaded to this repository must each be under 100 MB, GitHub's per-file limit; the cap does not apply to Hugging Face-hosted files. Git LFS is not used.
8. **Safe formats only.** Weights are safetensors or GGUF; metadata is JSON. Pickle-based formats (`.bin`, `.pt`, `.pth`, `.pkl`) and any code from a source repository are refused, because loading them can execute code.
9. **Results say who measured them.** Every run records whether the index maintainers performed it under an index protocol or it is reported by the author or a third party. Generated tables show the two separately.
10. **No symlinks** anywhere in the repository.

## Submissions and review

Maintainers import patches they find; anyone else opens a pull request. Both use the same files and checks:

1. Create the patch folder or a new `v<N>/`, usually with the intake tool, and complete the card and `notices/`.
2. CI and the review agent check:
   - schema validity of every authored file;
   - SHA-256 and size of every file, and the 100 MB cap for uploads here;
   - safe formats only;
   - slug, casing, fixed depth and no symlinks;
   - no merged file under any `v<N>/` modified or deleted;
   - license and notices present;
   - generated files in sync.
3. When the checks pass, the agent approves a new patch and queues it for benchmarking. Changes to an existing patch, and anything the agent cannot decide, go to a human maintainer; an approved new version is queued for benchmarking too. Results supplied by a submitter are recorded as reported.

## Reading the index from a client

The client tool lists models installed locally, lets the user pick one, then reads only that model's index.

**1. Find installed models without the network.**

| Source | Location | What it gives |
|---|---|---|
| Ollama | `~/.ollama/models/manifests/…` (or `$OLLAMA_MODELS`) | Model name and the model layer's `sha256:` digest, which is the SHA-256 of the GGUF file. The blob is itself a GGUF, so its header keys can be read too; a model pulled as `hf.co/<ns>/<repo>` also gives the repository ID. |
| Hugging Face | `~/.cache/huggingface/hub/models--<ns>--<model>/` (or `$HF_HUB_CACHE`) | Repository ID and revision from the path; large files in `blobs/` are named by their SHA-256. |
| llama.cpp | its download cache (`$LLAMA_CACHE`) and folders the user adds | GGUF files: `general.base_model.*` header keys, or a computed SHA-256. |
| LM Studio | `~/.lmstudio/models/<publisher>/<repo>/` | Repository ID from the path. |

**2. Pin the index.** Resolve the index repository's `main` branch to a commit SHA once per session, and fetch every index file and in-repository patch file from `https://raw.githubusercontent.com/<org>/<repo>/<sha>/<path>`. All reads in a session then come from one snapshot. Generated files hold repository paths, not finished URLs, because a file cannot contain the SHA of the commit that adds it.

**3. Find the base**, stopping at the first hit; a 404 means no match at that step:

1. `by-digest/<aa>/<sha256>.json` for an exact profile match;
2. `by-repo/<namespace>/<repo>.json` from the cache path or LM Studio path;
3. the GGUF `general.base_model.0.repo_url` key, then step 2;
4. manual selection from `bases.json`, shown as unverified.

**4. Fetch `models/<ns>/<model>/index.json`** and show its patches, latest versions and results.

**5. Download and check.** In-repository files come from the pinned raw URL; Hugging Face files from `https://huggingface.co/<repo_id>/resolve/<revision>/<path>`. Every file's size and SHA-256 must match before use. The client picks files by `format` and `role` for the chosen destination.

**6. Apply.**

- **Pull a LoRA:** install the adapter for the destination runtime.
- **Bake one patch:** write a new copy of the model. A `sparse-exact` patch assigns stored values directly to the weights; it is exact only on its target profile's precision. Baking into another precision, such as a quantized GGUF, requantizes or keeps edited tensors at higher precision, and the client labels the result approximate before baking.
- **Bake several LoRAs:** order matters. The client compares `targets` and warns when patches change the same tensors. A combination has no index results unless a run for that exact combination exists.

The original model is never modified. The client records the base profile, every patch version applied, their order and the index commit SHA next to the baked copy. A `contract_version` the client does not support stops it instead of being guessed.

## Tampering and availability

- **Bytes cannot be swapped unnoticed.** Hugging Face files are pinned to a full commit revision, whose content cannot change, and every file also carries our SHA-256 and size. A replaced file fails the check. The worst a source owner can do is make a file unavailable: delete the repository, make it private or gated, or squash its history.
- **Only pinned links are shown.** The index and client never present a bare repository link or a branch URL as the download.
- **Availability is checked.** A scheduled job requests every Hugging Face locator at its pinned revision and records the result and date. An unavailable source never removes an entry.
- **Mirrors where the license allows.** A redistributable file may get a second locator for identical bytes, in this repository or in a maintainer-owned Hugging Face repository.
- **Digests do not protect the index itself.** Anyone with write access could rewrite this repository's history, digests included. Clients record the index commit SHA they used, so a rewrite shows up as a changed or missing commit; `main` should be protected against force pushes.

## Hugging Face-hosted files

1. **Every file is pinned** to a full 40-hex revision with size and SHA-256. The list covers every file needed to load or apply the patch; documentation and git metadata files are excluded.
2. **Digest sources are explicit.** A SHA-256 reported by the Hugging Face API is recorded as `advertised`. Files for which the API reports only a git SHA-1 are downloaded and hashed. A digest is `verified` only with a retained whole-file hashing record; benchmarking a file produces one.
3. **Model-card metadata prefills placement.** `base_model` and `base_model_relation` (`adapter`, `merge`, `quantized`, `finetune`) suggest the base folder and kind; a reviewer confirms them.
4. **Private repositories are refused. Gated bases are identified, not retrieved:** a gated base is recorded by its revision and file digests, and obtaining it is left to the user.
5. **Later upstream commits do not move a pin.** Changed upstream weights are a new version.

## Storage modes

| Mode | Locator | Check before use |
|---|---|---|
| In repository | `kind: index-path`, `storage: git`, relative `path` inside the declaring version folder | Pin the index commit, resolve to a regular file inside that snapshot, check SHA-256 and size. |
| Hugging Face | `kind: hf`, `repo_type`, case-preserving `repo_id`, full 40-hex `revision`, relative `path` | Build the canonical location from the fields, download, check SHA-256 and size. |

Every file needs a SHA-256 with its status and source, and a size. Multiple locators are alternatives for identical bytes. A file without a digest is not a usable artifact; record it as missing evidence instead. Paths are ASCII segments of letters, digits, `_`, `-` and `.`, not starting with `.`; absolute paths, traversal, backslashes, URL encoding and arbitrary URLs are refused. Never execute locator strings, repository hooks or remote code from the index.

Evidence hosted outside this repository, such as a Hugging Face dataset, is referenced by locator plus SHA-256, since a repository commit cannot pin it.

## Evidence states

| State | Meaning |
|---|---|
| `UNKNOWN` | Not measured or not established; give the reason and owner. Stays visible. |
| `UNSUPPORTED` | Outside a declared capability (version, route, profile, transport); refused with a reason. |
| `REFUSED` | A requested comparison cannot meet its prerequisites; the request and reason are kept. |
| `INCONCLUSIVE` | Evidence cannot settle the decision. |
| `REJECT` | A declared criterion or veto failed on adequate evidence. Never used for an untested case. |
| `SUPPORTED` | A versioned policy accepts a stated use and profile on retained comparisons. Not a general quality claim. |

Negative, refused, inconclusive and unassessed results stay visible. A later decision adds a new assessment; it never rewrites a run. Results for one profile or version never fill in for another.

## Migration from the scaffold layout

The scaffold described `models/<publisher>/<model>/patches/<fix-id>/` pages, a top-level `payloads/` tree and a single hand-edited `catalog.json` holding every record. Changes required:

1. **This document.** Done.
2. **`schema/`.** Split the record contract into schemas for `base.json`, `patch.json`, `version.json`, `run.json`, comparisons, assessments and every generated file. Replace registered-bundle and JSON-pointer references with addresses for in-repository records; keep digest references for external evidence. Update `schema/README.md`.
3. **`examples/`.** Replace `TEMPLATE.json` with a template base folder and patch folder, still excluded from every generated file.
4. **Tests.** Update the `payloads/...` locator fixtures and add tests for the new schemas and the rules above.
5. **`tools/` (new):** generator with a check mode, covering every generated file; review checks; Hugging Face intake tool that drafts a patch folder or version from a repository and revision, without credentials or running repository code; availability checker.
6. **CI (new).** Checks on every pull request; the availability checker on a schedule.
7. **`README.md`.** Browsing, client consumption and how to submit. Keep caveats in the README warning and `DISCLAIMER.md` rather than repeating them in every document.
8. **Generated files.** Build them from the empty `models/` tree.

## Open decisions

- **Tooling language:** Rust, shared with the client tool, or Go, matching the current schema tests.
- **Run IDs:** naming scheme for runs, for example benchmark, profile and date.
- **Index protocols:** which benchmarks maintainers run by default within the budget.
- **Gated patch hosting:** whether a patch file hosted in a gated Hugging Face repository is accepted.
- **Patch ownership:** whether changes to an existing patch can be approved without a human, for example when they come from its original submitter or from a list of owners.
- **Mirror location:** this repository, a maintainer-owned Hugging Face repository, or both.
- **CI platform:** presumably GitHub Actions.
