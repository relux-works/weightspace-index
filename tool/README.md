# weightspace

`weightspace` is the command-line tool for the [Weightspace Index](../README.md). It builds and applies **exact replacement patches** for open models, derives experimental LoRA factors, and runs the index's offline checks and generator. This page is the user guide for the tool; the index itself — what is filed, where, and under what rules — is described in the [root README](../README.md) and [repository structure](../docs/structure.md).

> [!WARNING]
> A patched model may behave unexpectedly or unsafely. This tool checks bytes and tensor identities; it cannot tell you whether a patch does what its description claims. Inclusion in the index, a successful `weightspace` run, and published benchmark results do not prove that a third-party patch is safe or effective. Read the patch's card and evidence, and assess it for your intended use. See [DISCLAIMER.md](../DISCLAIMER.md).

## What it does, in plain language

A model checkpoint is a large file of named tensors (arrays of numbers). A **replacement patch** says: "for this exact base model, replace the values of these tensors with these new values." Applying it is plain assignment — the new values are copied bit-for-bit into a copy of the model — so the result is exact, with no rounding. For each changed tensor the tool picks whichever storage — in full, or sparsely as the changed positions plus the old and new values at those positions — makes the serialized patch file smaller, storing in full when the sizes tie, so a tensor changed in only a few places can still be stored in full. See [patch formats](../docs/patch-formats.md) for the file-level details.

`weightspace` (short alias **`wsp`**) provides two clean layers:

### Client CLI, Multi-Registry & Interactive TUI (`wsp`)

- `wsp` / `wsp tui` — interactive terminal UI: choose your inference engine (`llama.cpp`, `vLLM`, `Transformers / PEFT`, `Ollama`, `LM Studio`, `MLX-LM`, `Strata`, `DwarfStar / ds4`), see locally downloaded base models matching indexed patches first (followed by the rest of the catalog sorted newest-added first with `[not downloaded]`), optionally download a missing pinned base, pull & SHA-256-verify the patch/LoRA, apply replacements, and either launch the engine with recommended flags or print absolute paths.
- `wsp search [QUERY]` / `wsp info <BASE/SLUG>` — browse and inspect patches across all configured registries (no auth required).
- `wsp pull <BASE/SLUG> [--with-base] [--format auto|peft|gguf|replacement|all]` — download and SHA-256-verify artifacts (and optionally the pinned base profile from Hugging Face).
- `wsp models` / `wsp engines` — discover locally cached models (Hugging Face hub, `llama.cpp`, Ollama, LM Studio, Strata, `ds4`, Weightspace managed) and installed engines on `PATH`.
- `wsp run <BASE/SLUG> [--engine <NAME>] [--print-paths] [--download-base]` — pull, apply if replacement, and launch the chosen engine (or print absolute paths and the exact command).
- `wsp registry <list|add|remove|pin|unpin|default|sync>` — manage `~/.config/weightspace/registries.toml` with the official default registry plus any custom Git host (GitHub, corporate GitLab, Gitea, SSH URLs via system `git`), static HTTP index, or local directory, with commit/tag pinning and priority ordering.

### Offline Patch & Index Commands (`weightspace`)

Everything below runs **strictly offline** — zero network, zero checkpoint code execution:

- `weightspace replacement` — compare a base checkpoint with an edited checkpoint and write one replacement patch file (`.safetensors`) holding only what changed. This is for patch authors.
- `weightspace apply` — bake one or more replacement patch files into a **new copy** of a model directory (default), or modify `--in-place` after writing a SHA-256-verified backup under `.weightspace-backup/`. Supports stacking multiple `--patch` files with `--on-conflict=error|add|average|weighted|last-wins|first-wins` (`PATH:WEIGHT` for weighted delta merge).
- `weightspace restore --model <DIR>` — verify backup SHA-256 and restore exact original base weights from an `--in-place` apply.
- `weightspace derive-lora` — derive experimental LoRA factors from an exact replacement and matching base, or offline from sparse old values and supplied matrix shapes. Study claims NEVER transfer to this derived form; it needs its own evaluation. PEFT/llama.cpp/vLLM runtime loading is UNQUALIFIED.
- `weightspace check [repo]` — run every offline check over an index repository checkout. This is for contributors and CI.
- `weightspace generate [repo]` — rebuild the index's generated files in a checkout. This is for contributors.

## What it does not do

- It does **not** train or fine-tune models or run benchmarks. Inference engines (`llama.cpp`, `vLLM`, `Ollama`, `LM Studio`, `MLX-LM`, `Strata`, `ds4`) are launched externally or given printed paths; [workflows](../docs/workflows.md) shows where common runtimes fit.
- It does **not** prove patch quality, safety, or effectiveness, and a clean run is not an endorsement (see the warning above).
- It does **not** quantize, dequantize, or convert weights, and it does not read pickle-based formats (`.bin`, `.pt`, `.pth`, `.pkl`). Replacement checkpoints and patch files must be single-file [safetensors](https://github.com/huggingface/safetensors); sharded or indexed layouts are refused for `apply` (see [Inputs in detail](#inputs-in-detail)).

## Install

### From a release archive

Tagged releases are published under **GitHub Releases** for this repository, one archive per platform, where `<tag>` is the release tag including its leading `v` (for example, `v0.1.0-alpha.1`). A release contains the tool as of its tag; `v0.1.0-alpha.1` is tool version 0.1.0 and predates `derive-lora`. For newer commands, install [from source](#from-source).

| Platform | Archive |
|---|---|
| Linux, x86_64 | `weightspace-<tag>-x86_64-unknown-linux-musl.tar.gz` |
| macOS, Apple Silicon (arm64) | `weightspace-<tag>-aarch64-apple-darwin.tar.gz` |
| Windows, x86_64 | `weightspace-<tag>-x86_64-pc-windows-msvc.zip` |

Releases after `v0.1.0-alpha.1` also publish each binary unpacked, ready to run: `weightspace-<tag>-<target>` (`weightspace-<tag>-<target>.exe` on Windows), listed in the same `SHA256SUMS`. Check its SHA-256, make it executable (`chmod +x` on Linux and macOS), and put it on your `PATH` under the name `weightspace`.

Each archive contains a single top-level directory, `weightspace-<tag>-<target>/`, holding the `weightspace` binary (`weightspace.exe` on Windows) and its `LICENSE`, and each release publishes a combined `SHA256SUMS` file. To install one: download the archive and `SHA256SUMS`, verify the archive's SHA-256 before extracting it (stop on any mismatch), extract it, and put the binary from that directory in a directory already on your `PATH`. For any other platform or architecture, build from source.

### From source

You need Rust **1.85 or newer** on Linux, macOS, or Windows. From `tool/` in the repository:

```sh
cd tool
cargo build --locked --release --bin weightspace
```

`--locked` builds exactly the dependency versions recorded in `tool/Cargo.lock`. The binary is written to `tool/target/release/weightspace` (`tool\target\release\weightspace.exe` on Windows), relative to the repository root; run it from there or copy it to a directory on your `PATH`. Confirm it runs:

```sh
weightspace --help
```

Every command also accepts `-h`/`--help`, for example `weightspace apply --help`. There is no `--version` flag; the source version is the `version` in `tool/Cargo.toml`.

## Quick start

### Build a replacement patch (patch authors)

Given the original checkpoint and your edited checkpoint — both single-file safetensors with exactly the same tensor names, shapes, and dtypes — write the patch file:

```sh
weightspace replacement \
  --base ./base.safetensors \
  --edited ./edited.safetensors \
  --out ./my-patch.safetensors
```

On success it prints one line, for example `replacement written: ./my-patch.safetensors (2 tensors, 5 entries, 1234 bytes, sha256 …)`, where "entries" counts sparse changed positions plus the element counts of full tensors. If base and edited are identical, it refuses with `base and edited are identical; no replacement to write` — that is a normal outcome, not a damaged input.

To also gate the build on an authors' manifest of expected per-tensor hashes, add `--manifest ./manifest.json` (format under [Inputs in detail](#inputs-in-detail)). To control where the private input snapshots are made, add `--temp-dir <dir>` (see [Space, snapshots, and outputs](#space-snapshots-and-outputs)).

### Apply a replacement patch (patch users)

Given a patch file and a model directory holding a single root `model.safetensors`, bake the patch into a new directory:

```sh
weightspace apply \
  --patch ./my-patch.safetensors \
  --model ./my-model \
  --out ./my-model-patched
```

On success it prints `patched model written: ./my-model-patched (… tensors, … entries, … bytes, sha256 …)`; for `apply`, the byte count and SHA-256 describe the patched `model.safetensors` inside the new directory. `./my-model-patched` is an ordinary model directory — run it with your runtime as usual ([workflows](../docs/workflows.md)).

If the patch stores any tensor **in full**, you must also supply the expected base SHA-256 for **every** full-form tensor, one flag each, because the patch file does not record the bytes it replaces:

```sh
weightspace apply \
  --patch ./my-patch.safetensors \
  --model ./my-model \
  --out ./my-model-patched \
  --expect-tensor-sha256 model.layers.0.mlp.down_proj.weight=<64-hex-digit-sha256> \
  --expect-tensor-sha256 model.embed_tokens.weight=<64-hex-digit-sha256>
```

If one is missing, the refusal names the tensor that needs one. Where those digests should come from — and why computing them from the model in front of you defeats the check — is covered under [Inputs in detail](#inputs-in-detail).

### Derive experimental LoRA factors

**LoRA is a derived form for experimentation/combination; study claims NEVER transfer to it. It is a separate treatment needing its own evaluation. PEFT/llama.cpp/vLLM runtime loading is UNQUALIFIED.** The adapter is not asserted to be ready to load in any runtime.

```sh
weightspace derive-lora \
  --model ./my-model \
  --patch ./replacements.safetensors \
  --rank 2 --factor-dtype f32 --method svd \
  --expect-tensor-sha256 linear.weight=<64-hex-digit-base-tensor-sha256> \
  --temp-dir ./scratch --out ./derived-lora
```

Create `./scratch` first. Supply a trusted expected base digest for **each full-form** patch tensor, exactly as for `apply`; omit the flag for sparse tensors, whose saved old bytes are verified. The model uses the same single root `model.safetensors` profile as `apply`. Every patched tensor must be a nonempty 2-D BF16 matrix. Other tensors are not derived. Nonfinite base/patched values are refused. No inputs are downloaded, and no inference or benchmark is run.

For the existing explicit-rank modes, all four choices are explicit: `--rank` (positive), `--factor-dtype f32` (the only supported factor dtype), `--method svd|selector`, and `--out` (a **new** directory, refused if it already exists, even empty). The output parent must exist and be outside the input model. Input snapshots use `--temp-dir` or the system temporary directory. Successful output contains `adapter.safetensors` and `lora-report.json`; neither contains input paths or timestamps.

For each patch tensor, the tool reuses apply's snapshot reader and base verification, reconstructs its exact replacement values, expands BF16 exactly to F64, then subtracts in F64. An error-free residual check refuses a dyadic difference that F64 cannot represent exactly. A full stored-bit comparison scans every column; active column indices are sorted ascending, including columns changed only in their signed-zero bit pattern. Omitted columns have zero delta.

`svd` computes the thin SVD of the active-column matrix in F64 with a std-only cyclic one-sided Jacobi implementation. Wide matrices are transposed for the solve. Pairs run in fixed order; a sweep converges when every column cross product is at most `8 * f64::EPSILON * norm(p) * norm(q)`. Failure to converge in 256 sweeps is a refusal. Singular values are sorted descending, ties by original Jacobi column. Each U column's largest-magnitude element is positive, ties by lowest row. The top `rank` components produce balanced `B=U*sqrt(S)` and `A=sqrt(S)*Vᵀ`, with A zero-expanded to the original width, then each factor is separately cast to F32. Rank must not exceed `min(rows,k)` for SVD, where `k` is the number of active columns.

`selector` requires `rank=k` and builds `B=exact_delta[:,active_columns]`, with one-hot A rows selecting the original columns, before the F32 cast. Its real-algebra product is exact; storage and bake fidelity are measured rather than assumed. Rank exceeding k is always refused. All-zero numerical deltas can still have active signed-zero columns; relative error is `null` when the exact delta norm is zero.

Names are mapped **exactly** by appending literal `.lora_A` and `.lora_B` to the complete original UTF-8 tensor name. A is row-major `[rank,width]`, B is row-major `[rows,rank]`; no name rewriting, prefix stripping, transpose or target-module inference occurs. This suffix mapping is injective across tensor/factor pairs. The effective scale is **1**, recorded in the report; no runtime config is emitted. A runtime using alpha/r would need alpha=rank and user scale 1, plus separately qualified mapping and kernels.

The report includes used and all computed active singular values, structural zero count, active indices and k, Jacobi sweeps, exact-rank evidence, factor names/shapes and raw factor byte counts. Independent of SVD, exact BF16 differences scaled by `2^133` are eliminated modulo `1,000,000,007` and `1,000,000,009`. Each modular rank is a lower bound; equality to `min(rows,k)` proves exact rank. A smaller lower bound does **not** prove deficiency; `proven_exact_rank` is then `null`.

Errors are measured from the **stored F32 factors** widened to F64. Multiplication and accumulation proceed in increasing rank order in F64. The frozen bake adds that reconstructed delta to the exact BF16 base in F64, casts to F32, then rounds to BF16 round-to-nearest-ties-even. This is a fixed arithmetic experiment, not a runtime-kernel simulation. Nonfinite stored factors or overflow in the bake are refused.

- `delta_frobenius_error`: Frobenius norm of reconstructed delta minus exact delta; `relative_delta_error` divides this by `exact_delta_frobenius_norm` (null for zero denominator).
- `bf16_mismatch_count`: stored-bit mismatches against the exact patched tensor after the frozen bake.
- `max_ulp`, `mean_ulp_all_elements`, `mean_ulp_active_columns`: distances between monotone BF16 encodings (`!bits` for negative values, `bits | 0x8000` otherwise). Signed zero patterns remain distinct; active-column means include every row of each active column.
- `baked_frobenius_error`: Frobenius norm of baked BF16 values minus exact patched BF16 values, expanded to F64.
- `raw_factor_bytes`: A, B and combined bytes, excluding safetensors headers and the report.

Identical immutable inputs/options in the same numerical environment produce byte-identical outputs. No timestamps or randomized decomposition are used. Cross-platform/library/compiler bit identity, runtime-kernel fidelity and efficacy are unqualified; the sign convention does not uniquely identify all degenerate singular subspaces. Preserve exact generated bytes and evaluate the derived treatment separately. Study claims NEVER transfer to LoRA.

This analysis materializes one patched tensor and its active matrix in RAM, plus factors for all output tensors. Unlike streaming replacement/apply, it needs memory proportional to the analyzed matrices (`O(rows*width + rows*k + k²)` per tensor, plus stored factors); snapshots additionally need disk space. No derived index record or runtime config is written automatically. The following per-tensor mode is the canonical publication route.

### Publish a per-tensor selector without a global rank

`selector-exact` in tool version 0.2.0 chooses rank `k_t` separately for each tensor. Omit `--rank`; the existing explicit-rank `svd` and `selector` modes retain their arithmetic and refusal behavior. No SVD or modular-rank calculation is needed for this construction. Dense and whole-tensor matrices are supported, even when their factors are larger than the replacement. There is no sparsity, usefulness or size refusal in conversion.

With a model directory, use `--model ./model` and expected base digests for every full-form tensor, as above. For sparse-only patches, no base download is required:

```sh
weightspace derive-lora --patch ./replacements.safetensors \
  --tensor-shapes ./tensor-shapes.json \
  --method selector-exact --factor-dtype f32 --out ./derived-lora
```

The shape file is a JSON object naming exactly all patch tensors, for example `{"linear.weight": [1024, 3584]}`. These are trusted supplied original shapes, not inferred from the largest sparse index. Obtain them from pinned tensor-header evidence; the report embeds the shapes and input-file SHA-256. Indices must fit the declared dimensions. `--tensor-shapes` and `--model` are mutually exclusive; offline shapes cannot verify expected base digests. Only selector-exact supports offline input. The parent of `--out` must already exist; the output directory must be new.

A is a one-hot column selector, B contains the active delta columns, both stored as F32. Zero active columns give rank zero and zero-length factors, retaining the tensor in the adapter/report. Factor names, orientation and scale 1 follow the explicit-rank modes. Success emits `adapter.safetensors` and format-2 `lora-report.json` with per-tensor ranks, source and adapter digests/sizes, shape input digest, tool name/version, failure map and numerical metrics. It emits no adapter_config or runtime-specific name mapping.

**Exactness is scoped.** The selector product is algebraically exact before the F32 cast. F32 storage loss, signed-zero bake mismatches and bake overflow are measured and exported, never grounds for a worthiness refusal. Bake-error and ULP metrics cover successfully baked coordinates; `bake_failure_count` separately counts coordinates without a finite bake. `stored_delta_exact` and `measured_bake_exact` say what the measurement established. The frozen bake uses the same F64 accumulator, F32 cast and BF16 ties-to-even rule as the existing modes. Offline metrics use only real saved old/new coordinates (`verification_scope: saved-coordinates-only`, `measured_elements`); they do not invent the unseen base tensor. Delta factors outside the recorded changes are zero, but unseen signed zeros or nonfinite base values are not verified. Base-backed reports use `full-tensor` scope. No runtime fidelity or study/benchmark transfer is claimed.

The current representation requires nonempty 2-D BF16 matrices, finite old/new values, exact F64 differences and finite F32 stored factors. These are explicit representation limits, not a policy on whether a patch is useful. When a tensor is unsupported, all named tensors are examined: the command exits nonzero and publishes **only** a report (`status: unsupported`, named `failures`); no partial adapter is published. Malformed inputs, invalid option combinations, cancellation or I/O errors can fail before a report. Existing destinations are preserved. Full-form offline input reports that old values/base are missing. Changing the tensor class or numeric representation is future work.

This mode keeps measured coordinate pairs plus dense selector factors in memory, proportional to inspected elements + `k_t*(rows+width)`, and accumulates factors for all tensors. There is no output-size cap; ordinary memory/disk limits still apply. Records for the canonical adapter and report both use `role: derived`, `derived_from: [replacement_sha256]`, and `tool: {"name": "weightspace derive-lora", "version": "0.2.0"}`. `weightspace check` validates the pair and rebuilds sparse-old evidence offline. It cannot independently recheck full-tensor bake metrics without the base. The canonical pair profile currently requires exactly one source file; multi-source coverage needs a bundle/report contract and cannot be claimed through a partial pair. Other v1 derived names retain their existing checks; absence of dual artifacts remains valid for historical records.

### Check or regenerate an index checkout (contributors)

From a repository checkout, with no argument both commands act on the current directory (and its `docs/example/` tree):

```sh
weightspace check            # every offline check; prints "checks passed"
weightspace generate         # rewrite the generated files
weightspace check ./path/to/checkout
weightspace generate ./path/to/checkout
```

`check` writes nothing and fails when a generated file is missing, out of date, or no longer generated. `generate` rewrites the generated files and prints the total generated-file count per tree, even when unchanged files are skipped rather than rewritten. What is generated, and the rules behind the checks, are in [repository structure](../docs/structure.md) and the [record schema](../schema/README.md) — this guide does not duplicate them.

## Command reference

```text
weightspace replacement --base <BASE> --edited <EDITED> --out <OUT>
                        [--manifest <MANIFEST>] [--temp-dir <TEMP_DIR>]
weightspace apply --patch <PATCH> --model <MODEL> --out <OUT>
                  [--expect-tensor-sha256 NAME=HEX]... [--temp-dir <TEMP_DIR>]
weightspace derive-lora --patch <PATCH> --method selector-exact --factor-dtype f32
                       (--model <MODEL> | --tensor-shapes <SHAPES_JSON>) --out <OUT>
                       [--expect-tensor-sha256 NAME=HEX]... [--temp-dir <TEMP_DIR>]
weightspace derive-lora --patch <PATCH> --model <MODEL> --rank <RANK>
                       --method <svd|selector> --factor-dtype f32 --out <OUT>
                       [--expect-tensor-sha256 NAME=HEX]... [--temp-dir <TEMP_DIR>]
weightspace check [REPO]
weightspace generate [REPO]
```

- `replacement`: `--base` and `--edited` are the two checkpoint files; `--out` is the patch file to create; `--manifest` is the optional authors' manifest gate; `--temp-dir` chooses the snapshot directory (default: the system temporary directory).
- `apply`: `--patch` is the replacement file; `--model` is the model directory; `--out` is the new model directory to create; `--expect-tensor-sha256 NAME=HEX` (repeatable) gives the expected base SHA-256 for one full-form patch tensor; `--temp-dir` is as above.
- `check` / `generate`: the optional `REPO` positional argument is the repository root (default: current directory).

Run `weightspace <command> --help` for the authoritative wording; this page follows it.

## Inputs in detail

### `replacement`: two matching checkpoints

`--base` and `--edited` must each be a single safetensors file, and together they must describe **exactly the same tensors**: the same set of tensor names, and for each name the same shape and dtype. Anything else — a renamed tensor, a different dtype export, a quantized copy — is refused as a tensor mismatch rather than guessed at. The output records new **values**, never differences, choosing the full or sparse storage per tensor described in [patch formats](../docs/patch-formats.md). Tensors whose dtype elements are not whole stored bytes cannot be addressed element by element and are refused.

### The optional `--manifest` gate

`--manifest` points to a JSON file in one strict shape: a root object whose only field is `tensors`, an array with one object per checked tensor. Each object has exactly the five fields `name`, `shape`, `dtype`, `base_sha256`, and `edited_sha256` — a non-empty tensor name, its shape as unsigned integers, its safetensors dtype name, and the SHA-256 of that tensor's raw stored bytes in the base and edited checkpoints (64 hex characters each). No tensor may be listed twice, and the file is capped at 16 MiB. For example (replace the digests with real ones):

```json
{
  "tensors": [
    {
      "name": "model.layers.0.mlp.down_proj.weight",
      "shape": [2048, 8192],
      "dtype": "BF16",
      "base_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "edited_sha256": "1111111111111111111111111111111111111111111111111111111111111111"
    }
  ]
}
```

When a manifest is given, the build fails closed unless every listed tensor's shape, dtype, and both hashes match the checkpoints as read. It is a guard against building a patch from the wrong pair of files; it is data only, and nothing in it is executed.

### `apply`: one model directory, one weights file

`--model` must be a directory containing exactly one weights file at its root, named `model.safetensors`. A `model.safetensors.index.json`, several root `.safetensors` files, or a differently named single file are all refused as sharded or unsupported layouts — do not rename shards to work around this. Other files in the directory (config, tokenizer, and so on) are copied to the output byte-for-byte; only the patched tensor bytes differ. `--out` must not be inside `--model`.

### `apply`: expected digests for full-form tensors

A sparse patch tensor carries the old values it replaces, so `apply` can confirm your base holds those values before writing. A **full-form** tensor carries only the new values, so the patch file alone cannot prove it belongs on your base. That is what `--expect-tensor-sha256 NAME=HEX` is for: the SHA-256 of the base tensor's raw stored bytes, supplied once per full-form tensor (64 hex characters; uppercase is accepted and normalized; each name once). `apply` hashes the tensor in your model and refuses with `wrong base` — showing expected and computed digests — unless they match. Supplying a digest for a tensor that is sparse in the patch, or not in the patch at all, is refused as not a full-form tensor.

Take these digests from a source independent of the model in front of you: the patch author, or the index records for the exact base profile and patch version. Hashing your local tensor and feeding the result back proves nothing — it will always match, including on the wrong model. The same applies to whole files: before applying, check your patch and model files against the sizes and SHA-256 digests recorded in the index, as in the [root README](../README.md).

## Space, snapshots, and outputs

**Originals are never modified.** Before parsing, each checkpoint or patch input is copied into a private snapshot, and every read, comparison, and hash is taken from that snapshot. Editing, moving, or replacing an input file after its snapshot is taken therefore cannot change what the run sees — and if an input changes size while its snapshot is being copied, the run refuses instead of using a torn copy. That size check is a before/after comparison, not an atomic guarantee: a same-size in-place edit made while the copy is still running is outside it, so keep inputs unchanged until the run finishes. Snapshots are held only for the duration of the run.

That isolation costs temporary space:

- `replacement` holds snapshots of base and edited at once: allow roughly `size(base) + size(edited)` free in the snapshot directory, plus space for the patch file at `--out`.
- `apply` holds snapshots of the patch and of `model.safetensors`: allow roughly `size(patch) + size(model.safetensors)` free in the snapshot directory, plus space for a full copy of the model directory next to `--out` while it is staged.

Snapshots go in `--temp-dir` when given, else the system temporary directory. On Linux that default is often tmpfs, which is RAM: for large checkpoints, pass a directory on disk, for example `--temp-dir ./scratch` (create it first). For replacement/apply, apart from snapshots on tmpfs, the tool streams tensor bytes in bounded chunks and memory-maps its snapshots rather than loading a whole checkpoint into memory, so a large checkpoint does not need a checkpoint's worth of free RAM.

Outputs are **no-clobber**:

- `--out` must not already exist in any form (file, directory, or symlink), and its parent must already exist as a real directory — parent directories are not created for you. To re-run, choose a new `--out` or move the previous output aside yourself.
- Nothing is written through a symlink. Single-file outputs are published by a no-clobber hard link; model directories are staged beside the destination and renamed into place only after every check and write has completed (on Unix that publication is atomic; on Windows it is a local check-then-rename, not atomic against a concurrent writer in the same directory).
- A refusal (next section) publishes no output and leaves every input byte-unchanged. The one edge is an operating-system error after the final link or rename, which can leave a complete output visible while the command still reports failure — if that happens, treat the output as suspect and re-run to a fresh `--out`.

## When it refuses

Malformed or mismatched input is a **typed refusal**, never a crash or a best-effort write: the command prints a specific message to stderr, exits non-zero, and publishes nothing. A refusal is the tool working as designed; fix the cause it names rather than looking for a force flag — there isn't one. The most useful diagnoses:

| Message contains | Likely cause | Next step |
|---|---|---|
| `output already exists` | `--out` is taken | Pick a new `--out` |
| `needs an expected base digest` | Full-form tensor without a flag | Add `--expect-tensor-sha256` for it |
| `not a full-form tensor` | Digest given for a sparse or absent tensor | Remove that `--expect-tensor-sha256` |
| `wrong base` | Model is not the patch's base | Stop; get the exact base/profile |
| `tensor mismatch` | Name/shape/dtype sets differ | Export both checkpoints identically |
| `identical; no replacement` | Base and edited are the same | Nothing to build |
| `sharded or indexed` | Not a single `model.safetensors` | Use a single-file checkpoint |
| `malformed safetensors` | File is not valid safetensors | Re-download; check size and SHA-256 |
| `unsupported dtype` | Sub-byte element dtype | Use an unquantized profile |
| `manifest …` | Manifest shape or hash mismatch | Fix the manifest or the files |
| `changed while it was being snapshotted` | Input was modified mid-run | Re-run with stable inputs |
| `symlink refused`, `output parent`, `invalid output name` | Path breaks the custody rules | Use real directories and a fresh name |
| An OS error naming `--temp-dir` or `--out` | Usually out of space | Free space; move `--temp-dir` to disk |

For `check`, failures list each problem; if only generated files are reported, from `tool/`, run `cargo run --locked -- generate ..` (the failure output ends with that hint). Any other problem is in an authored record — see the [record schema](../schema/README.md) and [repository structure](../docs/structure.md).

Generated outputs are `lookup/{by-digest,by-repo}/`, `models/index.json`, `models/README.md` and each base's `index.json`/`README.md`, both at the repository root and under `docs/example/`. All are generated, including the example equivalents. Patch cards and source records remain authored.

## Development checks

From `tool/` (in CI, `cargo fmt --check` and `cargo run --locked -- check ..` run on the Linux lane only; clippy and the tests run on every lane — Linux, macOS, and Windows):

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- check ..
```

## Further reading

- [Root README](../README.md) — what the index is, browsing and submitting patches, and using a patch step by step.
- [Patch formats](../docs/patch-formats.md) — replacement vs. LoRA, full vs. sparse storage, derived files.
- [Repository structure](../docs/structure.md) — layout, records, hosting, review, and the rules `check` enforces.
- [Workflows](../docs/workflows.md) — running a patched copy or a LoRA with common runtimes.
- [Record schema](../schema/README.md) — the JSON contract for every index file.
- [Synthetic example](../docs/example/README.md) — a complete made-up base with one patch of each form.

## License

The `weightspace` tool is licensed under the MIT License; see [LICENSE](LICENSE). That covers the tool only — every patch and base model carries its own license and notices, and nothing here relicenses third-party material (see the root [README](../README.md) and NOTICE).
