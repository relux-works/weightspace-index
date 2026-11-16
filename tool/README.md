# weightspace

`weightspace` is the command-line tool for the [Weightspace Index](../README.md). It builds and applies **exact replacement patches** for open models, and it runs the index's offline checks and generator. This page is the user guide for the tool; the index itself — what is filed, where, and under what rules — is described in the [root README](../README.md) and [repository structure](../docs/structure.md).

> [!WARNING]
> A patched model may behave unexpectedly or unsafely. This tool checks bytes and tensor identities; it cannot tell you whether a patch does what its description claims. Inclusion in the index, a successful `weightspace` run, and published benchmark results do not prove that a third-party patch is safe or effective. Read the patch's card and evidence, and assess it for your intended use. See [DISCLAIMER.md](../DISCLAIMER.md).

## What it does, in plain language

A model checkpoint is a large file of named tensors (arrays of numbers). A **replacement patch** says: "for this exact base model, replace the values of these tensors with these new values." Applying it is plain assignment — the new values are copied bit-for-bit into a copy of the model — so the result is exact, with no rounding. For each changed tensor the tool picks whichever storage — in full, or sparsely as the changed positions plus the old and new values at those positions — makes the serialized patch file smaller, storing in full when the sizes tie, so a tensor changed in only a few places can still be stored in full. See [patch formats](../docs/patch-formats.md) for the file-level details.

`weightspace` has four commands:

- `weightspace replacement` — compare a base checkpoint with an edited checkpoint and write one replacement patch file (`.safetensors`) holding only what changed. This is for patch authors.
- `weightspace apply` — bake one replacement patch file into a **new copy** of a model directory. The original model is never modified. This is for people who have downloaded a patch and a matching base model and want a patched copy to run.
- `weightspace check [repo]` — run every offline check over an index repository checkout. This is for contributors and CI.
- `weightspace generate [repo]` — rebuild the index's generated files in a checkout. This is for contributors.

Everything runs **offline**. The tool never uses the network, never downloads models or patches, and never executes code from a repository or a checkpoint: inputs are read as data only. Anything you apply you must already have on disk, and checking downloaded files against the index's recorded sizes and SHA-256 digests remains your step before you run `apply` (see "Use a patch" in the [root README](../README.md)).

## What it does not do

- It does **not** train or fine-tune models, run inference, or run benchmarks. Run the patched copy with your usual runtime; [workflows](../docs/workflows.md) shows where common runtimes fit.
- It does **not** combine patches. A baked copy takes at most one replacement. LoRA adapters are applied by your runtime, as published — `weightspace` handles replacement patches only.
- It does **not** prove patch quality, safety, or effectiveness, and a clean run is not an endorsement (see the warning above).
- It does **not** quantize, dequantize, or convert weights, and it does not read pickle-based formats (`.bin`, `.pt`, `.pth`, `.pkl`). Checkpoints and patch files must be single-file [safetensors](https://github.com/huggingface/safetensors); sharded or indexed layouts are refused (see [Inputs in detail](#inputs-in-detail)).

## Install

### From a release archive (when published)

No downloadable release exists yet; until one is published, install [from source](#from-source) — that is the verified path. When releases are published, they will appear under **GitHub Releases** for this repository, one archive per platform, where `<tag>` is the release tag including its leading `v` (for example, `v0.1.0`):

| Platform | Archive |
|---|---|
| Linux, x86_64 | `weightspace-<tag>-x86_64-unknown-linux-musl.tar.gz` |
| macOS, Apple Silicon (arm64) | `weightspace-<tag>-aarch64-apple-darwin.tar.gz` |
| Windows, x86_64 | `weightspace-<tag>-x86_64-pc-windows-msvc.zip` |

Each archive will contain a single top-level directory, `weightspace-<tag>-<target>/`, holding the `weightspace` binary (`weightspace.exe` on Windows) and its `LICENSE`, and the release will publish a combined `SHA256SUMS` file. To install one: download the archive and `SHA256SUMS`, verify the archive's SHA-256 before extracting it (stop on any mismatch), extract it, and put the binary from that directory in a directory already on your `PATH`. For any other platform or architecture, build from source.

### From source

You need Rust **1.85 or newer** on Linux, macOS, or Windows. From the repository root:

```sh
cargo build --locked --release --bin weightspace
```

`--locked` builds exactly the dependency versions recorded in `Cargo.lock`. The binary is written to `target/release/weightspace` (`target\release\weightspace.exe` on Windows); run it from there or copy it to a directory on your `PATH`. Confirm it runs:

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

### Check or regenerate an index checkout (contributors)

From a repository checkout, with no argument both commands act on the current directory (and its `examples/` tree):

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

Snapshots go in `--temp-dir` when given, else the system temporary directory. On Linux that default is often tmpfs, which is RAM: for large checkpoints, pass a directory on disk, for example `--temp-dir ./scratch` (create it first). Apart from snapshots on tmpfs, the tool streams tensor bytes in bounded chunks and memory-maps its snapshots rather than loading a whole checkpoint into memory, so a large checkpoint does not need a checkpoint's worth of free RAM.

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

For `check`, failures list each problem; if only generated files are reported, run `cargo run -- generate` (the failure output ends with that hint). Any other problem is in an authored record — see the [record schema](../schema/README.md) and [repository structure](../docs/structure.md).

## Development checks

From the repository root (in CI, `cargo fmt --check` and `cargo run --locked -- check` run on the Linux lane only; clippy and the tests run on every lane — Linux, macOS, and Windows):

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- check
```

## Further reading

- [Root README](../README.md) — what the index is, browsing and submitting patches, and using a patch step by step.
- [Patch formats](../docs/patch-formats.md) — replacement vs. LoRA, full vs. sparse storage, derived files.
- [Repository structure](../docs/structure.md) — layout, records, hosting, review, and the rules `check` enforces.
- [Workflows](../docs/workflows.md) — running a patched copy or a LoRA with common runtimes.
- [Record schema](../schema/README.md) — the JSON contract for every index file.
- [Synthetic example](../examples/README.md) — a complete made-up base with one patch of each form.

## License

The `weightspace` tool is licensed under the MIT License; see [LICENSE](LICENSE). That covers the tool only — every patch and base model carries its own license and notices, and nothing here relicenses third-party material (see the root [README](../README.md) and NOTICE).
