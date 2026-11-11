# Record schema, contract version 1

[records.schema.json](records.schema.json) is one JSON Schema (Draft 2020-12) with a definition for every JSON file in the index. Every file carries `"contract_version": 1`; a reader that meets another version stops instead of guessing.

| Definition | File | Written by |
|---|---|---|
| `base` | `models/<ns>/<model>/base.json` | author |
| `patch` | `.../<patch-slug>/patch.json` | author |
| `version` | `.../<patch-slug>/v<N>/version.json` | author, usually drafted by the intake tool |
| `run` | `.../runs/<run-id>/run.json` and `.../v<N>/runs/<run-id>/run.json` | whoever ran it |
| `comparison` | `.../v<N>/comparisons/<id>.json` | maintainers |
| `assessment` | `.../v<N>/assessments/<id>.json` | maintainers |
| `index` | `models/<ns>/<model>/index.json` | generator |
| `bases` | `bases.json` | generator |
| `digestLookup` | `by-digest/<aa>/<sha256>.json` | generator |
| `repoLookup` | `by-repo/<namespace>/<repo>.json` | generator |

The path decides the definition; a JSON file anywhere else under `models/` is an error. [structure.md](../docs/structure.md) explains what each field means.

## What the schema settles, and what it cannot

The schema checks each file on its own: required fields, no unknown fields, digests as 64 lowercase hex characters, sizes as positive integers, revisions as full 40-hex commits, safe relative paths, and the allowed values of `kind`, `status`, `role` and decisions.

What it cannot see is checked by the layout tests and, later, by CI:

- folders at fixed depth with valid slugs, version folders `v1..vN` matching `patch.json`, no symlinks, no paths differing only in case;
- the target profile and every run's profile exist in `base.json`; comparisons and assessments name runs and comparisons that exist;
- source files match the kind: `.safetensors` for a replacement; `adapter_model.safetensors` with `adapter_config.json`, or one `.gguf`, for a LoRA; derived files come from this version's sources;
- an `index-path` locator is relative to the folder of the file that declares it, stays under `files/` for a version, points at a file with the recorded size and SHA-256, and every hosted file is listed;
- a replacement file is valid safetensors with full tensors or complete `#idx`/`#new`/`#old` trios and sorted unique indices.

Downloading Hugging Face files, recomputing their digests, rebuilding derived files and protecting merged versions from change need the network or Git history; they belong to the intake tool and CI.

## Tests

`go test ./...` from the repository root runs every check above against the [synthetic example](../examples/README.md), plus deliberately broken copies of it that must fail. The Go tests are interim: the tools will be written in Rust, and these checks move there. Until then the tests include a small schema evaluator that understands exactly the keywords this schema uses and rejects any other.
