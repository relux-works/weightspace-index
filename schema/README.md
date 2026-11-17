# Record schema, contract version 1

[records.schema.json](records.schema.json) is one JSON Schema (Draft 2020-12) with a definition for every index record. Every record carries `"contract_version": 1`; a reader that meets another version stops instead of guessing.

| Definition | File | Written by |
|---|---|---|
| `base` | `models/<ns>/<model>/base.json` | author |
| `patch` | `.../<patch-slug>/patch.json` | author |
| `version` | `.../<patch-slug>/v<N>/version.json` | author, usually drafted by the intake tool |
| `run` | `.../runs/<run-id>/run.json` and `.../v<N>/runs/<run-id>/run.json` | whoever ran it |
| `comparison` | `.../v<N>/comparisons/<id>.json` | maintainers |
| `assessment` | `.../v<N>/assessments/<id>.json` | maintainers |
| `index` | `models/<ns>/<model>/index.json` | generator |
| `modelsIndex` | `models/index.json` | generator |
| `digestLookup` | `lookup/by-digest/<aa>/<sha256>.json` | generator |
| `repoLookup` | `lookup/by-repo/<namespace>/<repo>.json` | generator |

The path decides the record definition; a JSON record anywhere else under `models/` is an error. Listed payloads under a version’s `files/` (including adapter configs and derivation reports) use their own formats, not the index record contract. [structure.md](../docs/structure.md) explains what each field means.

## What the schema settles, and what it cannot

The schema checks each file on its own: required fields, no unknown fields, digests as 64 lowercase hex characters, sizes as positive integers, revisions as full 40-hex commits, safe relative paths, and the allowed values of `kind`, `status`, `role` and decisions.

What it cannot see is checked by the layout tests and, later, by CI:

- folders at fixed depth with valid slugs, version folders `v1..vN` matching `patch.json`, no symlinks, no paths differing only in case;
- the target profile and every run's profile exist in `base.json`; comparisons and assessments name runs and comparisons that exist;
- source files match the kind: `.safetensors` for a replacement; `adapter_model.safetensors` with `adapter_config.json`, or one `.gguf`, for a LoRA; derived files come from this version's sources;
- an `index-path` locator is relative to the folder of the file that declares it, stays under `files/` for a version, points at a file with the recorded size and SHA-256, and every hosted file is listed;
- a replacement file is valid safetensors with full tensors or complete `#idx`/`#new`/`#old` trios and sorted unique indices.

The canonical derived `adapter.safetensors` / `lora-report.json` pair is checked offline: either name requires both, hosted under this version, with the same source digest and tool identity. The current canonical profile requires exactly one replacement source file; it refuses a partial claim over a multi-source version. Reports must be complete selector-exact format 2, F32, scale 1, and match all source tensors, factor names/shapes, per-tensor ranks, bytes and digests. For sparse-old reports the checker rebuilds factors and measured tensor evidence from #old/#new and the report's declared shapes. A base-backed full-tensor report needs the base to independently recheck its numerical claims; offline checking verifies its descriptors and bindings only.

The 2026-11-17 publication expectation is both forms for every entry; historical v1 records without derived files remain valid. Conversion currently covers nonempty 2-D BF16 matrices within the numeric boundary in [structure](../docs/structure.md#patch-forms). LoRA-source reverse conversion and unsupported tensor classes are unresolved, not silently skipped. No required fields or contract version are changed here; canonical-pair validation is a refinement and may reject previously accepted incomplete pairs. A universal mandatory rule requires an explicit future contract/migration decision.

Downloading Hugging Face files, recomputing remote digests, rebuilding other derived formats and protecting merged versions from change need the network or Git history; they belong to intake and CI.

Generated files are checked twice: each must satisfy its definition, and `weightspace check` fails when one differs from a fresh build.

## Tests

The checks above live in the Rust crate under [`tool/`](../tool/), so CI and the intake tool apply the same rules. `cargo test --locked` from `tool/` runs them against the [synthetic example](../docs/example/README.md) and against deliberately broken copies of it that must fail. Validation uses the `jsonschema` crate with remote and file references disabled. JSON Schema ignores unknown keywords, so a test also limits the schema to the keywords it actually uses, which catches a misspelt rule.

The path mapping also applies below `docs/example/`. Generated `models/README.md` and per-model tables are Markdown: regeneration checks their exact bytes, with links relative to each table, while JSON Schema covers the JSON records.
