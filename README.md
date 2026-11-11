> [!WARNING]
> Patched models may behave unexpectedly or unsafely. We cannot guarantee that a patch does only what its description claims, or that reviews and benchmarks reveal every effect. Repository inclusion, redistribution by us, and published results do not guarantee safety. Assess patches for your intended use, including possible unintended effects or potentially malicious changes introduced by third parties. See [DISCLAIMER.md](DISCLAIMER.md).

# Weightspace Index

**Find the change. Check the bytes. Read the evidence.**

An open index of weight patches for open models: what each patch changes, which exact model files it was made for, where to download it, and how it measured.

**Status: empty.** The layout, record schema, a synthetic example and its tests are in place. No patch has been filed yet, and the intake tool, the generator, the client tool and CI do not exist yet.

## Browse

Patches are filed under the model they were made for, at `models/<hf-namespace>/<hf-model>/`, with the model's exact Hugging Face spelling. That folder's `README.md` is a generated table of every patch and its results; each patch folder has a card (`README.md`) and one folder per version.

A patch is one of two forms ([details](docs/patch-formats.md)):

- **replacement**: new values for some of the base model's tensors, applied bit-exactly to a copy of the model;
- **lora**: an adapter, as published, loaded or merged by your runtime.

See [`examples/`](examples/README.md) for a complete synthetic base with one patch of each form.

## Use a patch without a tool

1. Pin the index: note the commit you read.
2. Open the version's `version.json`. Download each file from a listed locator: a Hugging Face file at its pinned revision, or a file in this repository at your pinned commit.
3. Check every file's size and SHA-256 against `version.json` (`shasum -a 256 <file>`, `wc -c < <file>`). Stop on any mismatch.
4. Check that your model files match the target profile's digests in `base.json`.
5. Apply it with your runtime: [workflows](docs/workflows.md).

## Submit a patch

Open a pull request that adds a patch folder or a new version, following [structure and rules](docs/structure.md#submissions-and-review). Patch files may stay on Hugging Face; files uploaded here must each be under 100 MB.

## For developers

- [Structure and rules](docs/structure.md): layout, records, hosting, review and the client's reading order.
- [Record schema](schema/README.md): one JSON Schema for every authored and generated file.
- `go test ./...` (Go 1.25+, standard library only, no network) checks the schema, the example tree and the layout rules.

## Licenses and attribution

There is no repository license file yet; that is an owner decision. Each patch carries its own license and notices in `notices/`, and nothing here relicenses third-party material. See [NOTICE](NOTICE.md).
