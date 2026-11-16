> [!WARNING]
> Patched models may behave unexpectedly or unsafely. We cannot guarantee that a patch does only what its description claims, or that reviews and benchmarks reveal every effect. Repository inclusion, redistribution by us, and published results do not guarantee safety. Assess patches for your intended use, including possible unintended effects or potentially malicious changes introduced by third parties. See [DISCLAIMER.md](DISCLAIMER.md).

# Weightspace Index

**Find the change. Check the bytes. Read the evidence.**

An open index of weight patches for open models: what each patch changes, which exact model files it was made for, where to download it, and how it measured.

**Status:** the index contains a first replacement patch for Qwen3.5-0.8B. CI runs offline schema, layout, digest and generated-output checks. The `weightspace` command-line tool ([user guide](tool/README.md)) builds and applies replacement patches and runs those checks offline. Inclusion does not establish a quality improvement.

## Browse

Start with the generated [model catalog](models/README.md), which lists every base and its patch count. Client discovery lives under `lookup/by-digest/` and `lookup/by-repo/`; [models/index.json](models/index.json) lists the base addresses.

Patches are filed under the model they were made for, at `models/<hf-namespace>/<hf-model>/`, with the model's exact Hugging Face spelling. That folder's `README.md` is a generated table of every patch and its results; each patch folder has a card (`README.md`) and one folder per version.

A patch is one of two forms ([details](docs/patch-formats.md)):

- **replacement**: new values for some of the base model's tensors, applied bit-exactly to a copy of the model;
- **lora**: an adapter, as published, loaded or merged by your runtime.

See [`docs/example/`](docs/example/README.md) for a documentation example: a complete synthetic base with one patch of each form.

## Use a patch

1. Pin the index: note the commit you read.
2. Open the version's `version.json`. Download each file from a listed locator: a Hugging Face file at its pinned revision, or a file in this repository at your pinned commit.
3. Check every file's size and SHA-256 against `version.json` (`shasum -a 256 <file>`, `wc -c < <file>`). Stop on any mismatch.
4. Check that your model files match the target profile's digests in `base.json`.
5. Apply it: for a replacement patch, `weightspace apply` bakes it into a copy of the model ([tool guide](tool/README.md)); for a LoRA, use your runtime ([workflows](docs/workflows.md)).

## Submit a patch

Open a pull request that adds a patch folder or a new version, following [CONTRIBUTING.md](CONTRIBUTING.md) and [structure and rules](docs/structure.md#submissions-and-review). Patch files may stay on Hugging Face; files uploaded here must each be under 100 MB.

## For developers

- [Structure and rules](docs/structure.md): layout, records, hosting, review and the client's reading order.
- [Record schema](schema/README.md): one JSON Schema for every authored and generated file.
- [`weightspace` tool guide](tool/README.md): install and use the `weightspace` binary from the index's Rust crate (Rust 1.85+) — build and apply replacement patches, run every offline check with `weightspace check`, and rebuild the generated files with `weightspace generate`. `cargo test --locked` from `tool/` tests the checks and the generator. None of them use the network.

## Tools

The standalone Rust crate lives in `tool/`, including its manifest and lockfile. With Rust 1.85+ and Cargo, run from the repository root:

```sh
cd tool
cargo build --locked --release --bin weightspace
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- generate ..
cargo run --locked -- check ..
```

Cargo writes build artifacts to `tool/target/`. The `weightspace` binary checks records offline, builds/applies replacements, and regenerates the catalogs, tables and client lookups described in [structure](docs/structure.md#generated-files). For an installed binary, use `weightspace check <repository>` or `weightspace generate <repository>`; the default repository is the current directory. To verify downloaded payloads, use `shasum -a 256 <file>` and `wc -c < <file>`; these print the hash and byte count without writing files. See the [tool guide](tool/README.md) for all commands.

## Licenses and attribution

There is no repository license file yet; that is an owner decision. Each patch carries its own license and notices in `notices/`, and nothing here relicenses third-party material. See [NOTICE](NOTICE.md).
