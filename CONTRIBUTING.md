# Contributing

Authored records live under `models/<namespace>/<model>/`: `base.json` describes exact base profiles, each patch has a card, `patch.json`, license notices, and `v<N>/` versions governed by the adoption policy. See [structure and rules](docs/structure.md) and the [schema](schema/README.md). The documentation example in `docs/example/` mirrors `models/` and `lookup/` without entering the real catalog.

To add an entry, select the exact base and profile, add a patch folder or a new version, and record upstream sources, pinned revisions, licenses, file sizes and SHA-256 digests. Keep required notices with the patch. Before adoption, existing version files may be edited in place; deletions and file-type changes remain forbidden. After adoption, add a new version instead of editing merged files. Rule 5 documents the transition. Describe measured evidence and limitations without unsupported quality or safety claims.

`lookup/`, `models/index.json`, `models/README.md`, and each base's `index.json` and `README.md` are generated, including their equivalents under `docs/example/`. Keep patch cards and source records authored. From `tool/`, regenerate and check before opening a pull request:

```sh
cargo run --locked -- generate ..
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- check ..
```

Cargo writes build artifacts under `tool/target/`. Submit the authored files and regenerated outputs together. Expect review of compatibility, byte identity, sources, licenses and the evidence behind claims; passing checks alone does not establish patch quality. See [DISCLAIMER.md](DISCLAIMER.md).
