> [!WARNING]
> Patched models may behave unexpectedly or unsafely. We cannot guarantee that a patch does only what its description claims, or that reviews and benchmarks reveal every effect. Repository inclusion, redistribution by us, and published results do not guarantee safety. Assess patches for your intended use, including possible unintended effects or potentially malicious changes introduced by third parties. See [DISCLAIMER.md](DISCLAIMER.md).

# Weightspace Index

**Find the change. Check the bytes. Read the evidence.**

A proposed open index of weight-space patches: what an edit changes, which exact model it belongs to, how to obtain it, and what evidence supports using it. Small edits deserve the same care as whole model releases. A familiar model name, a small download, or a promising result elsewhere cannot identify the files on your machine.

**Status: scaffold only.** This snapshot contains a record contract, an empty catalog, a labelled template and workflow documentation. It contains no real patch entries, payloads, supported application routes or measured model results. All example quality is **UNKNOWN**. No index reader, downloader or patch tool is released here.

## Start here

| Question | Read |
|---|---|
| What belongs in an entry? | [Structure and contribution rules](docs/structure.md) |
| Is this a sparse edit, an adapter or a whole model? | [Patch formats, side by side](docs/patch-formats.md) |
| How would I use native weights with vLLM? | [Safetensors workflow](docs/workflows.md#native-safetensors-with-vllm) |
| How would I use a GGUF with llama.cpp? | [GGUF workflow](docs/workflows.md#gguf-with-llamacpp) |
| What can a program read? | [Empty catalog](catalog.json), [contract](schema/README.md), [JSON Schema](schema/catalog.schema.json) |
| How would a contributor begin? | [TEMPLATE record](examples/TEMPLATE.json) — synthetic, never an entry |

## An entry is a map, not an endorsement

The catalog connects a stable fix identity to an exact profile, treatment and scoped assessment. Profiles identify files and numerical/runtime context. Treatments identify the route, targets, order and output. Runs own observations; comparisons own matched analyses; assessments apply a versioned decision policy. A score should have one authoritative home.

Payload locations may be repository-relative paths in a pinned index snapshot or structured external Hugging Face locations at full immutable revisions. Both require SHA-256 and byte counts. Location can change while byte identity stays the same. An **advertised** hash is a verification target reported by a source; a **verified** hash needs retained evidence of hashing the complete named bytes. Neither establishes quality or publisher authenticity.

**UNKNOWN is not a zero score, a pass or a rejection.** It means the required evidence does not establish the answer. Missing applicability, unsupported routes, refused comparisons, regressions and inconclusive assessments must remain visible. Evidence is specific to artifact digests, profiles, treatments and protocols; native results never fill a missing GGUF assessment.

## Use the index without a special tool

This is the manual contract for future qualified entries; the scaffold has none to apply.

1. Obtain a trusted, reviewed index snapshot with Git and pin its full commit. Read the entry, notices, exact-profile evidence and known gaps. The template cannot be selected for use.
2. For an in-repository payload, obtain the named regular file from that pinned snapshot. For an external payload, obtain only the named files from the full HF repository revision. Preserve required configuration, tokenizer/template, sidecars and notices. A branch name, LFS pointer or browser HTML is not the payload.
3. Count bytes and compute SHA-256 for every required file. Compare with the entry's expected values before any application. For example, `shasum -a 256 ./staging/payload.bin` and `wc -c < ./staging/payload.bin` inspect a file you have already obtained; the path is illustrative. Stop on a mismatch, missing digest, unavailable file or unresolved rights.
4. Only a separately qualified route can turn those verified inputs into a new destination. Preserve originals, verify the expected output and complete file set, and record the exact execution profile. This scaffold supplies no applicator command.
5. Read or obtain a matched assessment before relying on a benefit claim. Successful application, loading and hash equality answer different questions from useful behavior.

[The workflow examples](docs/workflows.md) show where vLLM and llama.cpp would fit after those gates. They are documentation, were not executed, and leave quality UNKNOWN.

## Attribution, licenses and changes

There is intentionally **no repository LICENSE file**: the repository license remains an owner decision. No permission for this repository's original material should be inferred from third-party terms. Each future entry must carry per-component licenses, author attribution, pinned sources, applicable notices and change history. Nothing in this index relicenses third-party payloads. Public availability and a matching digest do not settle redistribution rights.

The original work here is the scaffold, record format, synthetic template and explanatory workflow. Linked upstream designs and runtime documentation remain their authors' work; no upstream code or payload is copied. See [NOTICE](NOTICE.md) for attribution and [structure rules](docs/structure.md) for the obligations when real material is added.

Contributors can check the offline schema fixtures with `go test ./...` using Go 1.25 or later. The tests use the standard library and perform no network or model operations. They check the documented schema subset and controls; they do not qualify files, runtimes or patches.
