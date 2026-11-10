# Repository structure and change rules

**Status: target layout.** This snapshot is still the scaffold: no entry or payload is present, and the `models/` tree, per-patch files, generator and CI described below do not exist yet. [Migration from the scaffold layout](#migration-from-the-scaffold-layout) lists the changes that bring the repository to this shape.

## Goals

- **Humans** browse by the base model they run; opening that model's folder lists every patch for it, with no extra navigation.
- **Machines** locate any patch or release by a fixed path pattern, or read one generated index at a pinned commit.
- **Paths are permanent.** A path is the patch's address and never changes after merge.
- **Each fact is authored once.** Contributors edit small per-patch files; indexes and listings are generated from them.

## Target shape

```text
catalog.json                          GENERATED machine index of all patches and releases
schema/                               schemas for every authored and generated file
docs/                                 contracts and explanations
tools/                                generator and layout checks
CODEOWNERS                            per-patch review ownership
NOTICE.md                             attribution and change-marking policy
DISCLAIMER.md                         repository disclaimer
models/
  <hf-namespace>/                     base model namespace, exact casing
    <hf-model>/                       base model repository name, exact casing
      README.md                       GENERATED list of patches for this base
      base.json                       authored: known profiles of this base
      <patch-slug>/                   one folder per patch; the slug is permanent
        README.md                     authored: human description, usage, caveats
        patch.json                    authored: patch metadata
        notices/                      licenses and attribution for this patch
        releases/
          <version>/                  immutable once merged
            release.json              authored: exact identity of this release
            payload/                  optional, small in-repository payload files only
            evidence/                 verification and evaluation records
```

Fixed path patterns:

| Object | Pattern | Address |
|---|---|---|
| Base | `models/*/*/base.json` | `<hf-namespace>/<hf-model>` |
| Patch | `models/*/*/*/patch.json` | `<hf-namespace>/<hf-model>/<patch-slug>` |
| Release | `models/*/*/*/releases/*/release.json` | patch address + `@<version>` |

## File roles

| File | Authored or generated | Contains |
|---|---|---|
| `base.json` | Authored | One or more base profiles, each with an ID, source repository and full revision, file digests and sizes, format and precision/quantization. Repackaged forms of the same model (another format, a quantization) are additional profiles here. |
| `patch.json` | Authored | Title, short summary, `authors[]` (one or more people or organizations), `submitter`, patch kind (sparse exact, LoRA or whole artifact), license, tags and status (`active`, `deprecated`, `withdrawn`). |
| `release.json` | Authored | Target base profile (by ID from `base.json`), the treatment, every payload artifact with locator, SHA-256 and size, and references to its evidence. |
| Patch `README.md` | Authored | Prose for people. It links to machine facts instead of restating them. |
| `catalog.json` | Generated | All bases, patches and releases as records, each carrying its address. |
| Base `README.md` | Generated | Table of the patches filed under that base. |
| Other views | Generated | Optional listings by author, kind or tag. |

Generated files are committed so repository browsing works without tooling, and are never edited by hand. Browse pages link the authoritative records instead of maintaining independent scores.

## Layout rules

1. **Fixed depth.** No optional or variable path levels anywhere under `models/`.
2. **Base folder choice.** A patch is filed under the model it was made against. Repackaged weights (format conversion, quantization) are filed under their upstream model as a profile in `base.json`, not as a separate base folder. A model with its own training (a fine-tune) gets its own base folder.
3. **Casing.** Namespace and model folders preserve the source repository's exact spelling and case. Two paths that differ only in case are rejected, because case-insensitive filesystems cannot hold both.
4. **Patch slugs.** Lowercase ASCII, hyphen-separated, describing the effect. No author, base model name or version in the slug. Unique within a base. Never renamed: a retired patch changes its status in `patch.json` instead.
5. **Authorship lives in metadata, not paths.** Authors, submitter and the account that opened the change are recorded separately and may differ. Per-author review rights are granted through `CODEOWNERS` entries for the patch folder.
6. **Releases are immutable.** A new payload byte sequence, recompressed archive, base profile, treatment order or numerical route requires a new release folder. A merged release is never modified; corrections go in a new release or a correction record.
7. **Payload placement.** In-repository payload files live only in their own release's `payload/` folder and only below the size cap. Larger payloads stay external with a pinned revision and digest. An `index-path` locator must resolve inside the release folder that declares it.
8. **No symlinks** anywhere in the repository. Alternative groupings are generated views.
9. **Generated files stay in sync.** CI fails when any generated file differs from a fresh build.
10. **Record IDs and addresses are separate.** Catalog record IDs keep the schema's lowercase pattern; addresses keep the source repository's casing and are carried as their own field.

## Two storage modes, one identity gate

| Mode | Admitted representation | Required checks before use |
|---|---|---|
| In repository | `kind: index-path`, `storage: git`, normalized relative `path` | Caller pins the index commit; resolve inside that snapshot to a contained regular file, then check complete payload SHA-256 and size. |
| External | `kind: hf`, `repo_type`, case-preserving `repo_id`, full 40-hex `revision`, relative `path` | Source and terms are reviewed; construct the canonical HF location from fields, retrieve the named content, then independently check complete SHA-256 and size. |

Every artifact requires a digest, explicit hash status/source and size in bytes, whichever mode is used. Multiple locations mean alternatives for identical bytes, not different variants. No digest means no operational artifact record: retain a missing-evidence declaration with an owner instead. A source link in descriptive text cannot authorize fetching/applying unidentified bytes.

The schema admits ASCII path segments beginning with a letter, digit, `_` or `-`, followed by letters, digits, `_`, `-` or `.`; `/` separates segments. This preserves nested names, case, underscores and compound suffixes. Absolute paths, dot-only/hidden-leading segments, backslashes, URL encoding, query/fragment text and traversal are outside version 1. Admitted HF types are `model`, `dataset` and `space`, with exactly `namespace/repository` and an immutable revision. No arbitrary URL, custom endpoint, credential request, private/gated transport, Git LFS transport or generic GitHub payload locator is operationally supported here. Descriptive source citations are not payload locators.

These are record forms, not an implemented retriever. A future implementation must additionally prove source admission, containment, symlink safety, bounded download/redirect handling, no credential forwarding, complete-file verification and non-destructive failure. Preserve legitimate locally resolved HF snapshot symlinks and Windows copied snapshots when qualifying local discovery; require the same byte identities, not one cache layout. Unsupported transports must report that state. Never execute locator strings, repository hooks or remote code from the catalog.

## What a change requires

Before admitting a real entry, agree the exact record/route contract and paired positive/negative cases. Supply independently reviewed evidence for source revisions, digest scope/status, sizes, license duties, attribution, change marking and missing facts. Keep discovery separate from application eligibility: an honestly unknown entry may be browsable while application refuses.

Any publication needs independent review of the exact candidate and the repository's required checks and signed delivery policy. A schema pass alone cannot establish source accuracy, licensing, byte qualification or quality. Changing operational meanings or introducing a new restriction requires a new application contract version; no automatic migration or guessed default is permitted.

Keep originals and immutable evidence. Register every evidence bundle by digest and enumerate all its runs, comparisons, assessments and corrections, including negatives and unassessed records. Cross-bundle references bind bundle SHA-256 and record pointer. Inaccessible/unverified bundles make discovery completeness UNKNOWN. Retain correction edges, reasons and previous records; unresolved forks, cycles or missing targets refuse a unique current interpretation. Do not overwrite a score to express a new policy decision.

## States are not interchangeable

| State | Meaning and required behavior |
|---|---|
| Absent required field | Invalid record; reject structural admission. |
| `UNKNOWN` | Evidence missing; give reason and accountable role. Keep discoverable; no inferred defaults or support. |
| `UNSUPPORTED` | Version/route/profile/transport is outside a declared capability; refuse that operation with reason. |
| `REFUSED` | A requested comparison or interpretation cannot satisfy its prerequisites; preserve the request/reason. |
| `INCONCLUSIVE` | Evidence cannot resolve the policy decision. Never promote to support by omission. |
| `REJECT` | A declared criterion or veto failed on adequate evidence. Never use for an untested case. |
| `SUPPORTED` | A versioned policy accepts a specified use/profile on retained comparisons. It is not general model quality. |

The shipped catalog has no runs/comparisons, and its separate template has only an UNKNOWN assessment. The schema describes future evidence records without asserting that any exist. Populating them requires reviewed retained evidence and the semantic checks in the contract. Unsupported evidence must remain visible through coverage declarations; do not silently discard an incoming bundle or reinterpret a later version as version 1.

## Migration from the scaffold layout

The scaffold described `models/<publisher>/<model>/patches/<fix-id>/` browse pages and a separate top-level `payloads/<fix-id>/<release>/` tree. The target layout drops the `patches/` level so a base folder lists its patches directly, and moves payloads into the release that owns them. Changes required:

1. **This document.** Replaced the scaffold layout with the target shape and rules above; the storage-mode, identity and state rules are unchanged.
2. **`schema/`.** Add schemas for `base.json`, `patch.json` and `release.json`. Mark `catalog.json` as a generated document whose records carry their address. Extend `schema/README.md` to state which files are authored and which are generated. Keep the record-ID pattern; add a separate address field.
3. **`examples/TEMPLATE.json`.** Replace with a template patch folder (template `base.json`, `patch.json` and `release.json`) under the same sentinel and exclusion rules. Move the `payloads/TEMPLATE/...` sample paths to the in-release `payload/` location.
4. **`schema/schema_test.go`.** Update the `payloads/...` sample locator to the in-release location. Add tests for the new schemas and for the layout rules: fixed depth, case collisions, slug format, absence of symlinks and containment of in-repository locators within their release.
5. **`tools/` (new).** A generator, standard library only like the tests, that builds `catalog.json` and the per-base README tables from authored files, plus a check mode that reports differences without writing.
6. **CI (new).** Run the schema tests and the generator check on every change.
7. **`CODEOWNERS` (new).** Maintainers own everything by default; each patch folder adds its authors when admitted.
8. **`README.md`.** Explain browsing (`models/<hf-namespace>/<hf-model>/`), machine consumption (generated `catalog.json` at a pinned commit) and contribution (add a patch folder; never edit generated files).
9. **`catalog.json`.** Regenerate from the empty `models/` tree; the result must stay equivalent to the current empty catalog.

## Open decisions

- **Payload size cap** above which a payload must be external.
- **Release folder naming:** `v1`, `v2`, … or semantic versions.
- **Slug clash policy:** first merged wins; a later submitter chooses a more specific slug.
- **CI platform:** presumably GitHub Actions, since the repository is hosted on GitHub.
