# Repository structure and change rules

This is a scaffold contract. No entry or payload is present.

```text
catalog.json                        authoritative discovery and record arrays
schema/catalog.schema.json          structural contract, application version 1
schema/README.md                    meanings and consumer obligations
examples/TEMPLATE.json              synthetic authoring example; excluded from catalog
models/<publisher>/<model>/patches/<fix-id>/README.md   future browse pages
payloads/<fix-id>/<release>/...      optional future small payload files
docs/                              explanations and workflow examples
NOTICE.md                          attribution and change-marking policy
```

The future `models/` and `payloads/` locations are conventions, not files supplied by this snapshot. Browse pages link the authoritative records instead of maintaining independent scores. Preserve publisher/model spelling; use stable lowercase record IDs matching the schema. A directory rename does not change a fix identity. A new payload byte sequence, recompressed archive, treatment order or numerical route requires a new release/identity as appropriate.

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
