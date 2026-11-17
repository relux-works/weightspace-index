# Verification of rebuilt kit — 2026-11-17

This candidate implements only the accepted review fixes. **Mac/vLLM/Metal/2x12, real-model scores and PowerShell syntax/execution remain UNVERIFIED.** No independent acceptance of these edited bytes is claimed. New internal bundles/seals were generated; no old seal describes edited bytes.

## Preserved baseline

Before edits the reviewed `final-kit.tar.gz` and detached SHA receipt were copied byte-for-byte to restricted `../review-baseline/` and made read-only. Reviewed archive SHA256: `2266eeb12ce758ffbfbc4b5664fc634c919c11214ead5ff4c27e7e0a966d2e05`. They remain unchanged and are not included in the new archive.

## Changes and limits

- REPORTED.md, reported-claims.template.json and reported-claim.schema.json supply six pinned Qwen card claims, explicit settings/unknowns, status vocabulary, claim/run linking and separate display rules. run-summary.template.json has index-measured type, score fields, tier/coverage, trust/status, stable ID placeholders and relative artifact/hash references. It is a template, not a run; null values/hashes cannot be published as verified.
- RIKIYAT-REPRODUCTION.md pins the author HF card, RikiyaT's published Experiment 8 article, log revision `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, p50c manifests, eval/scoring code, both merged results and exact available item IDs. All ten source metrics are **not-comparable for inference with this kit**. Arithmetic readback reproduced published counts only. No scorer/backend is invented. The owner-facing 12-item card smoke/manual worksheet and hand-run target are removed. Remaining card files are retired internal compatibility fixtures only.
- `max_gen_toks` task/generation override through `--gen_kwargs` maps to HTTP `max_tokens`; output cap differs from context. RikiyaT cut_share requires actual finish_reason, which this backend loses. A cap setting cannot classify cap hits or justify a cap-hit-wrong rule.
- Loaded task evidence paths are kit/package-relative. PowerShell records its own script hash and guards process-start/stream finalization. These are static fixes only. Remaining PS divergences: post-seal verification failure may leave a stale seal; empty row files contain a blank line; signals and Windows ACL behavior differ. Windows support is not claimed.

## Fresh internal checks

These are synthetic Linux checks, not an owner Mac smoke gate. Existing fixture methods were used on a new `.temp/review-fixes/kit-under-test/`; cached data were copied locally. Original baseline archive/bundles were not relabelled. Only owned recorded server/evaluator PIDs were stopped. The bounded fixture used loopback ports 18122–18125; no model execution or forwarding.

| Check | Result and evidence in this kit |
|---|---|
| Explicit-list shards | MATH one item, MMLU two global IDs (business/biology local 0), both arm labels/seeds: all exit 0. Retired card transport fixture, one item per label: exit 0, internal diagnostics only. |
| `--tier n197 0 2` | Both suite tasks exit 0; exact frozen selection and category mapping. |
| Compare | Both A/A suite comparisons exit 0; descriptive synthetic results only, no measured model noise. |
| Wire and task fidelity | Eight short HTTP requests; cap/sampler/thinking/seed/stops and paired prompt identities pass. Four installed task instances and ten scorer/parser controls pass (evidence/wire-check.json, scoring-controls.json). |
| Full controls | 18/18 pass (evidence/controls.json); TERM after real fixture traffic gives nonzero failed sealed attempt; completed peer unchanged; new attempt recovers. |
| Intended negative reasons | 12/12 pass (evidence/intended-refusals.json), including partial paired coverage, corrupt seal, duplicate, protocol/config, benchmark and seed mismatches. Export scanner refuses nonempty secret-named fields and synthetic token shapes; nearby valid scan passes. Rejected raw output remains private/unsealed. |
| Static and tier controls | 201/201 pass (evidence/static.json); all Bash files parse; all 172 frozen tier seal entries reread. |
| Weighted summary | Two synthetic controls pass: known category weight and intended missing-category refusal (evidence/aggregation-check.json). |
| Bundle readback | Eleven fresh attempts, ten valid current seals: nine complete-unqualified, one failed/sealed, one export-refused/unsealed. Every sealed inventory/digest and operational Bash fingerprint reread (evidence/bundle-readback.json). |
| Records and source readback | Six reported records validate; four schema negatives refuse; measured template unpublished. Twenty source metric ratios/counts match, source IDs equal across arms; one changed numerator rejected (evidence/record-controls.json, source-readback.json, source-negative-control.json). |
| Operational byte scope | Bash/jq/YAML/backend/pins unchanged from reviewed baseline; only PS operational bytes changed. Final operational files equal internal candidate bytes (evidence/static-fixes.json). |

Exact commands and full logs are outside the archive, in the assembly workbench `.temp/review-fixes/`: run-final-current.sh, controls-current.py, fidelity.py, static.py, run-tier.sh and aggregation.py are task-local adaptations of existing private verification methods, not shipped tooling. Inputs retain the original verification/ fixtures. `results/lane-exits.txt` and `results/tier-exits.txt` give each lane's result. Each selected lane ran; no first-failure stop hid another lane.

The initial fidelity check incorrectly read an in-progress wire log and failed the request-count assertion. The final rerun after all shard lanes completed passed. Initial failure evidence remains private in results/fidelity.log; corrected evidence is results/fidelity-rerun.log. The evaluator venv lacks jsonschema; system Python3's installed jsonschema validated the records instead, without changing the evaluator environment. These limits are not hidden successes.

## New integrity and archive receipt

MANIFEST.sha256 is freshly computed over every kit file except itself, including INVENTORY.tsv and this document. INVENTORY.tsv lists every file and its exact size, including itself and MANIFEST.sha256. Unchanged tiers keep their original source seal; SOURCE-SHA256SUMS refers to original documentary inputs, never adapted bytes.

Send **`final-kit.tar.gz` + `final-kit.tar.gz.sha256` together**. Verify the detached receipt before extraction, then MANIFEST.sha256 and tiers/SHA256SUMS after extraction. The archive's own hash cannot be embedded in its contents without circularity; its exact digest and fresh-extraction results are in the assembly-root `../FIX3-REPORT.md` and detached receipt. Final packaging verifies every hash, inventory size and extracted byte against this tree, plus Bash syntax, tier seals and export scan. No temporary caches, bundles, fixtures, logs, symlinks or baseline originals are shipped.

## Packaging verification scope

Current packaging checks cover fresh-extraction MANIFEST.sha256 and inventory size readback, every JSON/JSONL parse, tiers/SHA256SUMS, bash -n for all four Bash files, exact extraction/tree equality, operational-byte equality against the preserved fix2 archive, export scan and preserved-baseline hash readbacks. The fix2 archive and detached receipt are retained read-only under ../review-baseline/fix2/; archive SHA256: `a50b9c644e3834d7137f80a9aba73e5e10c8280b29a18a7ed3642120775974b0`.

The internal execution results above are retained evidence. The full stub control set is not rerun; operational files remain byte-for-byte equal to fix2. MANIFEST.sha256, INVENTORY.tsv and the detached archive receipt are recomputed for the current bytes; unchanged source/tier seals retain their original meaning. Current packaging results and archive size/SHA256 are recorded in ../FIX3-REPORT.md. Fresh independent re-review is pending; no landing or delivery is claimed.
