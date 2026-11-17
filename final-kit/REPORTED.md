# Author-reported claims and measured records

`reported-claims.template.json` contains six filled, source-pinned **author-reported** Qwen3.5-0.8B language-table records. They are factual transcription examples, not measured runs. Source: [Qwen card at 2fc0636](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md), retrieved 2026-11-17.

| Benchmark | Non-thinking | Thinking |
|---|---:|---:|
| MMLU-Pro | 29.7% | 42.3% |
| IFEval | 52.1% | 44.0% |
| IFBench | unspecified | 21.0% |
| GPQA | unspecified | 11.9% |

The language table reports temperature 1.0, top_p .95, top_k 20 and presence_penalty 1.5. This table note is not a complete per-metric recipe: prompts, exact splits, metric subtypes, repetitions and seeds remain unspecified. Do not replace it with the card's separate VL settings or general deployment recommendations. No MATH-500 value is supplied in these language rows. The RikiyaT p50c claims are a separate source/protocol (RIKIYAT-REPRODUCTION.md).

## Record contract

`reported-claim.schema.json` defines the JSON structure and status enum for each individual record in the collection. Protocol comparability and bidirectional links still require evidence review; structural validation alone never proves them.

Each JSON record has `schema_version`, `record_type: author-reported` (the reported record), immutable `claim_id`, stable `model_id`, `patch_id`, claimed identity, source URL/revision/retrieval or publication date, benchmark and variant, value/unit/exact metric description, protocol ID, reported settings or literal `unspecified`, and `recorded_by`. Preserve unknowns; never infer a metric subtype from the benchmark name. A corrected claim gets a new ID with an explicit supersession link; do not rewrite published source meaning.

Verification vocabulary is exactly:

- `unverified`: no eligible comparable measured evidence yet; current filled records have no verification runs.
- `reproduced`: comparable recipe, coverage, model/patch identity and metric; difference within a tolerance/uncertainty rule declared before looking at outcomes. Store the rule and run evidence.
- `not-reproduced`: comparable recipe and sufficient completed evidence fail that declared rule. Integrity failure or missing data alone cannot establish this verdict.
- `not-comparable`: a documented recipe, identity, coverage or scoring mismatch prevents a verification verdict. Unknown recipes cannot become reproduced by matching a number.

`pilot_comparability` records the current kit's compatibility decision separately from a claim's evidentiary status. All filled claims are `unverified`; this pilot is `not-comparable` for verifying them. No inference or verification run is fabricated.

Claim → run: `verification_run_ids`; run → claim: `claim_ids` plus `claim_verifications` entries containing `claim_id`, status, rule, reason and relative evidence references. Resolve every ID to a small JSON summary; require both directions to agree before publication. Empty arrays mean no linked run. `trust_level` describes evidence custody, independently of claim verification; seals prove integrity, not reproduction.

## Which pilot protocol can address which claims?

| Protocol | Coverage | Verification of supplied author claims |
|---|---|---|
| SUITE v2, 197/R1, cap 32768 | MATH-500 and generative MMLU-Pro extension measurements | MMLU-Pro thinking is the same benchmark family, but prompt, repeats, exact recipe and partial-tier coverage do not establish the card number. Non-thinking MMLU and all IFEval/IFBench/GPQA claims are outside this lane. No supplied Qwen MATH claim exists to verify. |
| RikiyaT author-reproduction candidate | Extracted think35 protocol, all ten source metrics | No faithful executable lane is available with the current task/backend machinery; each metric is `not-comparable`. See the per-metric decisions. No replacement scorer/backend is added. |

The former 12-item card series and manual card worksheet are retired from the Mac pilot. They address none of these claims and are no reference protocol or target.

## Website file contract

`run-summary.template.json` is an **index-measured** template, deliberately with null score values and no actual runs or invented hashes. On completion, fill `scores` with objects containing `metric`, `value`, `unit`, `estimator`, `uncertainty`, `output_cap_tokens` and `coverage_ref`; preserve separate MATH metrics. Fill claim verdicts only with eligible evidence. Before completion expose coverage/progress, not interim scores. Trust vocabulary: submitted / re-scored / independently replicated / reference-run.

Stable IDs and relative paths link models, patches, protocols, claims, runs and bundle manifests. Reference heavy artifacts by path and verified content SHA256; never embed raw outputs in summaries. Hash a bundle's manifest/seal/archive as separate objects; null is missing and prohibits publication as verified. A filled summary must set `template_only: false`, have complete required fields, resolve references and pass the existing export scan. A patch page groups its run summaries by `patch_id`; all reading is possible by scanning JSON files, with no database.

Display author-reported values as a distinct reference series, visibly labelled with source and verification status. Keep `record_type` on collection and records. Never merge or average author-reported values with index-measured scores. A source's reported independent rerun remains author-reported until the index verifies its evidence; it is not an index measurement.
