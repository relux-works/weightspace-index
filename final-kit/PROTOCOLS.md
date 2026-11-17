# Frozen pilot protocols — 2026-11-17

The index defines core specifications. Default ranking: #1 MATH-500, #2 upstream **generative** MMLU-Pro (`mmlu_pro`, 14 children); IFEval is documented fallback only. `leaderboard_mmlu_pro` is a different likelihood task and is never substituted. Owner defaults supersede older kit2 IFEval-first text and both consultancy proposals wherever they conflict.

## SUITE series (ladder)

ID: `lm_eval-0.4.13-qwen35-text-thinking-32k-math500-mmlupro-tiers-v2` (SUPPLEMENT2 v2 ID).

| Field | Upstream identity / disclosed override |
|---|---|
| Harness, task code, metrics | lm_eval 0.4.13, installed upstream tasks; no task Python supplied or modified |
| MATH dataset | HuggingFaceH4/MATH-500, default/test, revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be, N=500 |
| MATH prompt and few-shot | Verbatim upstream `Problem:` / `Solution:` formatting, four fixed demonstrations and role construction; no card suffix |
| MATH stop and grading | Upstream `Problem:` stop, process_docs/process_results, exact_match and independent math_verify; no last-box-only filter and no cap-hit-wrong override |
| MMLU dataset | TIGER-Lab/MMLU-Pro, test, revision b189ec765aa7ed75c8acfea42df31fdae71f97be, N=12032 across 14 categories |
| MMLU elicitation | Verbatim description, five category validation demonstrations, options/prompt, role ordering, `Question:` stop |
| MMLU grading | Upstream strict `exact_match,custom-extract`; free text, no likelihood ranking or custom judge |
| Wrapper change | Dataset revision pin only; group inheritance selects installed upstream task |
| Generation overlay | do_sample=true; temperature=1.0, top_p=.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=true |
| Budget | max_gen_toks=32768, mapped to HTTP max_tokens=32768; no 81920-token runs |
| Backend controls | local-chat-completions; tokenizer_backend=null, tokenized_requests=false; server renders messages; batch_size=1, num_concurrent=12 per endpoint, timeout=14400, max_retries=1 (one total attempt) |
| Seeds | Harness 0/1234/1234/DRAW_SEED, default request/draw seed 1234; A/A second arm seed 1235 |
| Context / answer channel | Server must fit actual prompt + 32768 output; verify correct tokenizer/template, EOS and content versus reasoning_content. Untokenized client does not fit context or preserve separate reasoning/usage |
| Selection / estimator | Frozen stratified n197/R1 per benchmark; preserve weights, exact global IDs, derived child-local IDs and prompt hashes; completed-tier result only |

Suite correctness belongs to the pinned upstream scorer. A correctly scored capped response remains whatever that scorer says; report cap hits separately. Both MATH metrics are descriptive; retain their disagreement. Tier summaries emitted by the comparator use full stratum weights for exact_match; retain per-sample math_verify for separately weighted reanalysis.

## Author-reproduction eligibility

Candidate source protocol: `rikiyat-p50c-think35-32768-source-4d09ba6-v1`. This is a documentary source ID, not an executable runner protocol. RIKIYAT-REPRODUCTION.md pins the author HF card, RikiyaT's published Experiment 8 article, p50c manifests, configs/code and base/edited results, exact available item IDs, repetitions, extraction and cut definition. **All metrics are not-comparable** with the existing kit task/backend machinery; no automatic lane is enabled and no scorer/backend is invented.

The old `qwen35-0.8b-card-replication-32k-v1` task/selection remain only as retired internal diagnostic compatibility fixtures. They do not define the author's protocol. The 12-item card smoke and manual worksheet are removed from the Mac runbook; no exploratory hand-run is a reference or acceptance target. Legacy cap-hit-wrong policy is not RikiyaT's scoring: their correctness is independent of their finish_reason-based cut_share.

Generation `max_gen_toks` is passed through `--gen_kwargs` and mapped by lm_eval 0.4.13 to HTTP `max_tokens`. It limits generated output, not context. Context additionally needs room for the prompt. Without finish_reason, a cap alone cannot establish cap hits; retain null/missing classification. Only actual uniquely joined response metadata can support RikiyaT's cut definition.

Qwen card claims are separately transcribed in REPORTED.md and reported-claims.template.json; these are not RikiyaT results. SUITE extends coverage and is not a reproduction verdict for either source.

## Tiers, uncertainty and verdicts

Default: **197 items × R1 per benchmark**, seed `20261117-supplement2-v1`, frozen before outcomes. `plans/screening-197-R1.tsv` gives exact offsets in these lists, not raw dataset prefixes. 17 shards per benchmark: 16×12 + 5, alternating servers A/B; A=101 and B=96 items per benchmark per arm. Completed base+patch pairs define coverage. A/A controls are separate small shards, not extra R draws in the screening denominator.

Planning sensitivity is approximately 10 pp at 80% power under q=.25 and homogeneous-within-stratum scenario assumptions (see tiers/STATISTICS.md). It is neither a universal guarantee nor measured variance. Do not borrow tinyBenchmarks/IRT reconstruction accuracy for these raw stratified means. These tiers have no demonstrated subset-vs-full calibration for patched-model deltas or ranking.

Before completion, publish progress counts only. Publish scores only at completed frozen tiers; label `n197/R1 frozen-tier result, cap 32768`, never full benchmark score. Full-score claims require 500/12032 coverage and their appropriate estimator. Full-population and R8 alternatives are described in tiers/README.md and are not defaults. Never add draws or stop at an attractive outcome. No unadjusted inference across repeated nested looks; predeclare multiplicity allocation or a valid sequential design before such use.

Keep benchmark verdicts separate: improved / regression / inconclusive are legitimate outcomes. No pooled quality percentage. For pilot, report weighted paired effect, wins/losses/discordance, stratum-aware paired confidence interval and assumptions; **no equivalence or “quality preserved” claim**. A noninferiority margin remains an owner decision (`UNSET`); a future preserved-quality claim needs that margin fixed before results and a one-sided noninferiority analysis. The comparator emits descriptive counts/pairs, not a CI or a significance verdict. Manual analysis/re-score remains owed. Resample whole items within strata, keeping paired arms/draws together; R8 clusters draws by item. Zero discordance does not justify a zero-uncertainty claim from a degenerate bootstrap.

## Evidence, trust and lineage

Bundles start with `performed_by=index` (Timur's own pilot runs) and `trust_level=submitted`. Allowed role vocabulary: index / author / third-party. Trust progression is evidence-driven: submitted → re-scored → independently replicated → reference-run; none is inferred from seal validity. Official scores are recomputed from submitted per-sample logs using pinned dependencies (RESCORE.md), never accepted solely from submitted totals.

Record base weight manifest digest, patch digest (`none — base` in pilot), tokenizer and actual template digests, quantization bf16, conversion versions or original-BF16 declaration, patch-before/after-quantization status, runtime versions and layout evidence. Patch application stage is `not applicable — base` here; future patched arms explicitly record before/after. Changed precision/config belongs to a separate series, never averaged. Preserve engine response evidence for finish_reason and prompt/completion/total tokens; null means missing, never zero. Manual evidence recovery is required when upstream logs omit fields.

Author-side public-item adaptation and patch selection can bias results. Repeat execution cannot undo that; stronger general claims need frozen patch and fresh/protected confirmation. In-flight duplicates cannot be inferred from completed bundles. Deterministic first-valid-attempt selection must be declared before scores, with all other attempts retained privately and not included in comparison roots.
