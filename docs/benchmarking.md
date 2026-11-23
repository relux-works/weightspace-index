# Benchmarking patches: design decisions

How `bench/` measures a patch, and why. For usage, see [bench/README.md](../bench/README.md).

## Aims

- **Trustworthy and cheap to run**, so many people can contribute runs.
- **Close to stock lm-eval**, so other standard benchmarks can be added the same way.
- **Not tied to one machine or model.**
- **Few inventions.** Every setting comes from a named source: lm-eval, the base model's card, or, only where the card is silent, the patch author's published evaluation.

## Decisions

**Stock lm-eval tasks and scorers.** bench adds only what lm-eval lacks:
- frozen item lists with lm-eval's own item hashes, because stock tasks don't pin a dataset revision;
- a recorder that keeps every request and reply;
- filing and checking of results.

**Settings come from the base model's card.** This covers sampling and output length, and the benchmark prompt where the card gives one. For Qwen3.5 the card says to add "Please reason step by step, and put your final answer within \boxed{}." to math prompts, so its settings replace the stock MATH-500 prompt. The card doesn't say where the sentence goes, so it follows the problem after one newline, as in the evaluation of the thinking-loops patch by its authors. Models whose card gives no prompt use the stock one.

**The card's settings win over the patch author's (owner decision: Timur, 2026-11-20).** To be fair to every patch, sampling follows the base model's maker, not the settings a patch author evaluated with. For Qwen3.5-0.8B that is the card's recommendation for "thinking mode for text tasks": temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence penalty 1.5, repetition penalty 1.0. Qwen ran its own text benchmarks on the card with the same sampling, including the HMMT math benchmarks (the notes under the card's Language results). The card gives temperature 0.6 in two other places:
- its recommendation for thinking mode on vision-language or precise-coding tasks, there with presence penalty 0;
- the notes under its Vision Language results: every vision-language benchmark except video ran in thinking mode at temperature 0.6 with presence penalty 1.5.

RikiyaT's team evaluated the thinking-loops patch at temperature 0.6 with presence penalty 1.5, the combination in those Vision Language notes. Their code describes it as the card's thinking-mode benchmark setting. The card has not changed since 2026-03-02, so they read the same card. Results under an author's own settings may be filed as a separate settings file, but they never replace the card's. The comparison with the patch authors' evaluation is [below](#compared-with-the-patch-authors-evaluation).

**The card warns that Qwen3.5-0.8B loops in thinking mode.** Its Thinking Mode section says that with the recommended sampling this model falls into thinking loops more often than the other Qwen3.5 models, which can keep it from finishing. The card suggests tuning the sampling for each use case and stopping such replies early. bench keeps the recommended settings, so a score shows how the model does when run as its maker recommends, loops included. A tuned setting can be measured as a separate settings file.

**Output is capped at 32,768 tokens (owner decision: Timur, 2026-11-20).** This is the card's figure "for most queries". For math and programming competitions the card suggests 81,920, and bench does not follow it. Qwen3.5-0.8B loops until the cap on most hard items, so a longer cap would mostly buy longer loops at more than twice the compute. Because cut replies count as wrong (below), the cap is part of what a score means: solved within 32,768 tokens.

**Base and patched run as a pair, under the same seed.** Indexed patches are small. Facing the same random draws cancels much of the sampling noise from their difference. The design is called common random numbers.

**One seed for everything, for now.** One fixed seed per run is common practice; ours is 1234, lm-eval's default. Several passes per item are possible later by adding seeds to the protocol. Pairs already filed stay valid.

**No coordination between contributors.** Each session picks items at random among those filed the fewest times. Overlaps are allowed: the first filing counts, and later ones are kept as reproducibility checks. Maintainers can instead pick fixed item ranges and a seed.

**Batched requests by default; no byte-identical reruns.** Running one request at a time makes reruns byte-identical, but only on the same hardware and vLLM build, and it wastes most of a large GPU. Results are therefore checked statistically, as sampled benchmarks usually are. Batching doesn't break the pairing: base and patched still get the same random draws. Their replies split within the first few hundred characters anyway, because the weights differ.

**One model in memory at a time by default.** A session runs base on its items, then patched on the same items. For large models most people can't hold both. `--parallel` runs both at once when they fit.

**Recovery after the base half.** Bench writes a checkpoint after a managed sequential base run. Patched-only recovery verifies the saved replies, checkpoint hashes and pairing inputs against the new session before starting inference. The local recorder uses the operating system's listen backlog to absorb lm-eval's connection bursts.

**Baselines are paired per patch.** The index's general rule is one baseline per base profile, shared by all patches. A paired baseline must be measured in the same sessions as its patch, so bench files one per patch, next to the shared ones.

**The headline grades each reply's final answer (owner decision: Timur, 2026-11-20; replaces the 2026-11-19 decision to keep lm-eval's score as the headline).** The final answer is the inside of the last `\boxed{}` after the last `</think>`. A reply has no answer, and counts as wrong, when it never closed its thinking, when no box follows `</think>`, or when its last box is unclosed or empty. Nothing else in the text is read as an answer. math-verify compares the two boxed answers and nothing else, `verify(parse("\boxed{gold}"), parse("\boxed{answer}"))`. The gold is MATH-500's `answer` field, proven against the protocol's frozen target hash. This is how the original MATH evaluation reads an answer (the last box). The evaluation of the thinking-loops patch by its authors also grades the last box with math-verify. When `</think>` is missing, though, it reads the whole reply, so a box inside a cut thinking trace can count (see the [comparison](#compared-with-the-patch-authors-evaluation)).

Why: when `</think>` is missing, lm-eval scores the whole thinking trace, and `math_verify` falls back from `\boxed{}` to any answer-like expression or plain number in it. On the first 52 `math500-n197` pairs of the Qwen3.5-0.8B patch, lm-eval credited 14 cut replies; 12 of them held no `\boxed{}` answer, only a loop in which `math_verify` found a matching number. Such credit measures how a trace was cut, not whether the problem was solved.

- A reply that closed its thinking and boxed an answer before the token limit cut it counts as answered.
- The original MATH checker, `is_equiv` (lm-eval's copy), grades the same final answers as a cross-check.
- bench grades from the recorded replies, so `check` grades every reply again. For that, bench imports the pinned math-verify and lm-eval (owner approval, 2026-11-20).
- A math-verify timeout stops grading instead of counting as wrong, since a timed-out grade would depend on the machine's speed.
- A completed run records the headline as the score's `value`. Beside it are lm-eval's own score as `value_lm_eval`, the share of replies cut at the token limit as `cut_at_limit`, the share that finished without an answer as `finished_without_answer`, and `mean_completion_tokens`.
- `compare` reports the same figures and the MATH checker's score. `python -m bench disagreements` lists every reply where lm-eval, math-verify and the MATH checker disagree.
- An incomplete run keeps its `UNKNOWN` score.

The site's patch charts show provisional scores once at least 25% of the planned item–seed pairs have been filed for both base and patched models. The headline averages seeds within each filed item, then weights the items equally. The cut share uses all filed pairs. The charts show the filed count until the run is complete. Final scores come from the completed run records.

**All run data stays in git (owner decision: Timur, 2026-11-19).** Sessions, samples and the full request/response logs are filed in this repository. If checkout size ever becomes a problem, the `.jsonl` files get gzipped rather than hosted elsewhere.

## IFEval protocol and verification

IFEval measures whether the thinking-loops patch improves instruction following. Each prompt has one or more machine-checkable instructions. Use all 541 prompts in the official `google/IFEval` evaluation split, which the dataset names `train`. Pin the dataset revision, lm-eval task version, and per-item hashes. Send the stock prompt verbatim as one user message with zero examples. Keep the stock instruction checkers and pin their dependencies, sentence tokenizer tables, and language detection seed. [IFEval paper](https://arxiv.org/abs/2311.07911) · [lm-eval task](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/ifeval/ifeval.yaml)

Use the same paired design as MATH-500. The base and patched BF16 models answer the same prompts under the same seed and Qwen text-thinking settings, including a 32,768-token cap. The outcome is one if every official check passes on the visible reply after the last `</think>`, and zero if any check fails or the model never produces a visible reply. If a reply hits the cap after thinking ends, grade the text that reached the user and also record the cut. Verify the frozen prompt and instruction metadata against the logged document, and recompute the headline and the stock lm-eval score from the recorded reply. [Patch authors’ grading code](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/code/evaluate.py#L96-L114)

The headline is strict prompt-level accuracy. Weight each prompt equally, averaging its seeds first if more are added. Report paired differences, counts of discordant pairs, token-limit hits, replies that stop without an answer, and reply length. The authors’ reported 21.0% and 62.3% used two samples per prompt and temperature 0.6. The index uses the model card's temperature of 1.0 and starts with one seed, so the measured scores describe this protocol. [Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/#make-a-language-model-better-at-math)

The implementation freezes the official task and adds its grader to the existing recording, checking and merging path. Tests cover passing replies, failed instructions and unfinished thinking, and rejection of altered prompts, instruction metadata and scores.

The first standard session ran prompts 0–9 with seed 1234 and concurrency 5 on an M1 Pro. The base passed 3 of 10 and hit the 32,768-token cap on 3 replies. The patched model passed 6 of 10 and hit the cap on 1 reply. The session covers 10 of the 541 prompts. [Session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261120T182333Z-56a1bd/session.json)

**Literal character counts.** The pinned lm-eval letter-frequency checker randomly substitutes an ASCII letter when it receives punctuation. An audit of all 541 prompts found two affected items, 19 for `#` and 22 for `!`. Bench now counts the supplied character using the existing threshold rule, both in lm-eval runs and offline replay. Replay preserves each older session's recorded lm-eval diagnostic and verifies the unaffected instruction verdicts and their aggregation. The headline uses the corrected grading for all sessions.

Replay recovered the second session from its saved replies. Prompts 10–19 ran with seed 1234 and concurrency 8. Both checkpoints passed 7 of 10. One base reply hit the cap, and all patched replies finished within it. The two sessions give 10 of 20 base passes and 13 of 20 patched passes. [Second session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261121T050545Z-34d090/session.json)

The third session reused the verified base replies for prompts 20–39 and completed the patched half with seed 1234 and concurrency 8. Strict passes were 10 of 20 for base and 11 of 20 for patched. The base hit the token cap on six replies and the patch hit it on one. Across the first 40 prompts, strict passes total 20 for base and 24 for patched. [Third session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261121T075956Z-da71d8/session.json)

The fourth session ran prompts 40–89 with seed 1234 and concurrency 8. The base passed 30 of 50 and hit the token cap on seven replies. The patched model passed 26 of 50 and finished all replies within the cap. Across the 90 prompts filed so far, both checkpoints passed 50. The base hit the cap on 17 replies and the patched model hit it on two. [Fourth session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261121T113909Z-054de0/session.json)

The fifth session ran prompts 90–139 with seed 1234 and concurrency 8. The base passed 30 of 50 and hit the token cap on six replies. The patched model passed 25 of 50 and hit the cap on one reply. Across the 140 prompts filed so far, the base passed 80 and the patched model passed 75. The base hit the cap on 23 replies and the patched model hit it on three. [Fifth session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261121T161449Z-f252fb/session.json)

The sixth session ran prompts 140–179 with seed 1234 and concurrency 8. The base passed 24 of 40 and hit the token cap on six replies. The patched model passed 15 of 40 and hit the cap on three. Across the 180 prompts filed so far, the base passed 104 and the patched model passed 90. The base hit the cap on 29 replies and the patched model hit it on six. [Sixth session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261121T193908Z-3000ad/session.json)

The seventh session ran prompts 180–279 with seed 1234 and concurrency 8. The base passed 65 of 100 and hit the token cap on 18 replies. The patched model passed 54 of 100 and finished all replies within the cap. Across the 280 prompts filed so far, the base passed 169 and the patched model passed 144. The base hit the cap on 47 replies and the patched model hit it on six. [Seventh session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/ifeval.qwen3.5-0.8b-thinking.bf16/20261121T223745Z-6c5c2a/session.json)

## Compared with the patch authors' evaluation

RikiyaT's team evaluated the Qwen3.5-0.8B thinking-loops patch (`rikiyat-thinking-loops` v1) with their own code, published with the patch's logs: `evaluate.py` (protocol `think35`) and `mathqa.py`. On MATH-500 the two set-ups differ as below. The card is revision `2fc0636` of `Qwen/Qwen3.5-0.8B`, unchanged since 2026-03-02.

| Setting | Our bench | Patch authors | Why our bench uses its value |
|---|---|---|---|
| Thinking | on | on | — |
| Prompt | problem, a newline, then the card's math sentence | problem, a newline, then the card's math sentence | Card, Best Practices. The placement follows the patch authors. |
| Temperature | **1.0** | **0.6** | Card: thinking mode for text tasks, and the sampling of Qwen's own text benchmarks. The patch authors' 0.6 matches the card's Vision Language benchmark notes. |
| top_p | 0.95 | 0.95 | — |
| top_k | 20 | 20 | — |
| min_p | 0 | not set (vLLM default, 0) | — |
| Presence penalty | 1.5 | 1.5 | Card, thinking mode for text tasks. The card's 0.6 recommendation comes with presence penalty 0. |
| Repetition penalty | 1.0 | not set (vLLM default, 1.0) | — |
| Max new tokens | 32,768 | 32,768 | Card: "for most queries" (decision above). |
| Weights | full bf16 checkpoint (`Qwen3_5ForConditionalGeneration`; text prompts don't use the vision encoder) | text-only bf16 export (`Qwen3_5ForCausalLM`) | The index's pinned base weights. |
| Engine | vLLM server (vllm-metal and CUDA), queried by lm-eval's chat-completions client | vLLM offline engine (`LLM.generate`) | Stock lm-eval tasks (decision above). |
| Seed | 1234 | 0 | lm-eval's default (decision above). |
| Samples per problem | 1 | 4, averaged | One seed for now (decision above). |
| Problems | 197 of 500 (`math500-n197`, stratified by subject and level) | all 500 | The protocol's frozen sample. |
| Answer read | last `\boxed{}` after the last `</think>`. No `</think>` means no answer. | last `\boxed{}`, `\fbox{}`, `\boxed {..}` or `\boxed 5` after the last `</think>`, or anywhere in the reply when there is no `</think>` | Cut replies count as wrong (decision above). |
| Comparison with the gold answer | math-verify | math-verify | — |

Temperature, the prompts and seeds sampled, answer extraction, and the inference engine can all affect scores. A temperature-only test would hold the other conditions fixed. The [version 1 benchmark note](../bench/notes/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1.md) compares the gap at both temperatures and is shown beside the site’s benchmark results.

The [`qwen3.5-0.8b-thinking-t0.6`](../bench/settings/qwen3.5-0.8b-thinking-t0.6.json) settings are that test. They match the card's settings except for temperature 0.6, the patch authors' value, and are filed beside the card's run, never in place of it. All 197 items are filed, with seed 1234. Three sessions of 10 items ran with concurrency 10 on an M1 Pro. Two sessions of 84 and 83 items ran with concurrency 64 on a rented A100 SXM4 40 GB. The base answered 15 of 197 and hit the token cap on 182 replies. The patched model answered 96 of 197, hit the cap on 68, and finished without an answer on 11. The difference is +41.1 points (95% interval +34.2 to +48.0). The patched model alone was right on 81 items, the base alone on none. Fourteen patched replies that hit the cap end in a run of spaces, two of 30 on the M1 Pro and 12 of 167 on the A100. [First session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/math500-n197.qwen3.5-0.8b-thinking-t0.6.bf16/20261122T132808Z-f4dd5d/session.json) · [Second session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/math500-n197.qwen3.5-0.8b-thinking-t0.6.bf16/20261122T161946Z-37b005/session.json) · [Third session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/math500-n197.qwen3.5-0.8b-thinking-t0.6.bf16/20261122T191426Z-c46c81/session.json) · [Fourth session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/math500-n197.qwen3.5-0.8b-thinking-t0.6.bf16/20261122T230348Z-ed3230/session.json) · [Fifth session record](../bench/results/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/math500-n197.qwen3.5-0.8b-thinking-t0.6.bf16/20261122T233420Z-e19bed/session.json)

## Hardware

Any GPU that stock vLLM supports: NVIDIA with compute capability 7.0 or newer (RTX 20-series, T4 and up), or Apple Silicon through vllm-metal. Older cards such as the GTX 1070 are not supported. They would also need fp16 instead of the pinned bf16 weights, which changes results.

## Open questions

- **How many seeds per item.**
- **Comparison records in the index:** the index can't host per-item comparison data yet, so `compare` prints to the terminal. Fixing that needs a change to the index tool.
- **Running `bench check` in CI.**
- **More benchmarks:** MMLU-Pro is prepared but not run.

## Combination studies

Any set of patches can be applied together, so any set can be benchmarked; the site shows only the stacks someone reported. Each study is one JSON file under `bench/combinations/<namespace>/<model>/`. It names two or more members by patch slug, version, artifact filename and scale, and declares its profile, benchmark, protocol, item count (`denominator`), metric, unit, preferred direction and performer. File the unpatched baseline and the score of the whole stack. Also file each member's score alone under the same conditions whenever you can: either every member has an `alone` score or none does. File full measurement conditions in `protocol` and keep experiment outputs with the benchmark contribution.

The model page's Compare patches section shows each study as rows in the Measured group of its benchmark: the stack's score next to the unpatched base, listed after the single patches (pairs before triples). A member's solo score fills that patch's row when the patch has no result of its own for the same benchmark and item count. Only stacks of the patches' active releases are shown, and a patch page names the stacks that include it. The build rejects unavailable artifacts, mismatched profiles, repeated patches and a mix of members with and without solo scores. See the [synthetic studies](example/bench/combinations/example-org/Example-Model-1B/).
