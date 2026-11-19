# Benchmarking patches: design decisions

How `bench/` measures a patch, and why. For usage, see [bench/README.md](../bench/README.md).

## Aims

- **Trustworthy and cheap to run**, so many people can contribute runs.
- **Close to stock lm-eval**, so other standard benchmarks can be added the same way.
- **Not tied to one machine or model.**
- **Few inventions.** Every setting comes from a named source: lm-eval, the base model's card, or the patch author's published evaluation.

## Decisions

**Stock lm-eval tasks and scorers.** bench adds only what lm-eval lacks:
- frozen item lists with lm-eval's own item hashes, because stock tasks don't pin a dataset revision;
- a recorder that keeps every request and reply;
- filing and checking of results.

**Settings come from the base model's card.** This covers sampling and output length, and the benchmark prompt where the card gives one. For Qwen3.5 the card says to add "Please reason step by step, and put your final answer within \boxed{}." to math prompts, so its settings replace the stock MATH-500 prompt. The card doesn't say where the sentence goes, so it follows the problem after one newline, as in RikiyaT's evaluation. Models whose card gives no prompt use the stock one.

**Base and patched run as a pair, under the same seed.** Indexed patches are small. Facing the same random draws cancels much of the sampling noise from their difference. The design is called common random numbers.

**One seed for everything, for now.** One fixed seed per run is common practice; ours is 1234, lm-eval's default. Several passes per item are possible later by adding seeds to the protocol. Pairs already filed stay valid.

**No coordination between contributors.** Each session picks items at random among those filed the fewest times. Overlaps are allowed: the first filing counts, and later ones are kept as reproducibility checks. Maintainers can instead pick fixed item ranges and a seed.

**Batched requests by default; no byte-identical reruns.** Running one request at a time makes reruns byte-identical, but only on the same hardware and vLLM build, and it wastes most of a large GPU. Results are therefore checked statistically, as sampled benchmarks usually are. Batching doesn't break the pairing: base and patched still get the same random draws. Their replies split within the first few hundred characters anyway, because the weights differ.

**One model in memory at a time by default.** A session runs base on its items, then patched on the same items. For large models most people can't hold both. `--parallel` runs both at once when they fit.

**Baselines are paired per patch.** The index's general rule is one baseline per base profile, shared by all patches. A paired baseline must be measured in the same sessions as its patch, so bench files one per patch, next to the shared ones.

**Cut replies are scored as lm-eval scores them (owner decision: Timur, 2026-11-19).** When a reply hits the token limit before `</think>`, lm-eval scores the whole thinking trace and can count it correct. That score stays the headline, unchanged. The share of replies cut at the token limit is shown right beside it: a completed run records it as `cut_at_limit` next to the score's value, and `compare` reports the same share. `compare` also reports, as the secondary number, a strict score that counts cut replies as wrong. An incomplete run keeps its `UNKNOWN` score and records no cut share as final.

**All run data stays in git (owner decision: Timur, 2026-11-19).** Sessions, samples and the full request/response logs are filed in this repository. If checkout size ever becomes a problem, the `.jsonl` files get gzipped rather than hosted elsewhere.

## Hardware

Any GPU that stock vLLM supports: NVIDIA with compute capability 7.0 or newer (RTX 20-series, T4 and up), or Apple Silicon through vllm-metal. Older cards such as the GTX 1070 are not supported. They would also need fp16 instead of the pinned bf16 weights, which changes results.

## Open questions

- **Output cap:** the card's 32,768 for most queries, or its 81,920 for math competitions. Both models often loop until the cap on hard items.
- **How many seeds per item.**
- **Comparison records in the index:** the index can't host per-item comparison data yet, so `compare` prints to the terminal. Fixing that needs a change to the index tool.
- **Running `bench check` in CI.**
- **More benchmarks:** MMLU-Pro is prepared but not run.

## Combination studies

Any set of patches can be applied together, so any set can be benchmarked; the site shows only the stacks someone reported. Each study is one JSON file under `bench/combinations/<namespace>/<model>/`. It names two or more members by patch slug, version, artifact filename and scale, and declares its profile, benchmark, protocol, item count (`denominator`), metric, unit, preferred direction and performer. File the unpatched baseline and the score of the whole stack. Also file each member's score alone under the same conditions whenever you can: either every member has an `alone` score or none does. File full measurement conditions in `protocol` and keep experiment outputs with the benchmark contribution.

The site reads each study as changes from the base. With solo scores, a card first compares the stack with its best patch alone: is stacking worth it over applying that one patch? It then compares the stack with the solo changes added up: the patches help each other, add up, or get in each other's way. Without solo scores, a card can only compare the stack with the base. For percentages the site draws an approximate 95% range from the item count, treating every score as an independent sample, and calls a gap inside that range no difference. When that range is wider than either patch's own effect, the sum reading is "too close to call". Metrics in other units get no range. The model page lists the measured stacks, sorted by how far each beats its best patch alone, and has a picker for any set of its active patches. A patch page lists the studies of that release, including older partners. The build rejects unavailable artifacts, mismatched profiles, repeated patches and a mix of members with and without solo scores. See the [synthetic studies](example/bench/combinations/example-org/Example-Model-1B/).
