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

**Cut replies are scored as lm-eval scores them.** When a reply hits the token limit before `</think>`, lm-eval scores the whole thinking trace and can count it correct. That score is kept as the headline. `compare` also reports the cut share, and a score that counts cut replies as wrong.

## Hardware

Any GPU that stock vLLM supports: NVIDIA with compute capability 7.0 or newer (RTX 20-series, T4 and up), or Apple Silicon through vllm-metal. Older cards such as the GTX 1070 are not supported. They would also need fp16 instead of the pinned bf16 weights, which changes results.

## Open questions

- **Output cap:** the card's 32,768 for most queries, or its 81,920 for math competitions. Both models often loop until the cap on hard items.
- **How many seeds per item.**
- **Comparison records in the index:** the index can't host per-item comparison data yet, so `compare` prints to the terminal. Fixing that needs a change to the index tool.
- **Running `bench check` in CI.**
- **More benchmarks:** MMLU-Pro is prepared but not run.

## Combination studies

The model and patch pages can display a measured combination from one JSON file per study under `bench/combinations/<namespace>/<model>/`. Each study declares its profile, benchmark, protocol, sample count, metric, unit, preferred direction and performer. File the observed baseline and joint score plus an independent observation of each member alone under the same evaluation conditions. Identify each member by patch slug, version, artifact filename and scale. File full measurement conditions in `protocol` and keep experiment outputs with the benchmark contribution.

The display shows each treatment on one axis and compares the observed joint score with the best solo score. This descriptive comparison makes no claim of statistical significance. A reader can inspect the source record and exact artifacts. The build rejects unavailable artifacts, mismatched profiles, repeated patches and missing treatment values. See the [synthetic four-treatment example](example/bench/combinations/example-org/Example-Model-1B/arithmetic.json).
