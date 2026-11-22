# Bench: base versus patched, in pairs

Design decisions and open questions: [docs/benchmarking.md](../docs/benchmarking.md).

Benchmarks a patch by running the base model and the patched copy on the
same frozen items, with the same settings and **the same sampling seed**,
using **stock lm-eval** (`lm-eval run`, its own task definitions and
scorers). One command does everything: it prepares both models, serves
each in turn, runs lm-eval against it, verifies the result and files it.

Anyone can run a few items and commit what they filed. Nobody has to
coordinate. Each session picks items at random among those filed the
fewest times, so contributors rarely overlap. When they do, the first
filing counts and later ones act as reproducibility checks.

What this folder adds around lm-eval is small:

| Piece | Why lm-eval alone isn't enough |
|---|---|
| `protocols/*.json`: task, shot count, frozen item IDs with lm-eval's own `doc_hash` and `target_hash`, seeds | Stock tasks don't pin a dataset revision. The hashes make any dataset drift fail loudly instead of silently changing items. |
| `settings/*.json`: the base model card's recommended sampling settings and, where the card gives one, its benchmark prompt; the card is pinned by revision and SHA-256 | Sampling goes in as plain `--gen_kwargs` and `--model_args`; a prompt goes in through a task file that includes the stock task. |
| A loopback recorder between lm-eval and the server | lm-eval keeps only the reply text. The recorder keeps what was actually sent and returned: every sampling parameter and the seed, `finish_reason`, token counts and the full thinking trace. |
| `run`, `status`, `merge`, `check`, `compare`, `disagreements` (`python -m bench …`) | Pairing, filing, verifying and combining sessions. |

Weights are prepared with the index's own tool (`weightspace apply`).
This folder never edits weights.

Release-specific benchmark notes live at `notes/<namespace>/<model>/<patch-slug>/v<N>.md`.
They cite run records and published protocols, and the site displays them with that version’s results.

## Why pairs, and why one seed

The patches indexed so far are small. A small effect is easier to see
when both models face the same random draws. Base and patched therefore
answer each item under the same seed, which removes much of the sampling
noise from their difference. This is the common-random-numbers design.

Sampling stays the card's sampling. The seed only fixes which random
numbers are drawn.

**Batching.** By default a server gets all of a session's items at once
(`--concurrency`). Batched requests change the last bits of the logits,
so a reply also depends on what else shared its batch, and a rerun of the
same item and seed is not byte-identical. On vllm-metal 0.33, two
same-seed replies of the same weights split at character 303 with 10
requests in flight, and never at one at a time. On vllm-metal the random
draws depend only on the seed and the output position, so base and
patched still face the same draws (not yet checked on CUDA vLLM). Their replies split early anyway (after 55 and 455 characters
on two items at one request at a time), because the weights differ.

Byte-identical reruns would hold only on the same hardware and vLLM
build, so results are checked statistically instead, as sampled
benchmarks usually are. `--concurrency 1` is still available.

**Seeds.** A protocol lists its seeds. `math500-n197` starts with one,
`[1234]`, which is lm-eval's own default and what most published runs do:
one fixed seed for the whole benchmark. Adding seeds later (several
passes per item) only appends to that list. The protocol's digest leaves
the list out, so pairs already filed stay valid. A run is complete when
every item has a pair under every listed seed. Its score is the mean over
items of each item's mean over seeds.

## Protocol `ifeval`

- **Task and prompts** use stock lm-eval `ifeval` v4.0 with all 541 official prompts from the pinned [google/IFEval split](https://huggingface.co/datasets/google/IFEval/tree/966cd89545d6b6acfd7638bc708b98261ca58e84), named `train` in the dataset. Each prompt is sent verbatim as one user message, with zero examples and no task stop strings.
- **Headline** is strict prompt-level accuracy. A nonempty visible reply must pass every instruction check. When thinking is enabled, the grader reads the text after the last `</think>`. Missing or empty visible replies score zero. A reply cut after thinking is graded on the visible text already returned. vLLM must return raw content including the think-end token, as it does by default without a reasoning parser.
- **Verification** checks the logged prompt and instruction metadata against the frozen document, and replays strict grading with pinned packages, a fixed language detector seed, and pinned NLTK tokenizer tables. The recorded lm-eval score is retained alongside the headline because lm-eval can see thinking when the end token is missing.
- **Literal character counts** use the character supplied by IFEval, including `#` and `!`. The pinned lm-eval checker otherwise substitutes a random ASCII letter. New sessions record the corrected checker version. For older sessions, replay preserves the logged lm-eval score, verifies other instruction verdicts, and deterministically grades the literal count for the headline.
- **Comparison** uses the same seed and settings on base and patched. It reports strict passes and their difference over the prompts filed, alongside token-limit hits and reply lengths. Only a run covering every prompt under every protocol seed gets an index score.

The standard Qwen run uses `--protocol ifeval --settings qwen3.5-0.8b-thinking`. Use the command under [Running](#running) with those options.

## Protocol `math500-n197`

- **Task:** `minerva_math500` (lm-eval task version 3.1), on HuggingFaceH4/MATH-500 at revision `6e4ed1a…`.
  `hendrycks_math500` uses the same data with a different prompt and scorer.
  `minerva_math500` reports `math_verify`, which is the primary metric here. bench files it as lm-eval's own score, beside bench's grade of each reply's final answer (see below).
- **Items:** the 197 items in [`data/math500-ids.json`](data/math500-ids.json) (frozen selection, seed `20261117-supplement2-v1`).
- **Two deliberate changes to the stock task, both made through lm-eval's own options:**
  - `--num_fewshot 0`: the worked examples would otherwise reach a thinking model as earlier chat turns.
  - `until: []`: the stock stop string `Problem:` also shows up inside thinking traces and cut them mid-thought.
- **Prompt:** the stock task's, unless the model's settings file carries its card's prompt for this task (see below).
- **Score:** unweighted, over the frozen items. This describes these items; it is not a population-weighted estimate.

## Settings `qwen3.5-0.8b-thinking`

These come from the Qwen3.5-0.8B card, thinking mode for text tasks:

| Setting | Value |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 20 |
| `min_p` | 0 |
| `presence_penalty` | 1.5 |
| `repetition_penalty` | 1.0 |
| `enable_thinking` | true |
| Math prompt | the problem, a newline, then "Please reason step by step, and put your final answer within \boxed{}." |

The card's Best Practices give that sentence for math benchmarks but don't say where it goes. It follows the problem after one newline, as in the patch authors' evaluation. It replaces the stock `minerva_math500` prompt (`Problem:` … `Solution:`), which was written for few-shot base models.

**Prompt overrides.** By default a protocol runs the stock lm-eval prompt. A settings file can replace it for one task with its card's prompt (`task_overrides.<task>.doc_to_text`, nothing else). `run` then writes a two-line task file that `include`s the installed stock task and sets only `doc_to_text`; this is lm-eval's own mechanism. Data, scorer and frozen item hashes stay the stock task's. `run` checks that lm-eval resolved exactly this prompt. Another model whose card says nothing about prompts keeps the stock prompt. So prompts can differ between models on the same protocol, but never between a base and its patch.

Output is capped at 32,768 tokens, the card's figure for most queries. The card suggests 81,920 for math competitions; bench keeps 32,768, because the model loops until the cap on most hard items (see [docs/benchmarking.md](../docs/benchmarking.md)). The patch author evaluated at temperature 0.6; bench uses the card's 1.0 for text tasks.

**The MATH-500 headline grades each reply's final answer** using the inside of the last `\boxed{}` after the last `</think>`. No `</think>` (a reply cut while thinking), no box after it, or an unclosed or empty last box means no answer, which counts as wrong. math-verify then compares only the two boxed answers, `verify(parse("\boxed{gold}"), parse("\boxed{answer}"))`. The gold is MATH-500's `answer` field, which must hash to the protocol's frozen target. The original MATH checker (`is_equiv`) grades the same answers as a cross-check. lm-eval's own score differs: when `</think>` is missing it scores the whole trace, and `math_verify` falls back to any number it finds there.

A cut is recorded as `finish_reason: "length"`. A completed run's score carries the headline as `value`. Beside it are lm-eval's own score as `value_lm_eval`, the share of cut replies as `cut_at_limit`, the share that finished without an answer as `finished_without_answer`, and `mean_completion_tokens`. Each outcome records its `answer` and `correct` next to lm-eval's `score`. MATH-500 also records `correct_is_equiv`.

A settings file applies only to the base model whose card it cites. Another model gets its own file, copied from its own card.

## Running

1. **Set up a project venv** at `.venv` in the repository root (git-ignored), with Python 3.12 and [uv](https://docs.astral.sh/uv/):

   ```bash
   uv venv -p 3.12 .venv
   ```

   On a Mac with Apple Silicon (macOS 15 or later), install the harness and vLLM together from the lock file. It pins every package by exact version and SHA-256 — except `mlx-lm` and `mlx-vlm`, which vllm-metal takes from git, pinned there by commit — including the vllm-metal 0.33.0 and vLLM 0.33.0 release wheels:

   ```bash
   uv pip install -p .venv -r bench/requirements-macos.txt
   ```

   On Linux x86_64 with an NVIDIA GPU, install from the CUDA lock. It pins the harness and the vLLM 0.33.0 release from PyPI with every dependency by exact version and SHA-256. Its PyTorch build uses CUDA 13.0, so the GPU driver must support CUDA 13 (driver 580 or newer):

   ```bash
   uv pip install -p .venv -r bench/requirements-cuda.txt
   ```

   Elsewhere, install the harness (exact pins; `run` refuses other versions) and the vLLM build for your hardware into the same venv:

   ```bash
   uv pip install -p .venv -r bench/requirements.txt
   ```

   The macOS lock includes the IFEval packages. On other platforms, add them with `uv pip install -p .venv -r bench/requirements-ifeval.txt`. For running or checking IFEval, install the pinned sentence tokenizer once in the same environment.

   ```bash
   mkdir -p .venv/nltk_data/tokenizers
   curl --fail --location https://raw.githubusercontent.com/nltk/nltk_data/4f15a3d89eefe9748ec1c05be495d91289197155/packages/tokenizers/punkt_tab.zip -o .venv/punkt_tab.zip
   unzip -q .venv/punkt_tab.zip -d .venv/nltk_data/tokenizers
   ```

   Nothing needs activating. Every command below runs `.venv/bin/python`, and `run` starts the `vllm` installed beside it.

2. **Run a session.** On a Mac:

   ```bash
   .venv/bin/python -m bench run --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking --server-arg=--language-model-only --server-arg=--kv-cache-memory-bytes=4294967296 --hardware "M1 Pro 32 GB, vllm-metal 0.33.0"
   ```

   On an NVIDIA GPU:

   ```bash
   .venv/bin/python -m bench run --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking --hardware "RTX 4090, vllm 0.33"
   ```

   `--hardware` is free text filed with the session; give your real chip, memory and vLLM build.

   This:
   - downloads the base at the profile's pinned revision into `.temp/bench/models/`, unless `--base-dir` names a copy you already have;
   - builds the patched copy with `weightspace apply` (needs `cargo`), unless `--patched-dir` names one;
   - checks that the base weights are the profile's exact files and that every patched weights file differs from them;
   - picks a seed and `--count` items (default 10) at random among the pairs filed the fewest times;
   - for base, then patched: starts `vllm serve` on 127.0.0.1, waits for it, checks it serves that folder with enough context, runs `lm-eval run` on all the session's items through the recorder, and stops the server. Only one model is in memory at a time;
   - verifies both halves the same way `check` does and files the session.

   Useful options:
   - `--items 0:20 --seed 1234` picks items deterministically instead (a slice of the protocol's item order). Maintainers can use this to split the work by hand.
   - `--dry-run` prints the selection and stops.
   - `--parallel` serves base and patched at the same time and runs both halves at once. Use it only when both models fit in memory. It saves time when a single request leaves the hardware partly idle, as on a laptop.
   - `--reuse-base PATH` resumes from a completed base half in an earlier session work folder under `.temp/bench/sessions/`. Supply the same explicit item slice, seed, concurrency, hardware, and managed-server arguments. Bench verifies the saved base inputs and replies before starting the patched server. The new paired session records where the base half came from.
   - `--server-arg=--flag` passes an extra flag to `vllm serve` (repeatable). Use it for memory flags, for example `--server-arg=--gpu-memory-utilization=0.8`, or for Qwen3.5 on vllm-metal: `--server-arg=--language-model-only`. Don't pass local paths: the flags are recorded. `--api-key` is refused in any spelling, because a key here would be filed publicly. The servers listen only on 127.0.0.1; if you want a key anyway, set `VLLM_API_KEY`: vLLM reads it, and the recorder forwards it without recording it.
   - `--vllm PATH` points to another `vllm` executable. By default `run` uses the one in its own venv, else the one on `PATH`.
   - `--base-port` and `--patched-port` use servers you started yourself on 127.0.0.1. They must serve `--base-dir` and `--patched-dir`.
   - `--timeout` (default 7200 s) is how long lm-eval waits for one reply. Its own default, 300 s, is too short for long thinking replies.

   Each session is filed as:

   ```text
   bench/results/<namespace>/<model>/<patch>/v<N>/<protocol>.<settings>.<profile>/<UTC time>-<random>/
     session.json            seed, items, weights and model-file digests, harness versions, servers, who ran it
     base.samples.jsonl      lm-eval's own per-item log, unchanged
     base.requests.jsonl     every request and response, as sent and returned
     patched.samples.jsonl
     patched.requests.jsonl
   ```

   A failed or interrupted session files nothing.

3. **Commit the session folder** and open a pull request.

## Status, merging, checking, comparing

```bash
.venv/bin/python -m bench status --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking
```

```bash
.venv/bin/python -m bench merge
```

```bash
.venv/bin/python -m bench check
```

```bash
.venv/bin/python -m bench compare --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking
```

- **`status`** shows how many pairs are filed, open and repeated.
- **`merge`** writes two runs into the index, in its existing run format (`run.json` and `outcomes.jsonl`):
  - the patched run, in the version folder: `models/…/<patch>/v<N>/runs/<protocol>.<settings>.<profile>/`;
  - its paired baseline, in the base folder: `models/…/runs/<protocol>.<settings>.<profile>.<patch>-v<N>/`.

  Each run names the other. A run stays `INCOMPLETE`, with an `UNKNOWN` score and no cut share, until every item has a pair under every seed; then its score is the graded final answer, with lm-eval's own score (`value_lm_eval`), `cut_at_limit`, `finished_without_answer` and `mean_completion_tokens` beside it. Then regenerate the index tables with `weightspace generate`.
- **`check`** rebuilds every run from its sessions and fails on any difference. For every item, in both halves of a session, the lm-eval sample must:
  - be one of the session's frozen items, with the frozen hashes;
  - match exactly one recorded request, sent with exactly the protocol's and settings' parameters and the session's seed;
  - have scored the reply that the server actually returned;
  - carry the frozen gold answer for MATH-500 or the frozen prompt and instruction metadata for IFEval.

  `check` grades every final answer again. It uses the installed scorer pins and IFEval tokenizer tables offline. A math-verify timeout stops grading. Base and patched must differ only in the weights files, and every session must report the same harness and model files.
- **`compare`** works on whatever pairs are filed and says whether they are complete. The unit is the item: the patched-minus-base score, averaged over the item's seeds. It reports:
  - both headline scores, from the graded final answers;
  - the mean paired difference and counts of base-only and patch-only passes;
  - the share of replies cut at the token limit, the share that finished without an answer, how many replies gave an answer and were then cut, and mean reply length;
  - lm-eval's own scores and difference (`score_lm_eval`, `lm_eval`), and how many replies only lm-eval scored as right, or only the final answer;
  - for MATH-500, standard errors and 95% intervals for both differences, the original MATH checker's scores (`score_is_equiv`), and how many replies it grades differently from math-verify;
  - how many pairs came back byte-identical, and how many repeated filings reproduced the first (expected only at `--concurrency 1` on the same setup).
- **`disagreements`** lists replies where the final-answer grade differs from lm-eval or the MATH cross-check. Each line includes the final answer, scores, thinking status and a scored-text excerpt. MATH-500 includes the gold answer and boxes found in that text.

## Limits

- **A session is the contributor's report.** `check` proves it is internally consistent: right items, right parameters and seed, scored reply equals returned reply. It cannot prove the replies came from the stated weights. Anyone can rerun the same items and seed: the scores should agree statistically. Replies come back byte-identical only at `--concurrency 1` on the same hardware and vLLM build.
- **Same seed is reproducible only within one setup.** Different hardware or vLLM builds draw different random numbers from the same seed. Pairing still holds within each session, because base and patched run on the same machine with the same settings.
- **No comparison file in the index yet.** The index's `comparisons/` folder can't host the per-item data, so `compare` prints to the terminal.
- **Not run in CI yet.** `check` needs no model or network, only the pinned packages, but the CI workflow doesn't call it yet.
- **MATH-500 only reads `\boxed{...}` answers.** Other spellings (`\fbox{...}`, `\boxed 5`) give no answer. The card's prompt asks for `\boxed{}`.

## Updating the macOS and CUDA locks

`requirements-macos.txt` is generated from `requirements-macos.in`, which names the harness pins and the two vLLM release wheels. After changing either file, regenerate it from the `bench/` folder:

```bash
MACOSX_DEPLOYMENT_TARGET=15.0 uv pip compile -p 3.12 --python-platform aarch64-apple-darwin --generate-hashes --no-header requirements-macos.in -o requirements-macos.txt
```

`requirements-cuda.txt` is generated the same way from `requirements-cuda.in`:

```bash
uv pip compile -p 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header requirements-cuda.in -o requirements-cuda.txt
```

## Tests

```bash
uv pip install -p .venv pytest
```

```bash
.venv/bin/python -m pytest bench/tests -q
```

The MATH-500 tests run the real `lm-eval run` against a fake `vllm` executable (an OpenAI-shaped stub) in a temporary repository. They cover:
- freezing a protocol;
- deterministic and random selection;
- running and filing sessions, repeats, merging, comparing;
- final-answer grading: a cut reply with a boxed guess in its trace, and a finished reply with an unboxed answer, both credited by lm-eval and graded wrong, and listed by `disagreements`;
- reading the final answer (after `</think>`, last box, unclosed and empty boxes), math-verify and MATH checker verdicts, and a refused timeout;
- detection of tampered parameters, seeds, items, gold answers, replies, swapped weights and extra requests.

They need the MATH-500 test split, which is cached after the first download. They don't exercise the download or `weightspace apply` steps.

The IFEval tests replay synthetic instruction checks and paired results, including unfinished thinking and changes to frozen documents, prompts and grader verdicts. They need the pinned scorer and tokenizer tables.
