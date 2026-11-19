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
| `run`, `status`, `merge`, `check`, `compare` (`python -m bench …`) | Pairing, filing, verifying and combining sessions. |

Weights are prepared with the index's own tool (`weightspace apply`).
This folder never edits weights.

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

## Protocol `math500-n197`

- **Task:** `minerva_math500` (lm-eval task version 3.1), on HuggingFaceH4/MATH-500 at revision `6e4ed1a…`.
  `hendrycks_math500` uses the same data with a different prompt and scorer.
  `minerva_math500` reports `math_verify`, which is the primary metric here.
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

The card's Best Practices give that sentence for math benchmarks but don't say where it goes. It follows the problem after one newline, as in RikiyaT's evaluation of this patch. It replaces the stock `minerva_math500` prompt (`Problem:` … `Solution:`), which was written for few-shot base models.

**Prompt overrides.** By default a protocol runs the stock lm-eval prompt. A settings file can replace it for one task with its card's prompt (`task_overrides.<task>.doc_to_text`, nothing else). `run` then writes a two-line task file that `include`s the installed stock task and sets only `doc_to_text`; this is lm-eval's own mechanism. Data, scorer and frozen item hashes stay the stock task's. `run` checks that lm-eval resolved exactly this prompt. Another model whose card says nothing about prompts keeps the stock prompt. So prompts can differ between models on the same protocol, but never between a base and its patch.

Output is capped at 32,768 tokens, the card's figure for most queries. For math competitions the card suggests 81,920, so the cap is still an open choice.

lm-eval scores the text after `</think>`, its stock `think_end_token`. A reply cut before `</think>` is scored the way lm-eval scores it: the whole trace. The cut is recorded as `finish_reason: "length"`. A completed run's score carries the share of cut replies as `cut_at_limit`, beside the headline value; `compare` reports the same share and, as its secondary number, a score that counts cut replies as wrong.

A settings file applies only to the base model whose card it cites. Another model gets its own file, copied from its own card.

## Running

1. **Install the harness** (exact pins; `run` refuses other versions), plus vLLM for your hardware:

   ```bash
   python -m pip install -r bench/requirements.txt
   ```

2. **Run a session:**

   ```bash
   python -m bench run --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking --hardware "RTX 4090, vllm 0.33"
   ```

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
   - `--server-arg=--flag` passes an extra flag to `vllm serve` (repeatable). Use it for memory flags, for example `--server-arg=--gpu-memory-utilization=0.8`, or for Qwen3.5 on vllm-metal: `--server-arg=--language-model-only`. Don't pass local paths: the flags are recorded. `--api-key` is refused in any spelling, because a key here would be filed publicly. The servers listen only on 127.0.0.1; if you want a key anyway, set `VLLM_API_KEY`: vLLM reads it, and the recorder forwards it without recording it.
   - `--vllm PATH` points to the `vllm` executable.
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
python -m bench status --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking
```

```bash
python -m bench merge
```

```bash
python -m bench check
```

```bash
python -m bench compare --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --protocol math500-n197 --settings qwen3.5-0.8b-thinking
```

- **`status`** shows how many pairs are filed, open and repeated.
- **`merge`** writes two runs into the index, in its existing run format (`run.json` and `outcomes.jsonl`):
  - the patched run, in the version folder: `models/…/<patch>/v<N>/runs/<protocol>.<settings>.<profile>/`;
  - its paired baseline, in the base folder: `models/…/runs/<protocol>.<settings>.<profile>.<patch>-v<N>/`.

  Each run names the other. A run stays `INCOMPLETE`, with an `UNKNOWN` score and no cut share, until every item has a pair under every seed; then its score is lm-eval's score with `cut_at_limit` beside it. Then regenerate the index tables with `weightspace generate`.
- **`check`** rebuilds every run from its sessions and fails on any difference. For every item, in both halves of a session, the lm-eval sample must:
  - be one of the session's frozen items, with the frozen hashes;
  - match exactly one recorded request, sent with exactly the protocol's and settings' parameters and the session's seed;
  - have scored the reply that the server actually returned.

  Base and patched must also differ only in the weights files, and every session must report the same harness and the same model files.
- **`compare`** works on whatever pairs are filed and says whether they are complete. The unit is the item: the patched-minus-base score, averaged over the item's seeds. It reports:
  - both scores;
  - the mean difference with its standard error across items and a 95% interval;
  - the share of replies cut at the token limit, and scores that count those as wrong;
  - mean reply length;
  - how many pairs came back byte-identical, and how many repeated filings reproduced the first (expected only at `--concurrency 1` on the same setup).

## Limits

- **A session is the contributor's report.** `check` proves it is internally consistent: right items, right parameters and seed, scored reply equals returned reply. It cannot prove the replies came from the stated weights. Anyone can rerun the same items and seed: the scores should agree statistically. Replies come back byte-identical only at `--concurrency 1` on the same hardware and vLLM build.
- **Same seed is reproducible only within one setup.** Different hardware or vLLM builds draw different random numbers from the same seed. Pairing still holds within each session, because base and patched run on the same machine with the same settings.
- **No comparison file in the index yet.** The index's `comparisons/` folder can't host the per-item data, so `compare` prints to the terminal.
- **Not run in CI yet.** `check` needs no model or network, but the CI workflow doesn't call it yet.

## Tests

```bash
python -m pip install pytest
```

```bash
python -m pytest bench/tests -q
```

The tests run the real `lm-eval run` against a fake `vllm` executable (an OpenAI-shaped stub) in a temporary repository. They cover:
- freezing a protocol;
- deterministic and random selection;
- running and filing sessions, repeats, merging, comparing;
- detection of tampered parameters, seeds, items, replies, swapped weights and extra requests.

They need the MATH-500 test split, which is cached after the first download. They don't exercise the download or `weightspace apply` steps.
