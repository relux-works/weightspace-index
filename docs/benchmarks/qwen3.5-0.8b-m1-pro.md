# Qwen3.5-0.8B inference on M1 Pro

How fast the unpatched BF16 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
runs with [MLX-LM](https://github.com/ml-explore/mlx-lm) on an M1 Pro, from 512
to 131,072 input tokens, and how much memory it needs. Use it as a baseline for
what to expect on Apple Silicon, or rerun the same sweep on your own machine
and compare.

This is a hardware/runtime measurement, not a patch-quality evaluation or an
index assessment. It does not compare engines or quantizations.

## Results

Batch size 1, 256 generated tokens, averages of three timed trials after one
warm-up. Per-trial output is in the [`context-*.txt`](qwen3.5-0.8b-m1-pro/)
files.

| Input tokens | prompt_tps | ≈ first token (s) | generation_tps | MLX peak (GB) | total_time (s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 1,330.2 | 0.38 | 81.8 | 2.298 | 3.45–3.64 |
| 2,048 | 2,138.5 | 0.96 | 83.7 | 2.837 | 4.00–4.08 |
| 8,192 | 2,064.1 | 3.97 | 81.5 | 3.171 | 7.06–7.17 |
| 16,384 | 1,857.5 | 8.82 | 75.7 | 3.842 | 12.11–12.33 |
| 32,768 | 1,514.4 | 21.6 | 69.4 | 5.184 | 25.23–25.42 |
| 65,536 | 1,070.5 | 61.2 | 55.5 | 5.762 | 65.42–66.10 |
| 131,072 | 623.9 | 210.1 | 44.4 | 8.983 | 214.4–216.9 |

What this means in practice:

- **Short prompts feel instant.** Up to ~8k tokens the first token arrives in
  under 4 s and output streams at ~82 tokens/s.
- **Long prompts are dominated by prefill.** At 131k tokens you wait about
  3.5 minutes for the first token; prefill runs at ~29% of its 2k-token rate.
- **Decode slows with context**, from ~82 to ~44 tokens/s across the sweep.
- **Memory is not the limit on a 32 GB machine.** MLX's peak allocation stays
  under 9 GB at 131k tokens (whole-process footprint ~10 GiB, see below), well
  inside the 26.8 GB recommended Metal working set. A 16 GB Mac should handle
  the 32k–64k range comfortably.

Prefill throughput peaks at 2,048 tokens because that is the prefill step size.

### Metrics

These follow MLX-LM's own timing convention:

- `prompt_tps`: input tokens divided by time until the first generated token
  (excluding model load and tokenization). First-token time in the table is
  input tokens ÷ `prompt_tps`.
- `generation_tps`: decode throughput, including the Python
  streaming/detokenization path. MLX-LM counts all 256 output tokens, although
  the first is produced during prompt timing.
- `peak_memory`: MLX's process-lifetime peak allocation in decimal GB,
  including warm-up. Not whole-process memory.
- `total_time`: wall time of one timed generation call, range across the three
  trials.

### Whole-process memory at 131,072 tokens

`peak_memory` counts only MLX allocations. During the 131,072-token run the
process's physical footprint (Activity Monitor's “Memory”, via
`top -stats mem`) and RSS (`ps -o rss`) were sampled every ~2–4 s:
[`context-131072.footprint.tsv`](qwen3.5-0.8b-m1-pro/context-131072.footprint.tsv).

- Peak physical footprint was **~10 GiB** (highest MiB-resolution sample:
  9,999 MiB; `top` rounds to whole GiB above that). This matches MLX's
  8.98 GB (8.37 GiB) plus the process's non-MLX memory.
- RSS peaked at 1.69 GiB. RSS does not count Metal allocations in unified
  memory, so don't use it to size a machine.

Only this context was sampled; for the others, use `peak_memory` plus ~1.5 GiB
as a rough estimate.

## Setup

- **Model:** Qwen/Qwen3.5-0.8B, revision
  `2fc06364715b967f1860aea9cf38778875588b17` (the index's
  [`bf16` profile](../../models/Qwen/Qwen3.5-0.8B/base.json)), original BF16
  weights. No quantization, KV quantization, adapters, speculative decoding or
  images. MLX-LM loads the text path only (vision encoder and MTP tensors are
  excluded).
- **Runtime:** MLX-LM 0.32.2, MLX / MLX Metal 0.32.4, Transformers 5.23.0,
  Python 3.12. Full pinned list:
  [`requirements.txt`](qwen3.5-0.8b-m1-pro/requirements.txt).
- **Machine:** MacBook Pro (`MacBookPro18,3`), M1 Pro, 8 CPU cores (6P + 2E),
  14 GPU cores, 32 GB unified memory, macOS 27.2.1, on AC power, no thermal
  warnings before the run. Ordinary desktop apps were open; this was not an
  idle machine.
- **Input:** random token IDs (MLX seed 0), greedy decoding, EOS ignored, so
  every run produces exactly 256 tokens. This measures throughput, not answer
  quality or end-to-end chat latency.
- **Protocol:** each context in a fresh process, run in ascending order; one
  warm-up and three timed trials, 3 s pause before each; prefill step 2,048;
  no prompt/KV cache reuse. Model load time excluded. The model's advertised
  262,144-token context was not tested.

## Reproduce

Needs Apple Silicon, [uv](https://docs.astral.sh/uv/), and network access for
the install and download; the measurement itself runs offline. Run from the
repository root — everything goes into the git-ignored `.temp/`.

Install the pinned runtime and fetch the checkpoint:

```sh
uv venv --python python3.12 .temp/mlx-bench
uv pip install --python .temp/mlx-bench/bin/python \
  -r docs/benchmarks/qwen3.5-0.8b-m1-pro/requirements.txt

.temp/mlx-bench/bin/hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir .temp/qwen3.5-0.8b-bf16 \
  config.json model.safetensors.index.json \
  model.safetensors-00001-of-00001.safetensors \
  tokenizer.json tokenizer_config.json vocab.json merges.txt \
  chat_template.jinja LICENSE
```

Verify the weights (size 1,746,942,600 bytes); the digest must be
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
Digests for every downloaded file are in
[`inputs.json`](qwen3.5-0.8b-m1-pro/inputs.json).

```sh
shasum -a 256 \
  .temp/qwen3.5-0.8b-bf16/model.safetensors-00001-of-00001.safetensors
```

Run the sweep. `caffeinate -i` keeps the Mac from idle-sleeping during the
~10-minute 131k run; plug into AC power and close heavy apps for the most
comparable numbers. Drop contexts from the list to run a shorter sweep.

```sh
mkdir -p .temp/qwen-bench
caffeinate -i /bin/zsh -c '
set -e -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for ctx in 512 2048 8192 16384 32768 65536 131072; do
  .temp/mlx-bench/bin/python -u -m mlx_lm.benchmark \
    --model .temp/qwen3.5-0.8b-bf16 \
    --prompt-tokens "$ctx" --generation-tokens 256 \
    --batch-size 1 --num-trials 3 --prefill-step-size 2048 --delay 3 \
    2>&1 | tee ".temp/qwen-bench/context-${ctx}.txt"
done
'
```

The last line of each output file (`Averages: ...`) corresponds to one row of
the table above. If Metal is unavailable (for example inside a sandbox or VM),
MLX falls back to CPU and the numbers are not comparable — don't report them
against this table.

## Files

In [`qwen3.5-0.8b-m1-pro/`](qwen3.5-0.8b-m1-pro/):

- `context-<tokens>.txt` — raw `mlx_lm.benchmark` output per context.
- `context-131072.footprint.tsv` — process memory samples (seconds since
  sampling started, RSS in KB, physical footprint).
- `environment.json` — machine, OS, Metal device limits, protocol and
  limitations in machine-readable form.
- `requirements.txt` — exact package versions.
- `inputs.json` — size and SHA-256 of every model/tokenizer file, plus
  SHA-256 of the MLX-LM source files that define the benchmark and model.
