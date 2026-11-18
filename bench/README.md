# Frozen base versus edited benchmark

One Python command downloads the entry's exact pinned public base, validates and
applies its replacement tensors into a separate copy, then serves base and edited
weights sequentially on loopback using installed vLLM. It runs stock lm-eval
`local-chat-completions` on identical frozen rows and settings. The base is never
edited. No API key is used. Downloads need internet on the first real run.

**Mac/vLLM-metal execution UNVERIFIED; developed and stub-tested on Linux; no score is claimed until you run it.**

## Install on Mac

Use Python 3.12+ in an environment with working vLLM / vLLM-metal OpenAI serving.
For Apple Silicon, follow the upstream [vLLM-metal installation instructions](https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/).
Install this package into that same Python environment; installing the benchmark
alone does not install a hardware-specific vLLM build. In a fresh environment:

```sh
python3.12 -m venv .temp/bench-venv
source .temp/bench-venv/bin/activate
# Install the appropriate upstream vLLM-metal build into this environment first.
python -m pip install -e bench
python -m bench run --entry models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1 --benchmark math500
```

Use `--benchmark mmlu-pro` for generative MMLU-Pro. The runner detects the Python
vLLM OpenAI server module, otherwise the installed `vllm serve` command. It records
exact argv and PID, checks `/v1/models` with a bounded timeout, and terminates only
its own PID/process group, including exceptions, SIGINT and SIGTERM. An occupied
port is refused. A workdir lock prevents overlapping runs on the same files.

## Frozen data and scoring

`data/math500-ids.json` and `data/mmlu-pro-ids.json` are byte-for-byte copies of
`minerva_math500/n197.json` and `mmlu_pro/n197.json` from the reviewed final-kit
197/R1 frozen tier inventory, seed `20261117-supplement2-v1`. No selection was
re-derived. Each contains 197 global test-row IDs and its exact dataset revision.
MMLU-Pro runs against the global test split, without category filtering. Evaluation
visits selected rows in ascending ID order; the original frozen data stays intact.

The task prompts, filters and scorers come from pinned `lm_eval==0.4.13`:
`minerva_math500` and the generative MMLU-Pro default template. Both use explicit
zero-shot prompting here. The latter is one global task, not the category group.
The primary score is the task's `exact_match`; MATH also retains `math_verify`
in the original per-item sample. The reported aggregate is the plain unweighted
mean across all frozen items and repetitions, with edited minus base as delta.
It describes these selected rows, not a population-weighted estimate. The copied
tier's stratum metadata remains intact for provenance and is not used as weights.

`config.json` explicitly sets 32768 output tokens, sampling parameters, stop/EOS
policy, repetitions, seeds, prompt policy, timeouts and serving settings. The
same config is used for both variants and its original text is saved verbatim.
Repetitions are complete evaluations, each with seed `seed + repetition`, averaged
equally. Chat formatting is performed by the server using the identical copied
base tokenizer; model generation-config overrides are disabled with `vllm` policy.

## Outputs and reruns

Default workdir is `.temp/bench`; change it with `--workdir`. It contains verified
`base/` and `edited/` caches, one server log per variant, and:

- `results/<benchmark>-base.json`
- `results/<benchmark>-edited.json`
- `results/<benchmark>-compare.json`

Run records contain frozen data, generations, documents, actual prompts, filtered
answers, per-item scores, aggregate, model/replacement identities, lm-eval version,
timestamps, server argv/PID and the KV cache setting (null when unset).
Stock lm-eval local-chat drops endpoint `finish_reason`; it is recorded as
`unknown` even if the endpoint returned it. The backend is not forked.

Missing/empty generations, wrong counts/IDs, HTTP errors, changed caches and
incompatible tensor names/shapes/dtypes fail with nonzero exit. A valid base run
may remain if edited fails; no compare file or final table is produced then.
Existing results are never overwritten. Archive the `results/` directory before
rerunning with `--skip-download --skip-apply`; both caches must match their seals.
A fresh workdir performs a new download/apply. `--port` defaults to 8000;
`--kv-cache-memory-bytes` passes through only when set. The optional `--base-url`
must be an HTTP chat-completions URL on numeric loopback with that same port;
external servers are not supported.

```sh
python -m bench --rescore .temp/bench/results/math500-base.json --output .temp/bench/results/math500-base-rescored.json
```

`rescore` is an alias spelling without the leading dashes. It replays saved raw
generations through the original stock filters/scorers and recomputes the mean,
using saved documents with no inference or dataset download. It requires the
original lm-eval version; missing/incompatible scorer support is an error, never
a claimed rescore. The input record is preserved.

## Tools and offline checks

Python runs the benchmark; pip installs its declared dependencies; vLLM serves
weights; huggingface_hub downloads pinned snapshots; safetensors validates/applies
replacements; lm-eval generates and scores; pytest checks fixtures. Runtime output
lives under the selected workdir. No benchmark dataset or model is included.

```sh
python -m pip install -e 'bench[test]'
python -m pytest bench/tests -q
```

Tests use tiny synthetic tensors, saved-row fixtures and a loopback HTTP stub,
including the real `simple_evaluate` and stock local-chat backend. These checks do
not establish real checkpoint compatibility, Mac performance, GPU support, memory
requirements or score quality. No real model/dataset was downloaded in development.
