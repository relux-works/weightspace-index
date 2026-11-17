# Final Mac pilot kit — 2026-11-17

Run MATH-500 first and generative MMLU-Pro second, using two vLLM-metal BF16 servers and the frozen **197/R1** screen. Every score must state **32768 output-token cap**. SUITE measurements and author-reported references are separate records (PROTOCOLS.md, REPORTED.md). Mac/vLLM execution is **UNVERIFIED**. PowerShell syntax/execution is **NOT RUN / UNVERIFIED**.

**Author-protocol decision:** RIKIYAT-REPRODUCTION.md extracts the pinned p50c recipe. All source metrics are **not-comparable** with the current task/backend machinery; no automatic author-reproduction lane is enabled. The old 12-item card smoke and manual card worksheet are retired from the Mac pilot and are not acceptance gates or targets. Continue the independent suite steps below. Author-reported values and website JSON templates are in REPORTED.md.

**Transfer unit:** send `final-kit.tar.gz` **together with** `final-kit.tar.gz.sha256`. From the directory containing both, run `shasum -a 256 -c final-kit.tar.gz.sha256` before extraction, then verify the extracted `MANIFEST.sha256` before creating logs or installing. An archive without its detached receipt is an incomplete delivery.

## Prerequisites and install

Apple Silicon macOS, existing Python 3.12, Bash 3.2+, jq 1.6+, curl, shasum, tar. Your vLLM **0.33.0 + vllm-metal already installed** remain separate; this installer changes neither server environment. Check tools without dumping environment or credential values:

```bash
cd final-kit
shasum -a 256 -c MANIFEST.sha256 > manifest-check.log
bash --version > bash-version.txt
jq --version > jq-version.txt
python3.12 --version > python-version.txt
PYTHON_BIN=python3.12 bash setup/install.sh
export PILOT_PYTHON="$PWD/.venv/bin/python"
mkdir -p receipts
curl --fail --silent http://127.0.0.1:8000/v1/models | jq '{models:[.data[].id]}' > receipts/models-A.json
curl --fail --silent http://127.0.0.1:8001/v1/models | jq '{models:[.data[].id]}' > receipts/models-B.json
export MODEL_ID='REPLACE_WITH_EXACT_SERVED_ID'
cp lineage-template.json receipts/lineage.json
# Fill every REQUIRED field from actual weight/tokenizer/template/version evidence.
export PROFILE_FILE="$PWD/receipts/lineage.json"
```

Use sorted per-file SHA256 manifests to identify multiple weight/tokenizer files, then hash those manifests; record the actual template bytes separately. Set patch_digest=`none — base`, quantization=`bf16`, and patch_quantization_order=`not applicable — base`. Supply observed engine/plugin versions, conversion versions (or original BF16), and layout evidence. The runner rejects template placeholders. Different served aliases are allowed only with true model lineage recorded; labels are artifact names, not model identity. Do not include credentials. Dependency pins/install receipts are described in setup/VERSIONS.md. Reuse a successful venv.

## Server layout and memory evidence

The sole day-1 reference config is **Qwen3.5-0.8B / vllm-metal BF16 / 2x12**. Two servers on localhost ports 8000/8001, exactly **one lm-eval process per server**; each client has num_concurrent=12. Additional clients multiply concurrency. No MLX-LM substitution, quantized run or alternative layout is supported in this series.

Perform a **layout sweep on every machine** before accepting 2x12. Reuse your known-good vLLM launch command; **ALWAYS explicitly set `--kv-cache-memory-bytes` on BOTH servers**. The consultancy measured paging with the default pool, and throughput decay as generated sequences lengthen. Those qualitative observations are not a throughput measurement on your Mac. Choose explicit KV bytes from this machine's measured memory headroom, not an invented universal value. Keep prompt+32768 output context and sampler fixed while testing layouts; exploratory layouts are separate qualification observations, not mixed scored runs. Preserve exact launch commands, PIDs, model path/digests, context, versions, KV bytes, memory and output-throughput time series in `receipts/layout-sweep/`. If the installed command refuses the flag, stop scored traffic and record the incompatibility; do not silently omit it.

Start each server with its existing installed entry point, its port, BF16 configuration and explicit `--kv-cache-memory-bytes <measured bytes>`. No universal startup command is asserted: the supplied sources do not pin this Mac's actual vLLM-metal entry point or appropriate KV pool size. Save its `--help` and exact successful command. Check sufficient context and `message.content` versus `reasoning_content`, correct tokenizer/template/thinking and no hidden truncation.

Watch memory_pressure, vm_stat swap-ins/outs, and swap use **during sustained long-output load**, not just startup:

```bash
memory_pressure > receipts/memory-pressure-before.txt
vm_stat > receipts/vm-stat-before.txt
sysctl vm.swapusage > receipts/swap-before.txt
# While the two clients run, repeat these snapshots at dated intervals.
# Record generated tokens/s in early and late windows from existing server records.
memory_pressure > receipts/memory-pressure-after.txt
vm_stat > receipts/vm-stat-after.txt
sysctl vm.swapusage > receipts/swap-after.txt
```

If memory pressure/paging makes 2x12 unfit, preserve evidence and mark the reference config unqualified. A lower-concurrency layout is a separately named experiment; do not silently report it as 2x12. Expected throughput decay is a reason to measure the tail, not to multiply a short-request throughput by concurrency.

## Required step order

Times below are **estimates/operator observation budgets**, inherited or derived from source caps; **no Mac step has a measured duration in this kit**. A decode-only scenario at aggregate 182 generated tokens/s is explicitly hypothetical for this workload. Prefill, contention, retries, setup and long-output decay are extra; do not count concurrency twice.

**(0) Settings on the wire.** Estimate 5–15 min inspection, plus 45–90 min optional long-cap observation; not a completion guarantee. Before model spending, inspect the resolved server config and sanitized request/response records. Verify all backend.json fields actually arrive and take effect: max_tokens=32768, temperature=1, top_p=.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1, thinking=true, seed and actual suite stop. Never capture auth headers. Observe context acceptance and a naturally cap-reaching reply if available. Client configs and Linux stubs prove only transport, not real server semantics. Task/generation `max_gen_toks` through `--gen_kwargs` maps to HTTP `max_tokens`; this output budget is separate from context length. Missing finish_reason/usage in lm-eval is expected; a configured cap does not classify cap hits. RikiyaT cut_share requires the actual finish_reason (RIKIYAT-REPRODUCTION.md). Recover only uniquely matched existing server records (RESCORE.md). Check gate G1 before accepting any score.

**(1) Author-reproduction eligibility.** Read RIKIYAT-REPRODUCTION.md and retain its per-metric **not-comparable** decisions in the pilot report. The source uses think35 temperature .6, its own prompts/extraction and repeated samples; current SUITE cannot verify those values. No faithful lane is enabled with the existing backend/scorers, so this step requires no generations, card smoke or manual worksheet. Continue to (2). Do not use an exploratory hand-run as the reference or target.

**(2) SUITE smoke, both benchmarks.** Estimate 5–15 min inspection plus generation; three items/task at cap have a ~9 min decode-only scenario per task at assumed 182 tok/s. Run these sequentially on server A (server B stays free):

```bash
for task in minerva_math500 mmlu_pro; do
  bash shard-run.sh "$task" suite-smoke http://127.0.0.1:8000/v1/chat/completions smoke "plans/$task-smoke.json" > "receipts/$task-smoke.log" 2>&1
  printf '%s exit=%s\n' "$task" "$?" >> receipts/smoke-exits.txt
done
```

Do not skip the second benchmark because the first failed. Inspect each native sample, prompt hashes, inherited few shots/stops and suite metrics. Smoke is transport/scoring inspection, never a screen/full score.

**(3) One base-vs-base A/A shard pair per benchmark.** Estimate 10–20 min inspection plus generation; six capped responses per benchmark give ~18 min decode-only at assumed 182 tok/s. Keep the SAME actual model/lineage for labels `aa-a` and `aa-b`; change only request draw seed 1234→1235. For each task run the two commands on ports 8000/8001 concurrently, wait both before the next task:

```bash
for task in minerva_math500 mmlu_pro; do
  DRAW_SEED=1234 bash shard-run.sh "$task" aa-a http://127.0.0.1:8000/v1/chat/completions control "plans/$task-aa.json" > "receipts/$task-aa-a.log" 2>&1 &
  pid_a=$!
  DRAW_SEED=1235 bash shard-run.sh "$task" aa-b http://127.0.0.1:8001/v1/chat/completions control "plans/$task-aa.json" > "receipts/$task-aa-b.log" 2>&1 &
  pid_b=$!
  rc_a=0; wait "$pid_a" || rc_a=$?
  rc_b=0; wait "$pid_b" || rc_b=$?
  printf '%s A=%s B=%s\n' "$task" "$rc_a" "$rc_b" >> receipts/aa-exits.txt
  bash compare-bundles.sh "plans/$task-aa.json" "bundles/$task/aa-a" "bundles/$task/aa-b" --aa > "receipts/$task-aa-compare.json"
done
```

Report observed discordance and cap rates. A tiny A/A pair characterizes these units only; it is not a precise noise-floor estimate. Zero discordance is a possible result, not a proof of determinism.

**(4) Kill one shard.** Estimate 10–20 min plus generation/drain. Start a fresh smoke shard in background and record `$!` in receipts; after observing active requests, send `kill -TERM "$pid"` to that wrapper PID only, wait it, record nonzero exit and failed/incomplete state. Preserve a previously completed bundle's hashes. Check residual server requests drain using server records. No pkill/killall. An abrupt SIGKILL/power loss may leave unsealed state=running; never accept it. Rerun the exact selection with a **new attempt shard ID**, and compare recovered output to the requested item list. There is no intra-shard checkpoint: unfinished shard work may be lost/repeated. Keep failed attempts outside accepted comparison roots; preserve their original evidence separately.

**(5) Screening: 197/R1, two servers, small shards.** Exact plan: **plans/screening-197-R1.tsv**. Per benchmark: 17 shards, 16×12 plus 5, alternating A/B; 101 items A, 96 B. Offsets refer to frozen tier ordering. One arm across both benchmarks is 394 responses; a base+patch comparison is 788 responses. All-cap decode-only scenarios at assumed aggregate 182 tok/s: ~19.7 h per arm, ~39.4 h paired; actual shorter responses may cost less, tail/prefill may cost more. These are scenarios, not measured wall times. Start only after applicable suite gates; retain the documented author-protocol not-comparable decisions.

For each arm set `LABEL=base` (pilot patch=none), then later a separate qualified `LABEL=patch` with exact patch lineage. Do not call base-only coverage a paired screen. Run this lane loop twice, once for A and once for B, in parallel shells; each loop runs its shards serially:

```bash
# Shell A: SERVER=A PORT=8000; shell B: SERVER=B PORT=8001
SERVER=A; PORT=8000; LABEL=base
status=0
while IFS="$(printf '\t')" read -r task server shard tier start count; do
  [ "$server" = "$SERVER" ] || continue
  bash shard-run.sh "$task" "$LABEL" "http://127.0.0.1:$PORT/v1/chat/completions" "$shard" --tier "$tier" "$start" "$count" > "receipts/$task-$LABEL-$shard.log" 2>&1
  rc=$?; printf '%s %s %s\n' "$task" "$shard" "$rc" >> "receipts/lane-$SERVER-$LABEL-exits.txt"
  [ "$rc" -eq 0 ] || status=1
done < plans/screening-197-R1.tsv
printf 'lane %s overall=%s\n' "$SERVER" "$status"
# In a script, exit "$status". Run/report BOTH lanes even if one is red.
```

Use a shell without `set -e`, or explicitly capture each exit as above. Rerun skips only a matching complete sealed request; incomplete/corrupt/existing directories are refused without overwrite. A failed plan shard needs a new attempt ID and a separately recorded deterministic accepted-attempt mapping. Comparison roots contain only accepted attempts, never duplicates. `compare-bundles.sh tiers/TASK/n197.json bundles/TASK/base bundles/TASK/patch` requires full paired-tier coverage and matching protocols/configs; it emits descriptive counts, paired outcomes and weighted tier exact_match, not significance/CIs. Read RESCORE.md for official recomputation and manual CI.

**(6) Send back evidence.** Estimate 5–15 min packaging/inspection plus data transfer. Send sealed `bundles/` for every completed accepted shard, separate failed-attempt evidence, receipts/run logs, Mac dependency freeze and pip-check, server/Metal/Python versions, exact launch/config/context/KV settings, layout sweep results, memory-pressure/swap snapshots, wire-settings and sanitized finish/token evidence, A/A comparisons, kill test/drain/recovery results and mac-gates.md with each gate verdict/evidence locator. Send sanitized failure summaries only; never send rejected/unsealed raw attempts. Never send caches, venvs, credentials or auth/session records. No automatic network send is part of this kit. Scan exports and compute new manifests for any sanitized copies. Original sealed artifacts remain unchanged in restricted local custody.

## CLI, layout and tools

Bash contract:

```text
bash shard-run.sh TASK LABEL ENDPOINT SHARD INDEX_JSON
bash shard-run.sh TASK LABEL ENDPOINT SHARD --tier n197 START COUNT
bash compare-bundles.sh INDEX_JSON A_LABEL_ROOT B_LABEL_ROOT [--aa]
```

TASK: minerva_math500, mmlu_pro, 14 pinned mmlu_pro_CATEGORY children. `card-math` remains a retired internal diagnostic fixture, not an owner pilot step. An explicit selection JSON supplies task/dataset/revision/indices. For parent MMLU these are global IDs; child files use category-local IDs. `--tier` accepts available frozen parent n-lists only. No implicit raw-prefix range, arbitrary task or IFEval substitution. Empty/out-of-range/duplicate selections are refused. DRAW_SEED defaults 1234, PROFILE_FILE and MODEL_ID are required; PILOT_PYTHON defaults .venv/bin/python. Exits: 0 complete/skip; 64 CLI misuse; nonzero validation/inference/interruption failures. All outcomes remain deployment-unqualified.

PowerShell 7 candidate has the same positional contract (`pwsh -File shard-run.ps1 ...`), JSON libraries, schema and bundle paths; it uses Get-FileHash and forward-slash seal/source paths. It is **UNVERIFIED, syntax not executed**; do not treat it as day-1 Windows support. Its interruption/finalization semantics need target execution before use.

Bundles: `bundles/<task>/<label>/<shard>/` contains manifest.json, units.json, samples.json, model-args.json, generation.json, source-hashes.json, lm-eval.pid, run.log, native upstream results/samples, native-rows.jsonl, relative native-file inventory, items.jsonl and SHA256SUMS. Null finish/token/cap fields mean manual evidence is owed. Trust starts submitted; a valid seal is integrity evidence only. No code mutates upstream task/scorer or overwrites a bundle.

| Tool | Purpose / exact entry point | Outputs |
|---|---|---|
| Python 3.12 / venv / pip | `PYTHON_BIN=python3.12 bash setup/install.sh`; upstream installs only | .venv/, .cache/, .tmp/, receipts/ |
| lm_eval 0.4.13 | `bash shard-run.sh ...` invokes `python -m lm_eval run` | bundles/.../native, logs and sealed metadata |
| vLLM 0.33.0 + vllm-metal | Existing server entry point; explicit `--kv-cache-memory-bytes`; resolved command recorded per machine | Server receipts / layout sweep |
| Bash + jq | shard-run.sh, compare-bundles.sh, lib/*.jq; selection and JSON/integrity plumbing | bundles/, .tmp/, comparison stdout |
| shasum / sha256sum | `shasum -a 256 -c MANIFEST.sha256`; runner uses available SHA256 command | Manifest verification / SHA256SUMS |
| curl | Projected localhost `/v1/models` check above | receipts/models-A.json, models-B.json |
| memory_pressure / vm_stat / sysctl | Snapshot commands above | receipts/ memory/swap evidence |
| PowerShell 7 (optional, UNVERIFIED) | `pwsh -File shard-run.ps1 ...`; Get-FileHash | Same bundle layout |
| tar | `tar -czf pilot-results.tar.gz bundles receipts` after manual export scan | Transfer archive; exclude failed/unsealed exports |

The rebuilt clean kit archive and detached SHA256 receipt are one required transfer unit. See VERIFY.md for the exact Linux scope and the self-reference limitation of embedding an archive's own hash.
