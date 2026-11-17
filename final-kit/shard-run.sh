#!/usr/bin/env bash
# kit2-delta lineage: upstream CLI, pinned sampling, explicit selection, sealed native logs.
set -euo pipefail
umask 077
kit_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
source "$kit_dir/lib/common.sh"
usage() { echo 'usage: shard-run.sh TASK LABEL ENDPOINT SHARD INDEX_JSON | TASK LABEL ENDPOINT SHARD --tier n197 START COUNT' >&2; exit 64; }
[ "$#" -eq 5 ] || [ "$#" -eq 8 ] || usage
task=$1; label=$2; endpoint=$3; shard=$4
case "$task" in minerva_math500|mmlu_pro|card-math) ;; mmlu_pro_*) jq -e --arg t "$task" 'has($t)' "$kit_dir/task-pins.json" >/dev/null || refuse 'unknown child';; *) refuse 'unsupported task';; esac
[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$shard" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || refuse 'unsafe label/shard'
[[ "$endpoint" =~ ^http://(127\.0\.0\.1|localhost|\[::1\]):[0-9]+/v1/chat/completions$ ]] || refuse 'local full endpoint URL required'
: "${MODEL_ID:?set MODEL_ID}" "${PROFILE_FILE:?set PROFILE_FILE to completed lineage JSON}"
[[ "$MODEL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || refuse 'invalid served model identifier'
py=${PILOT_PYTHON:-"$kit_dir/.venv/bin/python"}
draw_seed=${DRAW_SEED:-1234}
[[ "$draw_seed" =~ ^[0-9]+$ ]] && [ "$draw_seed" -le 4294967295 ] || refuse 'invalid seed'
"$py" -m pip show lm_eval 2>/dev/null | grep -x 'Version: 0.4.13' >/dev/null || refuse 'lm_eval==0.4.13 required'
scan_file "$PROFILE_FILE" || refuse 'lineage export scan'
jq -e 'all(.[]; .!=null and .!="" and (tostring|contains("REQUIRED")|not)) and
  .reference_config=="qwen35-0.8b-vllm-metal-bf16-2x12-v1" and .quantization=="bf16" and
  ([.base_weights_digest,.tokenizer_digest,.template_digest]|all(.[];test("^[a-f0-9]{64}$")))' "$PROFILE_FILE" >/dev/null || refuse 'incomplete/nonreference lineage'
# Required string fields cannot be absent. Unknowns must not impersonate evidence.
jq -e 'has("patch_digest") and has("conversion_tool_versions") and has("patch_quantization_order") and has("engine_version_observed") and has("plugin_version_observed") and has("machine_layout_evidence")' "$PROFILE_FILE" >/dev/null || refuse 'missing lineage fields'
if [ "$#" -eq 8 ]; then
  [ "$5" = --tier ] || usage
  [[ "$6" =~ ^n[0-9]+$ && "$7" =~ ^[0-9]+$ && "$8" =~ ^[0-9]+$ ]] || refuse 'invalid tier/range'
  [ "$task" != card-math ] || refuse 'card uses explicit smoke indices'
  spec="$kit_dir/tiers/$task/$6.json"; start=$7; count=$8
  (cd "$kit_dir/tiers" && while IFS= read -r line; do
    digest=${line%%  *}; name=${line#*  }; [ "$(hash_file "$name")" = "$digest" ] || exit 65
  done < SHA256SUMS) || refuse 'frozen tiers changed'
else
  spec=$5; start=0; count=$(jq -er '.indices|length' "$spec")
fi
scan_file "$spec" || refuse 'selection export scan'
units=$(jq -n --arg task "$task" --argjson start "$start" --argjson count "$count" --slurpfile spec "$spec" --slurpfile pins "$kit_dir/task-pins.json" --slurpfile strata "$kit_dir/tiers/strata.json" -f "$kit_dir/lib/select.jq")
protocol=lm_eval-0.4.13-qwen35-text-thinking-32k-math500-mmlupro-tiers-v2
[ "$task" != card-math ] || protocol=qwen35-0.8b-card-replication-32k-v1
bundle="$kit_dir/bundles/$task/$label/$shard"
if [ -e "$bundle" ] || [ -L "$bundle" ]; then
  verify_tree "$bundle" || refuse 'existing incomplete/corrupt bundle; use a NEW attempt id'
  jq -e --argjson u "$units" --arg p "$protocol" --arg l "$label" --arg model "$MODEL_ID" --arg url "$endpoint" --argjson seed "$draw_seed" --slurpfile lin "$PROFILE_FILE" --slurpfile backend "$kit_dir/backend.json" '
    .state=="complete-unqualified" and .exit_status==0 and .sample_units==$u and .protocol_id==$p and .model_label==$l and .served_model==$model and .endpoint==$url and .request_seed==$seed and .lineage==$lin[0] and .sampling==$backend[0].generation_kwargs' "$bundle/manifest.json" >/dev/null || refuse 'existing bundle is not this completed request'
  echo 'SKIP: matching complete sealed bundle'; exit 0
fi
mkdir -p "$kit_dir/bundles/$task/$label" "$kit_dir/.cache/hf" "$kit_dir/.tmp"
mkdir "$bundle" || refuse 'bundle reservation failed'
export HF_HOME="$kit_dir/.cache/hf" XDG_CACHE_HOME="$kit_dir/.cache" TMPDIR="$kit_dir/.tmp" MPLCONFIGDIR="$kit_dir/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
printf '%s\n' "$units" > "$bundle/units.json"
jq 'group_by(.task)|map({key:.[0].task,value:map(.index)})|from_entries' "$bundle/units.json" > "$bundle/samples.json"
jq --arg model "$MODEL_ID" --arg url "$endpoint" --argjson seed "$draw_seed" '.model_args|.model=$model|.base_url=$url|.seed=$seed' "$kit_dir/backend.json" > "$bundle/model-args.json"
jq '.generation_kwargs' "$kit_dir/backend.json" > "$bundle/generation.json"
# Stable all-task source fingerprint; endpoint and model label are not source identity.
for p in "$kit_dir"/tasks/*.yaml "$kit_dir"/lib/* "$kit_dir/backend.json" "$kit_dir/task-pins.json" "$kit_dir/shard-run.sh"; do
 printf '%s\t%s\n' "${p#"$kit_dir/"}" "$(hash_file "$p")"
done | jq -Rn '[inputs|split("\t")|{key:.[0],value:.[1]}]|from_entries' > "$bundle/source-hashes.json"
jq -n --arg task "$task" --arg label "$label" --arg url "$endpoint" --arg shard "$shard" --arg model "$MODEL_ID" --arg protocol "$protocol" --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg selection "$(hash_file "$spec")" --argjson seed "$draw_seed" --slurpfile units "$bundle/units.json" --slurpfile lin "$PROFILE_FILE" --slurpfile backend "$kit_dir/backend.json" --slurpfile sources "$bundle/source-hashes.json" --slurpfile pins "$kit_dir/task-pins.json" -f "$kit_dir/lib/manifest.jq"  > "$bundle/manifest.json"
child=''
finish() {
 rc=$?; trap - EXIT INT TERM
 if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
 # Missing native output is a failed bundle, never a successful empty result.
 find "$bundle/native" -type f -name 'samples_*.jsonl' 2>/dev/null | LC_ALL=C sort > "$bundle/native-files.txt" || true
 : > "$bundle/native-rows.jsonl"
 while IFS= read -r f; do
   # Native task key is in the sample filename, not trusted directory naming.
   name=${f##*/samples_}; native_task=$(printf "%s" "${name%.jsonl}" | sed -E 's/_[0-9]{4}-.*$//')
   jq -c --arg t "$native_task" '.+{native_task:$t}' "$f" >> "$bundle/native-rows.jsonl" || rc=65
 done < "$bundle/native-files.txt"
 # Do not ship machine-dependent absolute native filenames.
 sed "s|$bundle/||" "$bundle/native-files.txt" > "$bundle/native-files.relative.txt"
 rm "$bundle/native-files.txt"
 jq -c --slurpfile units "$bundle/units.json" '. as $r | $units[0][] | select(.task==$r.native_task and .index==$r.doc_id) |
   .+{prompt_sha256:$r.prompt_hash,doc_sha256:$r.doc_hash,finish_reason:null,prompt_tokens:null,completion_tokens:null,total_tokens:null}' "$bundle/native-rows.jsonl" > "$bundle/items.jsonl" || rc=65
 if ! jq -e -s --slurpfile units "$bundle/units.json" '
   (length==($units[0]|length)) and (map([.task,.index])|unique|length)==length and
   (all(.[];(.prompt_sha256|test("^[a-f0-9]{64}$")) and (.doc_sha256|test("^[a-f0-9]{64}$"))))' "$bundle/items.jsonl" >/dev/null; then rc=65; fi
 jq --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson rc "$rc" '.utc_end=$time|.exit_status=$rc|.state=(if $rc==0 then "complete-unqualified" else "failed" end)' "$bundle/manifest.json" > "$bundle/manifest.next"
 mv "$bundle/manifest.next" "$bundle/manifest.json"
 if ! seal_tree "$bundle"; then
   jq '.state="export-refused"|.exit_status=65' "$bundle/manifest.json" > "$bundle/manifest.next"; mv "$bundle/manifest.next" "$bundle/manifest.json"
   echo 'REFUSED: export scan; preserve private unsealed attempt' >&2; exit 65
 fi
 verify_tree "$bundle" || exit 65
 echo "bundle=$bundle exit=$rc" >&2; exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ "$task" = card-math ]; then tasks=card_math; else tasks=$(jq -r 'keys|map("day1_"+.)|join(",")' "$bundle/samples.json"); fi
"$py" -m lm_eval run --model local-chat-completions --model_args "$(cat "$bundle/model-args.json")" --gen_kwargs "$(cat "$bundle/generation.json")" --tasks "$tasks" --include_path "$kit_dir/tasks" --apply_chat_template --batch_size 1 --seed "0,1234,1234,$draw_seed" --samples "$(cat "$bundle/samples.json")" --log_samples --output_path "$bundle/native" > "$bundle/run.log" 2>&1 &
child=$!; printf '%s\n' "$child" > "$bundle/lm-eval.pid"
wait "$child"
