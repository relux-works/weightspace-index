#!/usr/bin/env bash
set -euo pipefail
umask 077
kit_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
# shellcheck source=lib/common.sh
source "$kit_dir/lib/common.sh"

usage() {
    echo 'usage: shard-run.sh TASK LABEL ENDPOINT SHARD {INDEX_JSON | --tier NAME START COUNT}' >&2
    exit 64
}

prepare_request() {
    local spec=$1 start=$2 count=$3 path digest selection_hash
    cp -- "$PROFILE_FILE" "$scratch/lineage.json"
    cp -- "$spec" "$scratch/selection.json"
    scan_file "$scratch/lineage.json" || refuse 'lineage export scan'
    scan_file "$scratch/selection.json" || refuse 'selection export scan'
    jq -es -f "$kit_dir/lib/lineage.jq" "$scratch/lineage.json" >/dev/null ||
        refuse 'incomplete/nonreference lineage'
    if [[ -z $count ]]; then
        count=$(jq -e '.indices | length' "$scratch/selection.json")
    fi
    jq -n --arg task "$task" --argjson start "$start" --argjson count "$count" \
        --slurpfile spec "$scratch/selection.json" --slurpfile pins "$kit_dir/task-pins.json" \
        --slurpfile strata "$kit_dir/tiers/strata.json" -f "$kit_dir/lib/select.jq" > "$scratch/units.json"

    for path in "$kit_dir"/tasks/*.yaml "$kit_dir"/lib/* "$kit_dir"/backend.json \
                "$kit_dir"/task-pins.json "$kit_dir"/shard-run.{sh,ps1}; do
        digest=$(hash_file "$path") || return 1
        printf '%s\t%s\n' "${path#"$kit_dir/"}" "$digest"
    done | jq -Rn '[inputs | split("\t") | {key:.[0],value:.[1]}] | from_entries' \
        > "$scratch/source-hashes.json"

    selection_hash=$(hash_file "$scratch/selection.json")
    jq -n --arg task "$task" --arg label "$label" --arg url "$endpoint" --arg shard "$shard" \
        --arg model "$MODEL_ID" --argjson seed "$seed" --arg selection "$selection_hash" \
        --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --slurpfile units "$scratch/units.json" \
        --slurpfile lin "$scratch/lineage.json" --slurpfile sources "$scratch/source-hashes.json" \
        --slurpfile pins "$kit_dir/task-pins.json" --slurpfile backend "$kit_dir/backend.json" \
        -f "$kit_dir/lib/manifest.jq" > "$scratch/manifest.json"
}

collect_native_rows() {
    local path name task_name
    : > "$bundle/native-files.relative.txt" || return 1
    : > "$bundle/native-rows.jsonl" || return 1
    list_files "$bundle/native" > "$scratch/native-files" || return 1
    while IFS= read -r path; do
        name=${path##*/}
        [[ $name == samples_*.jsonl ]] || continue
        task_name=${name#samples_}
        task_name=${task_name%.jsonl}
        task_name=${task_name%_[0-9][0-9][0-9][0-9]-*}
        printf 'native/%s\n' "$path" >> "$bundle/native-files.relative.txt" || return 1
        jq -c --arg task "$task_name" '. + {native_task:$task}' "$bundle/native/$path" \
            >> "$bundle/native-rows.jsonl" || return 1
    done < "$scratch/native-files"
}

finish_attempt() {
    local status=$1
    : > "$bundle/items.jsonl"
    if ! collect_native_rows ||
       ! jq -nc --slurpfile units "$bundle/units.json" -f "$kit_dir/lib/items.jq" \
           "$bundle/native-rows.jsonl" > "$bundle/items.jsonl" ||
       ! jq -es --slurpfile units "$bundle/units.json" \
           'map([.task,.index]) | sort == ($units[0] | map([.task,.index]) | sort)' \
           "$bundle/items.jsonl" >/dev/null; then
        # Preserve the primary failure (including a signal) if output is also incomplete.
        if [[ $status == 0 ]]; then status=65; fi
    fi
    jq --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson rc "$status" \
        '.utc_end=$time | .exit_status=$rc | .state=(if $rc==0 then "complete-unqualified" else "failed" end)' \
        "$bundle/manifest.json" > "$scratch/final.json"
    mv -- "$scratch/final.json" "$bundle/manifest.json"
    if ! seal_tree "$bundle"; then
        jq '.state="export-refused" | .exit_status=65' "$bundle/manifest.json" > "$scratch/final.json"
        mv -- "$scratch/final.json" "$bundle/manifest.json"
        refuse 'export scan or seal failed; preserve private attempt'
    fi
    printf 'bundle=%s exit=%d\n' "$bundle" "$status" >&2
    exit "$status"
}

on_exit() {
    local status=$? tries
    trap - EXIT
    trap '' INT TERM
    if [[ -n $child ]]; then
        kill -TERM "$child" 2>/dev/null || true
        for ((tries=0; tries<50; tries++)); do
            kill -0 "$child" 2>/dev/null || break
            sleep 0.1
        done
        if (( tries == 50 )); then kill -KILL "$child" 2>/dev/null || true; fi
        wait "$child" 2>/dev/null || true
    fi
    if [[ $started == true ]]; then
        # A finalization error must still release the private scratch directory.
        set +e
        (set -e; finish_attempt "$status")
        status=$?
    fi
    rm -rf -- "$scratch" || exit 1
    exit "$status"
}

[[ $# == 5 || $# == 8 ]] || usage
task=$1
label=$2
endpoint=$3
shard=$4
case $task in
    minerva_math500|mmlu_pro|card-math) ;;
    mmlu_pro_*) jq -e --arg task "$task" 'has($task)' "$kit_dir/task-pins.json" >/dev/null || refuse 'unknown child' ;;
    *) refuse 'unsupported task' ;;
esac
id_pattern='^[A-Za-z0-9][A-Za-z0-9_.-]*$'
[[ $label =~ $id_pattern && $shard =~ $id_pattern ]] || refuse 'unsafe label/shard'
url_pattern='^http://(127\.0\.0\.1|localhost|\[::1\]):([0-9]{1,5})/v1/chat/completions$'
[[ $endpoint =~ $url_pattern ]] || refuse 'local full endpoint URL required'
port=$((10#${BASH_REMATCH[2]}))
(( port > 0 && port <= 65535 )) || refuse 'invalid endpoint port'
: "${MODEL_ID:?set MODEL_ID}" "${PROFILE_FILE:?set PROFILE_FILE to completed lineage JSON}"
[[ $MODEL_ID =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || refuse 'invalid served model identifier'
seed=${DRAW_SEED:-1234}
[[ $seed =~ ^[0-9]{1,10}$ ]] && (( 10#$seed <= 4294967295 )) || refuse 'invalid seed'
seed=$((10#$seed))
python=${PILOT_PYTHON:-"$kit_dir/.venv/bin/python"}
if [[ $python == */* && $python != /* ]]; then python="$PWD/$python"; fi
"$python" -m pip show lm_eval 2>/dev/null | grep -Fx 'Version: 0.4.13' >/dev/null ||
    refuse 'lm_eval==0.4.13 required'

ensure_directory "$kit_dir/.tmp"
scratch=$(mktemp -d "$kit_dir/.tmp/shard.XXXXXXXX")
child=''
started=false
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ $# == 8 ]]; then
    [[ $5 == --tier ]] || usage
    [[ $task == minerva_math500 || $task == mmlu_pro ]] || refuse 'tier requires a parent suite task'
    [[ $6 =~ ^n[0-9]+$ && $7 =~ ^[0-9]{1,9}$ && $8 =~ ^[0-9]{1,9}$ ]] || refuse 'invalid tier/range'
    verify_tree "$kit_dir/tiers" tiers || refuse 'frozen tiers changed'
    prepare_request "$kit_dir/tiers/$task/$6.json" "$((10#$7))" "$((10#$8))"
else
    prepare_request "$5" 0 ''
fi

for dir in "$kit_dir/bundles" "$kit_dir/bundles/$task" "$kit_dir/bundles/$task/$label"; do
    ensure_directory "$dir"
done
bundle="$kit_dir/bundles/$task/$label/$shard"
if [[ -e $bundle || -L $bundle ]]; then
    verify_tree "$bundle" || refuse 'existing incomplete/corrupt bundle; use a NEW attempt id'
    jq -es --slurpfile requested "$scratch/manifest.json" -f "$kit_dir/lib/resume.jq" \
        "$bundle/manifest.json" >/dev/null || refuse 'existing bundle is not this completed request'
    echo 'SKIP: matching complete sealed bundle'
    exit 0
fi
ensure_directory "$kit_dir/.cache"
ensure_directory "$kit_dir/.cache/hf"
ensure_directory "$kit_dir/.cache/matplotlib"
export HF_HOME="$kit_dir/.cache/hf" XDG_CACHE_HOME="$kit_dir/.cache" TMPDIR="$kit_dir/.tmp"
export MPLCONFIGDIR="$kit_dir/.cache/matplotlib" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1

mkdir -- "$bundle" || refuse 'bundle reservation failed'
mv -- "$scratch/units.json" "$scratch/source-hashes.json" "$scratch/manifest.json" "$bundle/"
started=true
jq 'group_by(.task) | map({key:.[0].task,value:map(.index)}) | from_entries' \
    "$bundle/units.json" > "$bundle/samples.json"
jq --arg model "$MODEL_ID" --arg url "$endpoint" --argjson seed "$seed" \
    '.model_args | .model=$model | .base_url=$url | .seed=$seed' "$kit_dir/backend.json" > "$bundle/model-args.json"
jq '.generation_kwargs' "$kit_dir/backend.json" > "$bundle/generation.json"
if [[ $task == card-math ]]; then
    tasks=card_math
else
    tasks=$(jq -r 'keys | map("day1_" + .) | join(",")' "$bundle/samples.json")
fi
(
    cd -- "$bundle"
    exec "$python" -m lm_eval run --model local-chat-completions \
        --model_args "$(<model-args.json)" --gen_kwargs "$(<generation.json)" \
        --tasks "$tasks" --include_path "$kit_dir/tasks" --apply_chat_template --batch_size 1 \
        --seed "0,1234,1234,$seed" --samples samples.json --log_samples --output_path "$bundle/native"
) > "$bundle/run.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$bundle/lm-eval.pid"
status=0
wait "$child" || status=$?
child=''
exit "$status"
