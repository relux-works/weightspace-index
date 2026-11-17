#!/usr/bin/env bash
set -euo pipefail
umask 077
kit_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
# shellcheck source=lib/common.sh
source "$kit_dir/lib/common.sh"

usage() {
    echo 'usage: compare-bundles.sh INDEX_JSON A_LABEL_ROOT B_LABEL_ROOT [--aa]' >&2
    exit 64
}

collect_bundles() {
    local root=$1 bundle
    [[ -d $root && ! -L $root ]] || refuse 'missing label root'
    local bundles=("$root"/*)
    [[ ${#bundles[@]} -gt 0 ]] || refuse 'no bundles'
    for bundle in "${bundles[@]}"; do
        verify_tree "$bundle" || refuse 'bundle seal/inventory/privacy'
        jq -es 'length == 1 and (.[0] | .state == "complete-unqualified" and .exit_status == 0)' \
            "$bundle/manifest.json" >/dev/null || refuse 'incomplete bundle'
        # Keep long prompts and completions out of the in-memory comparison.
        jq -nc --slurpfile manifest "$bundle/manifest.json" --slurpfile items "$bundle/items.jsonl" \
            '{manifest:$manifest[0],items:$items,
              native:[inputs | {native_task,doc_id,prompt_hash,doc_hash,exact_match,math_verify}]}' \
            "$bundle/native-rows.jsonl"
    done
}

[[ $# == 3 || $# == 4 ]] || usage
aa=false
if [[ $# == 4 ]]; then
    [[ $4 == --aa ]] || usage
    aa=true
fi
ensure_directory "$kit_dir/.tmp"
scratch=$(mktemp -d "$kit_dir/.tmp/compare.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
spec="$scratch/selection.json"
cp -- "$1" "$spec"
scan_file "$spec" || refuse 'selection scan'
task=$(jq -er '.task' "$spec")
[[ $task != card-math ]] || refuse 'card scoring blocked; use manual evidence review'

count=$(jq -e '.indices | length' "$spec")
jq -n --arg task "$task" --argjson start 0 --argjson count "$count" \
    --slurpfile spec "$spec" --slurpfile pins "$kit_dir/task-pins.json" \
    --slurpfile strata "$kit_dir/tiers/strata.json" -f "$kit_dir/lib/select.jq" > "$scratch/expected.json"

shopt -s dotglob nullglob
collect_bundles "$2" > "$scratch/a.jsonl"
collect_bundles "$3" > "$scratch/b.jsonl"
jq -n --argjson aa "$aa" --slurpfile spec "$spec" --slurpfile expected "$scratch/expected.json" \
    --slurpfile a "$scratch/a.jsonl" --slurpfile b "$scratch/b.jsonl" -f "$kit_dir/lib/compare.jq"
