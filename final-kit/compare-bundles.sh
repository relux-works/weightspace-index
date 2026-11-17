#!/usr/bin/env bash
# Paired completeness/identity/scoring validation from kit2-delta; no new scorer.
set -euo pipefail
kit_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
source "$kit_dir/lib/common.sh"
[ "$#" -eq 3 ] || [ "$#" -eq 4 ] || { echo 'usage: compare-bundles.sh INDEX_JSON A_LABEL_ROOT B_LABEL_ROOT [--aa]' >&2; exit 64; }
aa=false; if [ "$#" -eq 4 ]; then [ "$4" = --aa ] || refuse 'unknown option'; aa=true; fi
spec=$1; scan_file "$spec" || refuse 'selection scan'
task=$(jq -er .task "$spec"); [ "$task" != card-math ] || refuse 'card scoring blocked; use manual evidence review'
mkdir -p "$kit_dir/.tmp"; scratch=$(mktemp -d "$kit_dir/.tmp/compare.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
for side in a b; do
 if [ "$side" = a ]; then root=$2; else root=$3; fi
 [ -d "$root" ] && [ ! -L "$root" ] || refuse 'missing label root'
 : > "$scratch/$side.jsonl"
 n=0
 for b in "$root"/*; do
  [ -d "$b" ] && [ ! -L "$b" ] || refuse 'unexpected root entry'
  verify_tree "$b" || refuse 'bundle seal/inventory/privacy'
  jq -e '.state=="complete-unqualified" and .exit_status==0' "$b/manifest.json" >/dev/null || refuse 'incomplete bundle'
  jq -n --slurpfile m "$b/manifest.json" --slurpfile items "$b/items.jsonl" --slurpfile native "$b/native-rows.jsonl" '{manifest:$m[0],items:$items,native:$native}' >> "$scratch/$side.jsonl"
  n=$((n+1))
 done
 [ "$n" -gt 0 ] || refuse 'no bundles'
done
jq -n --arg task "$task" --argjson aa "$aa" --slurpfile spec "$spec" --slurpfile pins "$kit_dir/task-pins.json" --slurpfile strata "$kit_dir/tiers/strata.json" --argjson start 0 --argjson count "$(jq '.indices|length' "$spec")" -f "$kit_dir/lib/select.jq" > "$scratch/expected.json"
jq -n --argjson aa "$aa" --slurpfile spec "$spec" --slurpfile expected "$scratch/expected.json" --slurpfile a "$scratch/a.jsonl" --slurpfile b "$scratch/b.jsonl" -f "$kit_dir/lib/compare.jq"
