#!/usr/bin/env bash
# Adapted from kit2-delta's bundle validation; JSON plumbing uses jq, never Python.
set -euo pipefail
hash_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1; else shasum -a 256 "$1" | cut -d ' ' -f 1; fi; }
refuse() { echo "REFUSED: $*" >&2; exit 65; }
scan_file() {
  # Never print a matched value. These are detector patterns, not credentials.
  if LC_ALL=C grep -Eq '(sk-|hf_|ghp_|ghs_|github_pat_)[A-Za-z0-9_-]{16,}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----' "$1"; then return 1; fi
  case "$1" in *.json|*.jsonl)
    jq -e -s 'all(.. | objects | to_entries[];
       ((.key|ascii_downcase|gsub("-";"_"))|test("(^|_)(api_key|apikey|access_token|auth_token|refresh_token|authorization|password|passwd|secret|private_key|client_secret|token|secret_key|api_token|bearer_token|session_token|hf_token|github_token|credentials|credential|privatekey|clientsecret)$")|not)
       or (.value==null or .value=="" or .value==false or .value==[] or .value=={}))' "$1" >/dev/null 2>&1 || return 1;;
  esac
}
scan_tree() {
  local b=$1 p
  [ -z "$(find "$b" -type l -print -quit)" ] || return 1
  while IFS= read -r p; do scan_file "$p" || return 1; done < <(find "$b" -type f | LC_ALL=C sort)
}
seal_tree() {
  local b=$1 p
  scan_tree "$b" || return 1
  [ ! -e "$b/SHA256SUMS" ] || return 1
  while IFS= read -r p; do printf '%s  %s\n' "$(hash_file "$p")" "${p#"$b/"}"; done < <(find "$b" -type f ! -name SHA256SUMS | LC_ALL=C sort) > "$b/SHA256SUMS"
}
verify_tree() {
  local b=$1 line digest name actual=0 sealed=0
  [ -f "$b/SHA256SUMS" ] && scan_tree "$b" || return 1
  while IFS= read -r line; do
    digest=${line%%  *}; name=${line#*  }
    [[ "$digest" =~ ^[a-f0-9]{64}$ ]] || return 1
    case "$name" in ''|/*|../*|*/../*|*/..|*\\*|SHA256SUMS) return 1;; esac
    [ -f "$b/$name" ] && [ "$(hash_file "$b/$name")" = "$digest" ] || return 1
    sealed=$((sealed+1))
  done < "$b/SHA256SUMS"
  actual=$(find "$b" -type f ! -name SHA256SUMS | wc -l | tr -d ' ')
  [ "$actual" -eq "$sealed" ] && [ "$(cut -c 67- "$b/SHA256SUMS" | sort -u | wc -l | tr -d ' ')" -eq "$sealed" ]
}
