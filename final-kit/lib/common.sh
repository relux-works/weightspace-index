#!/usr/bin/env bash
# Shared by the Bash entry points. Callers set -euo pipefail, kit_dir and umask.
refuse() {
    printf 'REFUSED: %s\n' "$*" >&2
    exit 65
}

# Create one directory beneath an already checked parent. Concurrent mkdir is OK.
ensure_directory() {
    [[ ! -L $1 ]] && mkdir -p -- "$1" && [[ -d $1 && ! -L $1 ]] ||
        refuse "expected a directory: $1"
}

hash_file() {
    local result
    if command -v sha256sum >/dev/null 2>&1; then
        result=$(sha256sum < "$1") || return 1
    else
        result=$(shasum -a 256 < "$1") || return 1
    fi
    printf '%s\n' "${result%% *}"
}

scan_file() {
    local pattern='(sk-|hf_|ghp_|ghs_|github_pat_)[A-Za-z0-9_-]{16,}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----'
    [[ -f $1 && ! -L $1 ]] || return 1
    # grep exits 1 for no match. A read error must also refuse export.
    if LC_ALL=C grep -aEq -- "$pattern" "$1"; then
        return 1
    elif [[ $? != 1 ]]; then
        return 1
    fi
    case $1 in
        *.json|*.jsonl) jq -ne -f "$kit_dir/lib/privacy.jq" "$1" >/dev/null 2>&1 ;;
    esac
}

# Print sorted relative file names. Reject entries the line-based seal cannot encode.
list_files() (
    [[ -d $1 && ! -L $1 ]] && cd -- "$1" || exit 1
    find . ! -type d -print0 |
        while IFS= read -r -d '' path; do
            [[ -f $path && ! -L $path && $path != *[[:cntrl:]]* && $path != *\\* ]] || exit 1
            printf '%s\n' "${path#./}"
        done | LC_ALL=C sort
)

# Derive checksums from the on-disk inventory. Never open paths read from a seal.
checksums() {
    local root=$1 kind=$2 path digest
    list_files "$root" | while IFS= read -r path; do
        if [[ $kind == tiers ]]; then
            [[ $path == *.json ]] || continue
        else
            [[ $path != SHA256SUMS ]] || continue
            scan_file "$root/$path" || return 1
        fi
        digest=$(hash_file "$root/$path") || return 1
        printf '%s  %s\n' "$digest" "$path" || return 1
    done
}

verify_tree() {
    local root=$1 kind=${2:-bundle}
    [[ -f $root/SHA256SUMS && ! -L $root/SHA256SUMS ]] || return 1
    checksums "$root" "$kind" | LC_ALL=C sort > "$scratch/actual" || return 1
    LC_ALL=C sort "$root/SHA256SUMS" > "$scratch/sealed" || return 1
    [[ -s $scratch/actual ]] && cmp -s "$scratch/actual" "$scratch/sealed"
}

seal_tree() {
    # Publish only after the entire scan and checksum pass. ln refuses an existing file.
    checksums "$1" bundle > "$scratch/checksums" || return 1
    [[ ! -e $1/SHA256SUMS && ! -L $1/SHA256SUMS ]] &&
        ln "$scratch/checksums" "$1/SHA256SUMS"
}
