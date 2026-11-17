#!/usr/bin/env bash
set -euo pipefail
umask 077
kit_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
# shellcheck source=../lib/common.sh
source "$kit_dir/lib/common.sh"

python=${PYTHON_BIN:-python3.12}
venv="$kit_dir/.venv"
receipts="$kit_dir/receipts"
if [[ -e $venv || -L $venv ]]; then
    echo 'REFUSED: .venv exists. Reuse a successful install or choose a fresh kit copy.' >&2
    exit 73
fi
version=$("$python" --version)
[[ $version == 'Python 3.12.'* ]] || refuse 'Python 3.12 is required'
for dir in "$kit_dir/.tmp" "$kit_dir/.cache" "$kit_dir/.cache/pip" "$receipts"; do
    ensure_directory "$dir"
done
for name in python.txt install.log pip-check.txt resolved-dependencies.txt help.txt tasks.txt; do
    path="$receipts/$name"
    [[ ! -L $path && ( ! -e $path || -f $path ) ]] || refuse "invalid receipt path: $path"
done

# venv accepts an existing directory. Reserve it exclusively before writing receipts.
mkdir -- "$venv" || exit 73
export TMPDIR="$kit_dir/.tmp" PIP_CACHE_DIR="$kit_dir/.cache/pip"
export XDG_CACHE_HOME="$kit_dir/.cache" HF_HOME="$kit_dir/.cache/hf"
export PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
printf '%s\n' "$version" > "$receipts/python.txt"
"$python" -m venv "$venv"
python="$venv/bin/python"
if "$python" -m pip install -r "$kit_dir/setup/requirements.txt" > "$receipts/install.log" 2>&1; then
    "$python" -m pip check > "$receipts/pip-check.txt"
else
    status=$?
    printf 'Install failed. See %s/install.log\n' "$receipts" >&2
    exit "$status"
fi
"$python" -m pip freeze > "$receipts/resolved-dependencies.txt"
"$python" -m lm_eval --help > "$receipts/help.txt"
"$python" -m lm_eval ls tasks > "$receipts/tasks.txt"
echo 'Installed local evaluator. Read GATES.md before running the pilot.'
