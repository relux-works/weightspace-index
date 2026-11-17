#!/usr/bin/env bash
set -euo pipefail
umask 077
kit_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
py=${PYTHON_BIN:-python3.12}
mkdir -p "$kit_dir/.tmp" "$kit_dir/.cache/pip" "$kit_dir/receipts"
export TMPDIR="$kit_dir/.tmp" PIP_CACHE_DIR="$kit_dir/.cache/pip"
export XDG_CACHE_HOME="$kit_dir/.cache" HF_HOME="$kit_dir/.cache/hf"
export PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
if [ -e "$kit_dir/.venv" ]; then
    echo 'REFUSED: .venv exists. Reuse it or choose a fresh kit copy; nothing deleted.' >&2
    exit 73
fi
"$py" --version > "$kit_dir/receipts/python.txt"
"$py" -m venv "$kit_dir/.venv"
"$kit_dir/.venv/bin/python" -m pip install -r "$kit_dir/setup/requirements.txt" \
    > "$kit_dir/receipts/install.log" 2>&1
"$kit_dir/.venv/bin/python" -m pip check > "$kit_dir/receipts/pip-check.txt"
"$kit_dir/.venv/bin/python" -m pip freeze > "$kit_dir/receipts/resolved-dependencies.txt"
"$kit_dir/.venv/bin/python" -m lm_eval --help > "$kit_dir/receipts/help.txt"
"$kit_dir/.venv/bin/python" -m lm_eval ls tasks > "$kit_dir/receipts/tasks.txt"
printf '%s\n' 'Installed local evaluator. Read GATES.md; Mac deployment and card scoring remain unqualified.'
