# Versions and install boundary

Target: Apple Silicon macOS, existing Python 3.12, Bash 3.2+, jq 1.6+, curl, macOS shasum and tar. vLLM 0.33.0 + vllm-metal already installed by the pilot operator; installer does not touch them. PowerShell candidate requires PowerShell 7 + jq, and is UNVERIFIED.

`requirements.txt` is copied verbatim from kit2: lm_eval[api,math]==0.4.13 and exercised scorer/data pins. It is not a full cross-platform lock. Record the actual Mac Python version, `pip freeze`, `pip check`, evaluator help and task listing under receipts before runs. Mac wheel availability/install remains UNVERIFIED. No new interpreter, Homebrew package, inference package or global installation is included. If jq is missing, the prerequisite is unresolved; do not silently install a package manager.

`PYTHON_BIN=python3.12 bash setup/install.sh` creates .venv, .tmp, .cache and receipts within the kit, refuses an existing .venv, and redirects install output there. Reuse the successful venv. No custom Python ships in the kit; invocations of `python -m pip` / `python -m lm_eval` run installed upstream packages. JSON selection, bundle manifests, exports and comparison are Bash/jq or PowerShell.

One supported day-1 model config: Qwen3.5-0.8B, vLLM-metal BF16, two servers with num_concurrent=12 each (2x12), thinking enabled, 32768 output cap. Different quantization, conversion, backend, context/sampling, model or server layout is another series; unsupported day 1. A layout sweep is still required on each machine before accepting 2x12.
