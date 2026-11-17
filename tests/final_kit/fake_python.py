#!/usr/bin/env python3
"""Stand in for the documented Python entry points. No packages or server needed."""

import json
import os
from pathlib import Path
import shutil
import sys
import time


def main(args):
    mode = os.environ.get("FIXTURE_MODE")
    if args == ["--version"]:
        print(os.environ.get("FIXTURE_VERSION", "Python 3.12.9"))
        return 0
    if args[:2] == ["-m", "venv"]:
        target = Path(args[2]) / "bin"
        target.mkdir(parents=True)
        shutil.copy2(__file__, target / "python")
        return 0
    if args == ["-m", "pip", "show", "lm_eval"]:
        print("Version: 0.4.13")
        return 0
    if args[:3] != ["-m", "lm_eval", "run"]:
        return 17 if mode == "install-error" else 0

    def option(name):
        return args[args.index(name) + 1]

    root = Path(option("--output_path"))
    if mode == "error":
        print("startup failed", file=sys.stderr, flush=True)
        return 23
    root.mkdir()
    if mode == "wait":
        (root / "ready").touch()
        for _ in range(450):
            if (root / "release").exists():
                break
            time.sleep(.1)
    argument = option("--samples")
    if mode == "require-sample-file" or not argument.startswith("{"):
        samples = json.loads(Path(argument).read_bytes())
    else:
        samples = json.loads(argument)
    for task, indices in samples.items():
        group = "day1_" + task if task != "card_math" else task
        assert group in option("--tasks").split(",")
        path = root / f"samples_{task}_2026-11-17.jsonl"
        with path.open("w") as log:
            for index in indices + ([-1] if mode == "extra" else []):
                row = dict(doc_id=index, prompt_hash="a" * 64, doc_hash="b" * 64,
                           exact_match=1, math_verify=0)
                if mode == "private":
                    row["credentials"] = "fixture field"
                print(json.dumps(row), file=log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
