"""Prepare and invoke the pinned lm-eval task."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import core, ifeval
from .core import HARNESS_MODEL, Protocol, Settings, gen_kwargs, pinned_packages, read_json, rel, require


def work_dir(*parts: str) -> Path:
    path = core.REPO.joinpath(".temp", "bench", *parts)
    path.mkdir(parents=True)
    return path


def task_arg(p: Protocol, s: Settings | None, work: Path) -> str:
    """Include the stock task if it needs a revision pin or a card prompt."""
    prompt = s.doc_to_text(p.task) if s else None
    dataset = p.data["dataset"]
    revision = dataset["revision"] if dataset.get("pin_revision") else None
    if p.task == ifeval.TASK:
        require(prompt is None, "IFEval requires the stock prompt")
        require(revision, "IFEval requires a pinned dataset revision")
    if prompt is None and revision is None:
        return p.task
    root = Path(importlib.util.find_spec("lm_eval").submodule_search_locations[0]) / "tasks"
    stock = ([root / "ifeval" / "ifeval.yaml"] if p.task == ifeval.TASK
             else list(root.rglob(f"{p.task}.yaml")))
    require(len(stock) == 1 and stock[0].is_file(), f"could not resolve the stock {p.task} task")
    path = work / f"{p.task}.yaml"
    lines = {"include": str(stock[0])}
    if revision is not None:
        lines["dataset_kwargs"] = {"revision": revision}
    if prompt is not None:
        lines["doc_to_text"] = prompt
    # JSON scalars and objects are valid YAML values.
    path.write_text("".join(f"{key}: {json.dumps(value)}\n" for key, value in lines.items()))
    return str(path)


def harness_packages(p: Protocol) -> dict[str, str]:
    return ifeval.prepare() if p.task == ifeval.TASK else pinned_packages()


def lm_eval_argv(p: Protocol, s: Settings | None, ids: list[int] | None, base_url: str | None,
                 served: str | None, seed: int | None, out: Path, timeout: int = 0,
                 concurrency: int = 1, task: str | None = None) -> list[str]:
    if base_url is None:  # frozen-hash pass: lm-eval's built-in dummy model
        model, model_args, chat = "dummy", {}, []
    else:
        model = HARNESS_MODEL
        model_args = {"model": served, "base_url": base_url, "num_concurrent": concurrency,
                      "max_retries": 1, "tokenized_requests": False, "tokenizer_backend": None,
                      "timeout": timeout, "seed": seed, **s.data["model_args"]}
        chat = ["--apply_chat_template"]
    module = "bench.ifeval" if p.task == ifeval.TASK else "lm_eval"
    selection = [] if ids is None else ["--samples", json.dumps({p.task: ids})]
    return [sys.executable, "-m", module, "run", "--model", model,
            "--model_args", json.dumps(model_args), "--tasks", task or p.task,
            "--num_fewshot", str(p.data["num_fewshot"]), *chat,
            "--gen_kwargs", json.dumps(gen_kwargs(p, s) if s else p.data["gen_kwargs"]),
            *selection, "--log_samples", "--output_path", str(out)]


def run_logged(argv: list[str], log: Path, what: str, prefix: str = "") -> None:
    print(prefix + "+", shlex.join(argv), flush=True)
    env = {**os.environ, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
    with log.open("w") as f:
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        for line in proc.stdout:
            sys.stdout.write(prefix + line)
            f.write(line)
    require(proc.wait() == 0, f"{what} exited with {proc.returncode}; see {rel(log)}")


def lm_eval_outputs(out: Path, task: str) -> tuple[Path, dict]:
    samples = list(out.rglob(f"samples_{task}_*.jsonl"))
    results = list(out.rglob("results_*.json"))
    require(len(samples) == 1 and len(results) == 1, f"{rel(out)}: expected one samples and one results file")
    return samples[0], read_json(results[0])


def check_results(p: Protocol, results: dict, model: str, prompt: str | None = None) -> dict:
    require(list(results["configs"]) == [p.task], f"lm_eval ran {list(results['configs'])}")
    config = results["configs"][p.task]
    if prompt is not None:
        require(config["doc_to_text"] == prompt, "lm_eval did not use the settings' prompt")
    require(results["lm_eval_version"] == p.data["harness"]["lm_eval"],
            f"lm_eval {results['lm_eval_version']} ran; the protocol pins {p.data['harness']['lm_eval']}")
    require(results["config"]["model"] == model, f"lm_eval ran model {results['config']['model']}")
    require(results["n-shot"][p.task] == p.data["num_fewshot"], "lm_eval used another shot count")
    require(config["metadata"]["version"] == p.data["task_version"],
            f"{p.task} is version {config['metadata']['version']}, the protocol pins {p.data['task_version']}")
    dataset = p.data["dataset"]
    if dataset.get("pin_revision"):
        require(config["dataset_path"] == dataset["repo_id"]
                and config["test_split"] == dataset["split"]
                and config["dataset_kwargs"]["revision"] == dataset["revision"],
                "lm_eval used a dataset, split or revision outside the protocol")
    if p.task == ifeval.TASK:
        require(config["doc_to_text"] == "prompt", "lm_eval did not use the stock IFEval prompt")
    return config
