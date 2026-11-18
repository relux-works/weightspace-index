"""One server, frozen items, identical settings, two sets of weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).parent
TASKS = {"math500": "minerva_math500", "mmlu-pro": "bench_mmlu_pro"}
SECRET_KEY = re.compile(
    r"(^|_)(api_key|secret|password|authorization|credential|private_key|access_token|token|cookie)(_|$)",
    re.I,
)
TOKEN = re.compile(r"(?:hf_|ghp_|github_pat_|sk-|glpat-|xox[baprs]-)[A-Za-z0-9_\-]{20,}")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_export(value: Any) -> None:
    """Reject credential-like fields and values before writing artifacts."""
    if isinstance(value, dict):
        for key, item in value.items():
            require(not (SECRET_KEY.search(key) and item), "secret-named field refused")
            safe_export(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            safe_export(item)
    elif isinstance(value, str):
        require(not TOKEN.search(value), "token-shaped value refused")


def write_json(path: Path, value: Any) -> None:
    safe_export(value)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
    temporary.replace(path)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_url(url: str) -> str:
    parsed = urlsplit(url)
    require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
            and not parsed.username and not parsed.password and not parsed.query
            and not parsed.fragment and parsed.path == "/v1/chat/completions"
            and parsed.port is not None, "only a 127.0.0.1 chat endpoint is allowed")
    return url


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_text()
    cfg = json.loads(raw)
    template = read_json(HERE / "config.json")
    require(cfg.keys() == template.keys(), "config keys differ from explicit schema")
    require(cfg["generation"].keys() == template["generation"].keys(), "generation keys missing/unknown")
    require(cfg["server"].keys() == template["server"].keys(), "server keys missing/unknown")
    require(cfg["generation"]["max_gen_toks"] == 32768, "max output tokens must be 32768")
    require(type(cfg["repetitions"]) is int and cfg["repetitions"] > 0, "invalid repetitions")
    require(type(cfg["seed"]) is int and cfg["seed"] >= 0, "invalid seed")
    require(type(cfg["num_fewshot"]) is int and cfg["num_fewshot"] == 0, "only explicit zero-shot supported")
    for key in ("request_timeout", "startup_timeout"):
        require(type(cfg[key]) is int and cfg[key] > 0, "invalid timeout")
    g = cfg["generation"]
    for key in ("temperature", "top_p", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"):
        require(type(g[key]) in (int, float) and math.isfinite(g[key]), "nonfinite sampling value")
    require(g["temperature"] > 0 and g["do_sample"] is True, "sampling must be enabled")
    require(0 < g["top_p"] <= 1 and 0 <= g["min_p"] <= 1, "invalid sampling probability")
    require(type(g["top_k"]) is int and g["top_k"] > 0 and g["repetition_penalty"] > 0, "invalid sampling")
    require(-2 <= g["presence_penalty"] <= 2 and -2 <= g["frequency_penalty"] <= 2, "invalid penalty")
    require(g["until"] == [] and cfg["eos_string"] == "<|im_end|>", "stop policy changed")
    require(cfg["apply_chat_template"] is True and cfg["fewshot_as_multiturn"] is False
            and cfg["system_instruction"] is None, "unsupported prompt policy")
    server = cfg["server"]
    require(server["dtype"] in ("bfloat16", "float16", "float32", "auto"), "invalid server dtype")
    require(type(server["max_model_len"]) is int and server["max_model_len"] > 32768,
            "model context must leave room for prompt plus output")
    require(type(server["max_num_seqs"]) is int and server["max_num_seqs"] == 1,
            "only one active sequence supported")
    require(server["generation_config"] == "vllm" and server["trust_remote_code"] is False,
            "unsupported server policy")
    require(cfg["aggregate"] == "unweighted item mean across repetitions", "unsupported aggregate")
    safe_export(cfg)
    return cfg, raw


def load_ids(path: Path) -> dict:
    frozen = read_json(path)
    ids = frozen["indices"]
    require(type(frozen["n"]) is int and len(ids) == frozen["n"] > 0, "wrong frozen item count")
    require(all(type(i) is int and 0 <= i < frozen["population"] for i in ids), "invalid frozen ID")
    require(len(set(ids)) == len(ids), "duplicate frozen IDs")
    require(re.fullmatch(r"[0-9a-f]{40}", frozen["revision"]), "dataset revision is not pinned")
    return frozen


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "path escapes entry/model directory")
    return path


def read_entry(entry: Path) -> tuple[dict, list[Path], dict]:
    card = read_json(entry.parent / "patch.json")
    version = read_json(entry / "version.json")
    base = read_json(entry.parent.parent / "base.json")
    require(card["kind"] == "replacement", "entry is not a replacement patch")
    profiles = [p for p in base["profiles"] if p["id"] == version["target_profile"]]
    require(len(profiles) == 1, "base profile missing/ambiguous")
    profile = profiles[0]
    require(re.fullmatch(r"[0-9a-f]{40}", profile["source"]["revision"]), "base revision must be pinned")
    patches, receipts = [], []
    for file in version["files"]:
        if file["role"] != "source":
            continue
        locators = [p for p in file["locators"] if p["kind"] == "index-path"]
        require(len(locators) == 1, "exact local replacement locator required")
        path = inside(entry, locators[0]["path"])
        require(path.suffix == ".safetensors", "replacement must be safetensors")
        require(path.stat().st_size == file["size_bytes"] and digest(path) == file["sha256"],
                "replacement checksum mismatch")
        patches.append(path)
        receipts.append({k: file[k] for k in ("name", "sha256", "size_bytes")})
    require(patches, "no replacement tensors")
    identity = {"base": profile["source"], "edits": receipts, "entry": "/".join(entry.parts[-4:])}
    return profile, patches, identity


def inventory(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(root).parts and p.name != ".complete.json"}


def download_base(profile: dict, base: Path, skip: bool) -> None:
    marker = base / ".complete.json"
    if marker.exists():
        require(read_json(marker) == {"source": profile["source"], "files": inventory(base)},
                "base cache is incomplete/changed")
        return
    require(not skip, "no complete pinned base cache; remove --skip-download")
    from huggingface_hub import snapshot_download
    source = profile["source"]
    snapshot_download(repo_id=source["repo_id"], revision=source["revision"], local_dir=base, token=False)
    for file in profile["files"]:
        path = inside(base, file["path"])
        require(path.stat().st_size == file["size_bytes"] and digest(path) == file["sha256"], "base checksum mismatch")
    require((base / "config.json").is_file() and (base / "tokenizer_config.json").is_file(),
            "base configuration incomplete")
    write_json(marker, {"source": source, "files": inventory(base)})


def apply_patch(base: Path, patches: list[Path], edited: Path) -> None:
    """Validate all replacements before publishing a separate copied model."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    require(not edited.exists(), "edited model already exists")
    require(not edited.resolve().is_relative_to(base.resolve())
            and not base.resolve().is_relative_to(edited.resolve()), "base/edited directories overlap")
    tensors, locations = {}, {}
    for patch in patches:
        for name, tensor in load_file(patch).items():
            require(name not in tensors, "duplicate replacement tensor")
            tensors[name] = tensor
    require(tensors, "empty replacement")
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in set(handle.keys()) & tensors.keys():
                original = handle.get_slice(name)
                require(name not in locations, "ambiguous base tensor")
                require(list(tensors[name].shape) == original.get_shape(), "replacement shape mismatch")
                require(str(tensors[name].dtype) == str(handle.get_tensor(name).dtype), "replacement dtype mismatch")
                locations[name] = shard.name
    require(locations.keys() == tensors.keys(), "replacement tensor name missing from base")
    edited.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".edited-", dir=edited.parent))
    try:
        shutil.copytree(base, stage, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache", ".complete.json"))
        for shard in set(locations.values()):
            with safe_open(base / shard, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            content = load_file(base / shard)
            content.update({name: tensors[name] for name in tensors if locations[name] == shard})
            save_file(content, stage / shard, metadata=metadata)
        stage.rename(edited)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def server_argv(model: Path, port: int, kv: int | None, cfg: dict) -> list[str]:
    try:
        module = importlib.util.find_spec("vllm.entrypoints.openai.api_server")
    except ModuleNotFoundError:
        module = None
    if module:
        argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model)]
    else:
        executable = shutil.which("vllm")
        require(executable, "install vLLM/vLLM-metal with OpenAI server support first")
        argv = [executable, "serve", str(model)]
    argv += ["--host", "127.0.0.1", "--port", str(port), "--served-model-name", "bench-model"]
    for key in ("dtype", "max_model_len", "max_num_seqs", "generation_config"):
        argv += ["--" + key.replace("_", "-"), str(cfg["server"][key])]
    if kv is not None:
        argv += ["--kv-cache-memory-bytes", str(kv)]
    return argv


@contextmanager
def serve(argv: list[str], port: int, timeout: int, log: Path) -> Iterator[int]:
    """Own only the launched PID/group, including startup and failure cleanup."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    with log.open("w") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=stream, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            opener = build_opener(ProxyHandler({}))
            while time.monotonic() < deadline:
                require(process.poll() is None, "server exited before health check; see server log")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                        models = json.load(response)["data"]
                    if any(m["id"] == "bench-model" for m in models):
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("server health timeout")
            yield process.pid
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def make_task(benchmark: str, frozen: dict, cfg: dict, documents: list[dict] | None = None) -> Any:
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import __file__ as task_file
    from lm_eval.tasks._yaml_loader import load_yaml
    root = Path(task_file).parent
    name = "minerva_math/minerva_math500.yaml" if benchmark == "math500" else "mmlu_pro/_default_template_yaml"
    task_cfg = load_yaml(root / name)
    task_cfg.update(task=TASKS[benchmark], dataset_path=frozen["dataset"],
                    dataset_kwargs={"revision": frozen["revision"]}, num_fewshot=cfg["num_fewshot"], repeats=1,
                    generation_kwargs=copy.deepcopy(cfg["generation"]))
    if benchmark == "mmlu-pro":
        task_cfg.update(fewshot_split=None, fewshot_config=None)
    if documents is not None:
        from datasets import Dataset
        task_cfg["custom_dataset"] = lambda **kwargs: {"test": Dataset.from_list(documents)}
    # Repetitions are separate complete evaluations; take_first must never discard repeats.
    return ConfigurableTask(config=task_cfg)


def evaluate(benchmark: str, frozen: dict, cfg: dict, url: str, task: Any = None) -> list[dict]:
    """Use the stock backend; preserve all generations and require every item."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager
    validate_url(url)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1"
    task = task if task is not None else make_task(benchmark, frozen, cfg)
    require(len(task.eval_docs) == frozen["population"], "dataset population changed")
    # Harness visits selected rows in dataset order; sorted IDs keep doc_id labels aligned.
    items = []
    for repetition in range(cfg["repetitions"]):
        args = dict(model="bench-model", base_url=url, tokenizer_backend=None, tokenized_requests=False,
                    num_concurrent=1, max_retries=1, timeout=cfg["request_timeout"],
                    seed=cfg["seed"] + repetition, eos_string=cfg["eos_string"], truncate=False,
                    header={"Content-Type": "application/json"})
        result = simple_evaluate(model="local-chat-completions", model_args=args, tasks=[task],
                                 task_manager=TaskManager(include_defaults=False),
                                 samples={task.task_name: sorted(frozen["indices"])}, batch_size=1,
                                 num_fewshot=cfg["num_fewshot"], log_samples=True, bootstrap_iters=0,
                                 apply_chat_template=cfg["apply_chat_template"],
                                 fewshot_as_multiturn=cfg["fewshot_as_multiturn"],
                                 system_instruction=cfg["system_instruction"],
                                 gen_kwargs=copy.deepcopy(cfg["generation"]),
                                 random_seed=cfg["seed"], numpy_random_seed=cfg["seed"],
                                 torch_random_seed=cfg["seed"], fewshot_random_seed=cfg["seed"])
        samples = result["samples"][task.task_name]
        require(len(samples) == frozen["n"] and {s["doc_id"] for s in samples} == set(frozen["indices"]),
                "incomplete evaluation: item IDs/count differ")
        for sample in samples:
            require(len(sample["resps"]) == 1 and len(sample["resps"][0]) == 1
                    and isinstance(sample["resps"][0][0], str) and sample["resps"][0][0].strip(), "missing generation")
            items.append({"id": sample["doc_id"], "repetition": repetition, "sample": sample,
                          "generation": sample["resps"][0][0], "score": float(sample["exact_match"]),
                          "finish_reason": "unknown"})
    return items


def aggregate(record: dict) -> float:
    frozen, items = record["frozen"], record["items"]
    ids = frozen["indices"]
    require(len(ids) == frozen["n"] and len(set(ids)) == len(ids), "invalid frozen count")
    expected = {(i, r) for i in ids for r in range(record["settings"]["repetitions"])}
    require(len(items) == len(expected) > 0 and {(i["id"], i["repetition"]) for i in items} == expected,
            "incomplete run: no final score")
    require(all(i["score"] in (0.0, 1.0) and isinstance(i["generation"], str) and i["generation"].strip() for i in items),
            "invalid score/generation")
    return sum(i["score"] for i in items) / len(items)


def compare(base: dict, edited: dict) -> dict:
    for key in ("benchmark", "frozen", "settings", "settings_verbatim", "identity", "lm_eval_version"):
        require(base[key] == edited[key], "comparison contracts differ")
    require(base["variant"] == "base" and edited["variant"] == "edited", "wrong comparison variants")
    a, b = aggregate(base), aggregate(edited)
    return {"benchmark": base["benchmark"], "base": a, "edited": b, "delta": b - a,
            "n": base["frozen"]["n"], "repetitions": base["settings"]["repetitions"],
            "frozen": base["frozen"], "settings": base["settings"], "identity": base["identity"], "created_at": utc()}


def rescore(record: dict) -> dict:
    """Replay the installed task's filters and scorer using saved documents only."""
    from lm_eval.api.instance import Instance
    require(record["lm_eval_version"] == importlib.metadata.version("lm_eval"),
            "rescore requires the original lm-eval version")
    aggregate(record)
    documents = [item["sample"]["doc"] for item in record["items"]]
    task = make_task(record["benchmark"], record["frozen"], record["settings"], documents)
    instances = []
    for item in record["items"]:
        instance = Instance(request_type="generate_until", doc=item["sample"]["doc"], arguments=(), idx=0,
                            metadata=(task.task_name, item["id"], 1))
        instance.resps = [item["generation"]]
        instances.append(instance)
    task._instances = instances
    task.apply_filters()
    for item, instance in zip(record["items"], instances, strict=True):
        filtered = next(iter(instance.filtered_resps.values()))
        metrics = task.process_results(instance.doc, [filtered])
        item["score"] = float(metrics["exact_match"])
        item["sample"].update(metrics)
        item["sample"]["filtered_resps"] = [filtered]
    record["aggregate"] = aggregate(record)
    record["rescored_at"] = utc()
    return record


def run(args: argparse.Namespace) -> None:
    """Serialize workdir mutations so reruns cannot replace an active model."""
    import fcntl
    args.workdir.mkdir(parents=True, exist_ok=True)
    with (args.workdir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("workdir already has an active run") from error
        run_locked(args)


def run_locked(args: argparse.Namespace) -> None:
    cfg, raw = load_config(args.config)
    frozen = load_ids(HERE / "data" / f"{args.benchmark}-ids.json")
    profile, patches, identity = read_entry(args.entry.resolve())
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    require(1 <= args.port <= 65535, "invalid port")
    require(args.kv_cache_memory_bytes is None or args.kv_cache_memory_bytes > 0, "invalid KV cache size")
    url = validate_url(args.base_url or f"http://127.0.0.1:{args.port}/v1/chat/completions")
    require(urlsplit(url).port == args.port, "endpoint port differs from server port")
    results = work / "results"
    results.mkdir(exist_ok=True)
    require(not any((results / f"{args.benchmark}-{name}.json").exists() for name in ("base", "edited", "compare")),
            "result exists; choose a fresh --workdir or archive previous results")
    base, edited = work / "base", work / "edited"
    download_base(profile, base, args.skip_download)
    seal = {"base": read_json(base / ".complete.json"), "edits": identity["edits"]}
    if args.skip_apply:
        require(read_json(edited / ".complete.json") == {"inputs": seal, "files": inventory(edited)},
                "edited cache incomplete/changed")
    else:
        apply_patch(base, patches, edited)
        write_json(edited / ".complete.json", {"inputs": seal, "files": inventory(edited)})
    records = []
    for variant, model_dir in (("base", base), ("edited", edited)):
        argv = server_argv(model_dir, args.port, args.kv_cache_memory_bytes, cfg)
        started = utc()
        with serve(argv, args.port, cfg["startup_timeout"], work / f"{variant}-server.log") as pid:
            items = evaluate(args.benchmark, frozen, cfg, url)
        record = dict(benchmark=args.benchmark, variant=variant, identity=identity, frozen=frozen,
                      settings=cfg, settings_verbatim=raw, lm_eval_version=importlib.metadata.version("lm_eval"),
                      started_at=started, finished_at=utc(), items=items,
                      server={"argv": argv, "pid": pid, "kv_cache_memory_bytes": args.kv_cache_memory_bytes},
                      finish_reason_note="Stock lm-eval local-chat-completions drops finish_reason; unknown.")
        record["aggregate"] = aggregate(record)
        write_json(results / f"{args.benchmark}-{variant}.json", record)
        records.append(record)
    comparison = compare(*records)
    write_json(results / f"{args.benchmark}-compare.json", comparison)
    print("base       edited     delta      n")
    print(f"{comparison['base']:.6f}   {comparison['edited']:.6f}   {comparison['delta']:+.6f}   {comparison['n']}")


def interrupted(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("--entry", type=Path, required=True)
    command.add_argument("--benchmark", choices=TASKS, required=True)
    command.add_argument("--config", type=Path, default=HERE / "config.json")
    command.add_argument("--workdir", type=Path, default=Path(".temp/bench"))
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--base-url")
    command.add_argument("--kv-cache-memory-bytes", type=int)
    command.add_argument("--skip-download", action="store_true")
    command.add_argument("--skip-apply", action="store_true")
    command = commands.add_parser("rescore")
    command.add_argument("result", type=Path)
    command.add_argument("--output", type=Path, required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--rescore":
        argv[0] = "rescore"
    args = parser.parse_args(argv)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "rescore":
            require(args.output.resolve() != args.result.resolve(), "rescore output must be separate")
            write_json(args.output, rescore(read_json(args.result)))
        else:
            run(args)
    except (Exception, KeyboardInterrupt) as error:
        # Avoid echoing endpoints, credentials, or potentially sensitive library errors.
        message = str(error) if isinstance(error, (ValueError, TimeoutError)) else type(error).__name__
        message = TOKEN.sub("[redacted]", message)
        print(f"benchmark failed: {message}; no final comparison written", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0
