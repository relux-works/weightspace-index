"""Manage model servers and record a matched base-and-patched session."""

from __future__ import annotations

import os
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import core, ifeval
from .core import (HARNESS_MODEL, SESSION_NAME, VARIANTS, Protocol, Settings, Version, check_weights,
                   dumps, non_weights, read_json, read_jsonl, rel, require, sha256_file)
from .harness import (check_results, harness_packages, lm_eval_argv, lm_eval_outputs,
                      run_logged, task_arg, work_dir)
from .proxy import Recorder, get_json
from .runs import headline, is_cut
from .sessions import derive_session, derive_variant, task_config_sha256

SERVED_NAME = "bench"


def model_files(model_dir: Path) -> dict[str, str]:
    """SHA-256 of every file in a model folder (dot paths skipped)."""
    out = {}
    for path in sorted(model_dir.rglob("*")):
        parts = path.relative_to(model_dir).parts
        if path.is_file() and not any(part.startswith(".") for part in parts):
            out["/".join(parts)] = sha256_file(path)[0]
    return out


def check_server(port: int, model_dir: Path, s: Settings) -> tuple[str, dict]:
    models = get_json(port, "/v1/models")
    require(isinstance(models, dict) and len(models.get("data", [])) == 1,
            f"127.0.0.1:{port}/v1/models must list exactly one model; is the server up?")
    m = models["data"][0]
    served, root = m["id"], m.get("root") or m["id"]
    require(Path(root).expanduser().resolve() == model_dir,
            f"server on port {port} serves {root}, not {model_dir}")
    require(not served.startswith(("/", "~", ".")),
            "served model name is a local path; start vllm with --served-model-name bench")
    max_len = m.get("max_model_len")
    require(isinstance(max_len, int) and max_len >= s.data["min_max_model_len"],
            f"server max_model_len {max_len} < {s.data['min_max_model_len']} that the settings need")
    version = get_json(port, "/version")
    return served, {"max_model_len": max_len,
                    "version": version.get("version") if isinstance(version, dict) else None}


def ensure_base(v: Version, base_dir: Path) -> None:
    if base_dir.exists():
        return
    src = v.profile["source"]
    partial = base_dir.with_name(base_dir.name + ".partial")
    print(f"downloading {src['repo_id']}@{src['revision']} into {base_dir} ...", flush=True)
    from huggingface_hub import snapshot_download  # installed with lm-eval
    snapshot_download(repo_id=src["repo_id"], revision=src["revision"], local_dir=partial)
    partial.rename(base_dir)


def ensure_patched(v: Version, base_dir: Path, patched_dir: Path, work: Path) -> None:
    if patched_dir.exists():
        return
    partial = patched_dir.with_name(patched_dir.name + ".partial")
    require(not partial.exists(), f"{partial} is left from an interrupted build; remove it")
    partial.parent.mkdir(parents=True, exist_ok=True)
    tmp = core.REPO / ".temp"
    argv = ["cargo", "run", "--release", "--locked", "--manifest-path",
            str(core.REPO / "tool" / "Cargo.toml"), "--", "apply",
            "--patch", str(v.source_file()), "--model", str(base_dir), "--out", str(partial),
            "--temp-dir", str(tmp)]
    run_logged(argv, work / "apply.log", "weightspace apply")
    partial.rename(patched_dir)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_server_args(args: list[str]) -> None:
    """Refuse vllm serve's --api-key in any spelling it accepts (underscores,
    abbreviations). Server flags are filed in public session records; a key
    goes in VLLM_API_KEY instead, which the recorder forwards unrecorded."""
    for arg in args:
        name = arg.split("=", 1)[0].lstrip("-").lower().replace("_", "-")
        require(not (arg.startswith("-") and len(name) >= len("api-k") and "api-key".startswith(name)),
                f"--server-arg {arg.split('=', 1)[0]}: flags are filed publicly; set VLLM_API_KEY instead")


def default_vllm() -> str:
    """The vllm installed in the same environment as this Python (a project
    venv run without activating it), else whichever vllm is on PATH."""
    beside = Path(sys.executable).parent / "vllm"
    return str(beside) if beside.exists() else "vllm"


@contextmanager
def server(a, model_dir: Path, s: Settings, log: Path, external_port: int | None):
    """A vLLM server on 127.0.0.1 for one model, stopped afterwards; or
    an already-running server the caller named."""
    if external_port:
        yield external_port
        return
    port = free_port()
    argv = [a.vllm, "serve", str(model_dir), "--served-model-name", SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", str(s.data["min_max_model_len"]),
            "--generation-config", "vllm", "--max-num-seqs", str(a.concurrency), *a.server_arg]
    print("+", shlex.join(argv), flush=True)
    with log.open("w") as f:
        proc = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + a.startup_timeout
        while get_json(port, "/v1/models") is None:
            require(proc.poll() is None, f"vllm exited with {proc.returncode}; see {rel(log)}")
            require(time.time() < deadline, f"vllm not up after {a.startup_timeout} s; see {rel(log)}")
            time.sleep(2)
        yield port
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def reuse_base(source: Path, stage: Path, common: dict, files: dict, server_args: list[str],
               p: Protocol, s: Settings) -> tuple[dict, str]:
    """Verify a completed base half against the new session before using it."""
    checkpoint = read_json(source / "base.checkpoint.json")
    for key, expected in common.items():
        require(checkpoint["run"].get(key) == expected, f"saved base has different {key}")
    base = checkpoint["base"]
    require(base["model_files"] == files, "saved base has different model files")
    require(base["server"]["managed"] and base["server"]["args"] == server_args,
            "saved base requires the same managed server arguments")
    samples = source / "session/base.samples.jsonl"
    requests = source / "session/base.requests.jsonl"
    derive_variant(p, s, common["items"], common["seed"], read_jsonl(samples), read_jsonl(requests),
                   ifeval_grader=base.get("ifeval_grader"))
    for path in (samples, requests):
        shutil.copyfile(path, stage / path.name)
    return base, checkpoint["task_config_sha256"]


def run(a, v: Version, p: Protocol, s: Settings, seed: int, ids: list[int], how: str) -> None:
    a.concurrency = a.concurrency or len(ids)
    require(a.concurrency >= 1, "--concurrency must be at least 1")
    if a.reuse_base:
        require(a.items is not None and not (a.parallel or a.base_port or a.patched_port),
                "--reuse-base requires --items and both halves must use managed servers")
    print(f"{p.id} / {s.id} on {v.folder}: seed {seed}, {len(ids)} items ({how}): {ids}")
    if a.dry_run:
        return
    packages = harness_packages(p)

    started = time.time()
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started)) + "-" + secrets.token_hex(3)
    work = work_dir("sessions", name)
    stage = work / "session"
    stage.mkdir()
    cache = core.REPO / ".temp" / "bench" / "models"
    dirs = {
        "base": Path(a.base_dir).expanduser().resolve() if a.base_dir
        else cache / v.base_folder / v.profile["id"],
        "patched": Path(a.patched_dir).expanduser().resolve() if a.patched_dir
        else cache / v.folder,
    }
    ensure_base(v, dirs["base"])
    files = {}
    for variant in VARIANTS:
        if variant == "patched":
            ensure_patched(v, dirs["base"], dirs["patched"], work)
        print(f"hashing {dirs[variant]} ...", flush=True)
        files[variant] = model_files(dirs[variant])
        check_weights(v.profile, files[variant], variant)
    require(non_weights(files["base"]) == non_weights(files["patched"]),
            "base and patched model folders differ outside the weights files")

    task = task_arg(p, s, work)
    common = {"protocol_sha256": p.sha256, "settings_sha256": s.sha256, "profile": v.profile["id"],
              "seed": seed, "items": ids, "concurrency": a.concurrency, "parallel": a.parallel,
              "hardware": a.hardware}
    record = dict(common)

    def half(variant: str, port: int, external: int | None) -> str:
        """One model's half of the session, against a server on `port`."""
        t0 = time.time()
        served, info = check_server(port, dirs[variant], s)
        if a.reuse_base:
            require(all(value == record["base"]["server"].get(key) for key, value in info.items()),
                    "patched server version or context length differs from the saved base")
        recorder = Recorder(port, stage / f"{variant}.requests.jsonl")
        try:
            argv = lm_eval_argv(p, s, ids, f"http://127.0.0.1:{recorder.port}/v1/chat/completions",
                                served, seed, work / f"{variant}.out", a.timeout, a.concurrency, task)
            run_logged(argv, work / f"{variant}.lm_eval.log", "lm_eval",
                       f"[{variant}] " if a.parallel else "")
        finally:
            recorder.close()
        samples_path, results = lm_eval_outputs(work / f"{variant}.out", p.task)
        config = task_config_sha256(check_results(p, results, HARNESS_MODEL, s.doc_to_text(p.task)))
        shutil.copyfile(samples_path, stage / f"{variant}.samples.jsonl")
        record[variant] = {"served_model": served, "model_files": files[variant],
                           "server": {**info, "managed": not external,
                                      "args": [] if external else a.server_arg},
                           "seconds": round(time.time() - t0)}
        if variant == "base" and not a.parallel and not external:
            require(model_files(dirs["base"]) == files["base"], "the base model folder changed during its run")
            base = dict(record["base"])
            if p.task == ifeval.TASK:
                base["ifeval_grader"] = ifeval.GRADER
            checkpoint = {"run": common, "base": base, "task_config_sha256": config}
            temp = work / "base.checkpoint.tmp"
            temp.write_text(dumps(checkpoint))
            temp.rename(work / "base.checkpoint.json")
        return config

    external = {"base": a.base_port, "patched": a.patched_port}
    configs = set()
    if a.parallel:
        # Both models in memory: two servers, both halves at once.
        with ExitStack() as stack:
            ports = {variant: stack.enter_context(
                server(a, dirs[variant], s, work / f"{variant}.vllm.log", external[variant]))
                for variant in VARIANTS}
            with ThreadPoolExecutor(len(VARIANTS)) as pool:
                futures = [pool.submit(half, variant, ports[variant], external[variant])
                           for variant in VARIANTS]
                configs = {f.result() for f in futures}
    else:
        variants = VARIANTS
        if a.reuse_base:
            source = Path(a.reuse_base).expanduser().resolve()
            require(SESSION_NAME.fullmatch(source.name), "--reuse-base needs a session work folder")
            base, config = reuse_base(source, stage, common, files["base"], a.server_arg, p, s)
            if p.task == ifeval.TASK:
                base.setdefault("ifeval_grader", None)
            record["base"] = base
            record["base_reused_from"] = source.name
            configs.add(config)
            variants = ("patched",)
            print(f"Reusing {len(ids)} verified base replies from {rel(source)}", flush=True)
        for variant in variants:
            with server(a, dirs[variant], s, work / f"{variant}.vllm.log", external[variant]) as port:
                configs.add(half(variant, port, external[variant]))
    require(len(configs) == 1, "lm_eval resolved the task differently for base and patched")
    for variant in VARIANTS:
        require(model_files(dirs[variant]) == files[variant],
                f"the {variant} model folder changed during the run")
    record.update({
        "harness": {"packages": packages, "task_config_sha256": configs.pop()},
        "timeout": a.timeout,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    })
    if p.task == ifeval.TASK:
        record["ifeval_grader"] = ifeval.GRADER
    (stage / "session.json").write_text(dumps(record))
    # The same verification `check` applies; nothing is filed if it fails.
    sess = derive_session(v, p, s, stage)
    dest = v.results_dir(p, s) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage), dest)

    b, q = sess.outcomes["base"], sess.outcomes["patched"]
    same = sum(b[i]["content_sha256"] == q[i]["content_sha256"] for i in ids)
    label = "prompts passed strict checks" if p.task == ifeval.TASK else "final answers right"
    for variant, o in (("base", b), ("patched", q)):
        right = sum(map(headline, o.values()))
        cut = sum(map(is_cut, o.values()))
        print(f"  {variant:8} {right:g}/{len(ids)} {label}, "
              f"{cut} cut at the token limit")
    print(f"  {same}/{len(ids)} pairs with identical replies")
    print(f"filed {rel(dest)}; commit it, then python -m bench merge")
