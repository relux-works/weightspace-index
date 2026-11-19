"""python -m bench <command>; see bench/README.md."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import random
import re
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

from . import core
from .core import (HARNESS_MODEL, PERFORMED_BY, VARIANTS, BenchError, Protocol, Settings,
                   Version, all_results, build_runs, check_settings_fit, check_weights, compare,
                   coverage, derive_session, dumps, gen_kwargs, read_json, read_jsonl, rel,
                   require, sha256_file, task_config_sha256)
from .proxy import Recorder, get_json

SERVED_NAME = "bench"


def work_dir(*parts: str) -> Path:
    path = core.REPO.joinpath(".temp", "bench", *parts)
    path.mkdir(parents=True)
    return path


def pinned_packages() -> dict[str, str]:
    """Installed versions of every pin in bench/requirements.txt; any
    difference is an error, since another scorer version can score the
    same reply differently."""
    out = {}
    for line in (core.BENCH / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, version = line.partition("==")
        name = re.sub(r"\[.*\]", "", name).strip()
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise BenchError(f"{name} is not installed; pip install -r bench/requirements.txt") from None
        require(installed == version, f"{name} {installed} installed, bench/requirements.txt pins {version}")
        out[name] = installed
    return out


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


def task_arg(p: Protocol, s: Settings | None, work: Path) -> str:
    """The stock task name, or, when the settings carry the card's prompt
    for this task, a task file that includes the installed stock task and
    replaces only its doc_to_text (lm-eval's own include mechanism)."""
    prompt = s.doc_to_text(p.task) if s else None
    if prompt is None:
        return p.task
    root = Path(importlib.util.find_spec("lm_eval").submodule_search_locations[0]) / "tasks"
    stock = list(root.rglob(f"{p.task}.yaml"))
    require(len(stock) == 1, f"expected one installed {p.task}.yaml, found {len(stock)}")
    path = work / f"{p.task}.yaml"
    # JSON strings are valid YAML double-quoted scalars.
    path.write_text(f"include: {json.dumps(str(stock[0]))}\ndoc_to_text: {json.dumps(prompt)}\n")
    return str(path)


def lm_eval_argv(p: Protocol, s: Settings | None, ids: list[int], base_url: str | None,
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
    return [sys.executable, "-m", "lm_eval", "run", "--model", model,
            "--model_args", json.dumps(model_args), "--tasks", task or p.task,
            "--num_fewshot", str(p.data["num_fewshot"]), *chat,
            "--gen_kwargs", json.dumps(gen_kwargs(p, s) if s else p.data["gen_kwargs"]),
            "--samples", json.dumps({p.task: ids}), "--log_samples", "--output_path", str(out)]


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
    return config


# ---------------------------------------------------------------- selection


def parse_slice(text: str) -> slice:
    m = re.fullmatch(r"(-?\d*):(-?\d*)", text)
    require(m, f"--items {text!r}: expected START:END over the protocol's item order, e.g. 0:20")
    a, b = (int(x) if x else None for x in m.groups())
    return slice(a, b)


def select(p: Protocol, counts: dict[tuple[int, int], int], seed: int | None,
           items: str | None, count: int, rng: random.Random) -> tuple[int, list[int], str]:
    """(seed, item ids, how they were chosen)."""
    require(seed is None or seed in p.seeds,
            f"--seed {seed} is not one of the protocol's seeds {p.seeds}")
    if items is not None:
        require(seed is not None or len(p.seeds) == 1, "the protocol has several seeds; pass --seed")
        ids = p.ids[parse_slice(items)]
        require(ids, f"--items {items} selects nothing")
        return seed if seed is not None else p.seeds[0], ids, f"items {items}"
    pool = [c for c in p.cells() if seed is None or c[1] == seed]
    least = min(counts[c] for c in pool)
    pool = [c for c in pool if counts[c] == least]
    chosen_seed = seed if seed is not None else rng.choice(sorted({c[1] for c in pool}))
    ids = sorted(rng.sample([i for i, sd in pool if sd == chosen_seed],
                            min(count, sum(sd == chosen_seed for _, sd in pool))))
    how = f"random among pairs filed {least} time(s)"
    return chosen_seed, ids, how


# ---------------------------------------------------------------- models


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


# ---------------------------------------------------------------- commands


def cmd_freeze(a) -> None:
    """Fill a draft protocol's items with lm-eval's own doc/target hashes."""
    p = Protocol.load(a.protocol)
    require(not p.data["items"], f"{rel(p.path)} is already frozen")
    sel = p.data["selection"]
    sel_path = core.BENCH / sel["file"]
    require(sha256_file(sel_path)[0] == sel["sha256"], f"{sel['file']} differs from its pinned sha256")
    ids = sorted(read_json(sel_path)["indices"])
    pinned_packages()
    work = work_dir("freeze", f"{p.id}-{time.time_ns()}")
    run_logged(lm_eval_argv(p, None, ids, None, None, None, work / "out"), work / "lm_eval.log",
               "lm_eval")
    samples, results = lm_eval_outputs(work / "out", p.task)
    check_results(p, results, "dummy")
    hashes = {x["doc_id"]: [x["doc_hash"], x["target_hash"]] for x in read_jsonl(samples)}
    require(sorted(hashes) == ids, "lm_eval returned other items than the selection")
    p.data["items"] = [[i, *hashes[i]] for i in ids]
    text = dumps(p.data)
    # One item per line, and the seeds on one line, keep diffs readable.
    text = re.sub(r'\[\n\s+(\d+),\n\s+("[0-9a-f]{64}"),\n\s+("[0-9a-f]{64}")\n\s+\]',
                  r"[\1, \2, \3]", text)
    text = re.sub(r'"seeds": \[[^\]]*\]', lambda m: '"seeds": ' + json.dumps(p.seeds), text)
    p.path.write_text(text)
    print(f"froze {len(ids)} items into {rel(p.path)}")


def load(a) -> tuple[Version, Protocol, Settings]:
    v, p, s = Version.resolve(a.entry), Protocol.load(a.protocol), Settings.load(a.settings)
    check_settings_fit(v, s)
    require(p.data["items"], f"{rel(p.path)} has no frozen items; run python -m bench freeze")
    return v, p, s


def cmd_status(a) -> None:
    v, p, s = load(a)
    counts = coverage(v, p, s)
    print(f"{rel(v.results_dir(p, s))}")
    print(f"  {len(p.ids)} items x {len(p.seeds)} seed(s) = {len(counts)} pairs")
    for seed in p.seeds:
        mine = [n for (i, sd), n in counts.items() if sd == seed]
        print(f"  seed {seed}: {sum(n > 0 for n in mine)} filed, {sum(n == 0 for n in mine)} open, "
              f"{sum(max(n - 1, 0) for n in mine)} repeated")
    open_ids = [i for i in p.ids if any(counts[(i, sd)] == 0 for sd in p.seeds)]
    if open_ids and len(open_ids) <= 40:
        print(f"  open items: {open_ids}")


def cmd_run(a) -> None:
    check_server_args(a.server_arg)
    v, p, s = load(a)
    seed, ids, how = select(p, coverage(v, p, s), a.seed, a.items, a.count, random.Random())
    a.concurrency = a.concurrency or len(ids)
    require(a.concurrency >= 1, "--concurrency must be at least 1")
    print(f"{p.id} / {s.id} on {v.folder}: seed {seed}, {len(ids)} items ({how}): {ids}")
    if a.dry_run:
        return
    packages = pinned_packages()

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

    task = task_arg(p, s, work)
    record = {"protocol_sha256": p.sha256, "settings_sha256": s.sha256,
              "profile": v.profile["id"], "seed": seed, "items": ids}
    def half(variant: str, port: int, external: int | None) -> str:
        """One model's half of the session, against a server on `port`."""
        t0 = time.time()
        served, info = check_server(port, dirs[variant], s)
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
        for variant in VARIANTS:
            with server(a, dirs[variant], s, work / f"{variant}.vllm.log", external[variant]) as port:
                configs.add(half(variant, port, external[variant]))
    require(len(configs) == 1, "lm_eval resolved the task differently for base and patched")
    for variant in VARIANTS:
        require(model_files(dirs[variant]) == files[variant],
                f"the {variant} model folder changed during the run")
    record.update({
        "harness": {"packages": packages, "task_config_sha256": configs.pop()},
        "concurrency": a.concurrency, "parallel": a.parallel, "timeout": a.timeout,
        "performed_by": a.performed_by, "hardware": a.hardware,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    })
    (stage / "session.json").write_text(dumps(record))
    # The same verification `check` applies; nothing is filed if it fails.
    sess = derive_session(v, p, s, stage)
    dest = v.results_dir(p, s) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage), dest)

    b, q = sess.outcomes["base"], sess.outcomes["patched"]
    same = sum(b[i]["content_sha256"] == q[i]["content_sha256"] for i in ids)
    for variant, o in (("base", b), ("patched", q)):
        right = sum(x["score"] for x in o.values())
        cut = sum(x["finish_reason"] == "length" for x in o.values())
        print(f"  {variant:8} {right:g}/{len(ids)} {p.metric}, {cut} cut at the token limit")
    print(f"  {same}/{len(ids)} pairs with identical replies")
    print(f"filed {rel(dest)}; commit it, then python -m bench merge")


def targets(a) -> list[tuple[Version, Protocol, Settings]]:
    given = {a.entry is None, a.protocol is None, a.settings is None}
    require(len(given) == 1, "pass --entry, --protocol and --settings together, or none of them")
    return all_results() if a.entry is None else [load(a)]


def cmd_merge(a) -> None:
    for v, p, s in targets(a):
        runs = build_runs(v, p, s)
        for variant, out in v.run_dirs(p, s).items():
            run_text, lines = runs[variant]
            out.mkdir(parents=True, exist_ok=True)
            (out / "run.json").write_text(run_text)
            (out / "outcomes.jsonl").write_text(lines)
            print(f"wrote {rel(out)} ({json.loads(run_text)['state']})")
    print("then regenerate the index tables: cd tool && cargo run --locked -- generate ..")


def cmd_check(a) -> None:
    problems = []
    for v, p, s in all_results():
        try:
            runs = build_runs(v, p, s)
        except BenchError as e:
            problems.append(str(e))
            continue
        for variant, out in v.run_dirs(p, s).items():
            for name, text in zip(("run.json", "outcomes.jsonl"), runs[variant]):
                path = out / name
                if not path.is_file() or path.read_text() != text:
                    problems.append(f"{rel(path)}: differs from its sessions; run python -m bench merge")
    for problem in problems:
        print(problem, file=sys.stderr)
    require(not problems, f"{len(problems)} problem(s)")
    print("bench results consistent")


def cmd_compare(a) -> None:
    v, p, s = load(a)
    c = compare(v, p, s)
    print(json.dumps(c, indent=2))
    state = "complete" if c["complete"] else f"PARTIAL, {c['pairs']} of {c['protocol_pairs']} pairs"
    print(f"\n{p.metric} over {c['items']} items ({state}): base {c['base']['score']:.3f}, "
          f"patched {c['patched']['score']:.3f}, difference {c['difference']:+.3f} "
          f"(95% {c['interval_95'][0]:+.3f} .. {c['interval_95'][1]:+.3f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def entry_args(cmd, required=True):
        cmd.add_argument("--entry", required=required,
                         help="patch version folder, models/<ns>/<model>/<patch>/v<N>")
        cmd.add_argument("--protocol", required=required, help="bench/protocols/<id>.json")
        cmd.add_argument("--settings", required=required, help="bench/settings/<id>.json")

    c = sub.add_parser("freeze", help="fill a draft protocol's frozen item hashes")
    c.add_argument("--protocol", required=True)
    c.set_defaults(fn=cmd_freeze)

    c = sub.add_parser("status", help="which item-seed pairs are filed")
    entry_args(c)
    c.set_defaults(fn=cmd_status)

    c = sub.add_parser("run", help="run base and patched on the same items and seed, and file the session")
    entry_args(c)
    c.add_argument("--seed", type=int, help="one of the protocol's seeds (default: chosen with the items)")
    c.add_argument("--items", help="START:END slice of the protocol's items, e.g. 0:20 "
                                   "(default: --count random items among the least-filed pairs)")
    c.add_argument("--count", type=int, default=10, help="items per session when picking at random")
    c.add_argument("--dry-run", action="store_true", help="print the selection and stop")
    c.add_argument("--base-dir", help="base model folder (default: downloaded under .temp/bench/models)")
    c.add_argument("--patched-dir", help="patched copy (default: built with weightspace apply)")
    c.add_argument("--vllm", default="vllm", help="vllm executable")
    c.add_argument("--server-arg", action="append", default=[],
                   help="extra vllm serve argument, repeatable; write --server-arg=--flag")
    c.add_argument("--base-port", type=int, help="use a server already serving --base-dir on this port")
    c.add_argument("--patched-port", type=int, help="use a server already serving --patched-dir on this port")
    c.add_argument("--parallel", action="store_true",
                   help="serve base and patched at the same time (needs memory for both models)")
    c.add_argument("--concurrency", type=int,
                   help="requests in flight per server (default: all of the session's items); "
                        "1 makes reruns on the same setup byte-identical")
    c.add_argument("--timeout", type=int, default=7200,
                   help="seconds lm-eval waits for one reply (its default, 300, is too short "
                        "for long thinking replies)")
    c.add_argument("--startup-timeout", type=int, default=1800, help="seconds to wait for vllm")
    c.add_argument("--performed-by", default="third-party", choices=PERFORMED_BY)
    c.add_argument("--hardware", help="free text, e.g. 'M1 Pro 32 GB, vllm-metal'")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("merge", help="write run.json + outcomes.jsonl from the filed sessions")
    entry_args(c, required=False)
    c.set_defaults(fn=cmd_merge)

    c = sub.add_parser("check", help="every merged run equals a fresh merge of its sessions")
    c.set_defaults(fn=cmd_check)

    c = sub.add_parser("compare", help="paired base-versus-patched comparison of the filed pairs")
    entry_args(c)
    c.set_defaults(fn=cmd_compare)

    a = parser.parse_args(argv)
    try:
        a.fn(a)
    except (BenchError, OSError) as e:
        print(f"bench: {e}", file=sys.stderr)
        return 1
    return 0
