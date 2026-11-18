"""End-to-end on stock lm-eval against a fake `vllm` executable, in a temp repo.

Needs the pinned harness (bench/requirements.txt) and the MATH-500 test
split (downloaded once by `datasets`, then cached). No model is loaded:
the fake `vllm serve` is an OpenAI-shaped stub that answers from which
weights file it was pointed at.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from bench import cli, core

REAL = Path(core.BENCH)
IDS = [2, 14, 17]  # gold answers: \frac{14}{3}, \sqrt{51}, \pi
ENTRY = "models/Qwen/Qwen3.5-0.8B/p-x/v1"
RUN_ID = "tiny.qwen3.5-0.8b-thinking.bf16"
RESULTS = f"bench/results/Qwen/Qwen3.5-0.8B/p-x/v1/{RUN_ID}"

# Both answer item 14 identically and right. Elsewhere the base is cut in a
# thinking loop and the patched copy answers pi (right for item 17 only).
FAKE_VLLM = '''#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
assert args[0] == "serve", args
root = args[1]
port = int(args[args.index("--port") + 1])
max_len = int(args[args.index("--max-model-len") + 1])
name = args[args.index("--served-model-name") + 1]
patched = open(root + "/model.safetensors", "rb").read() == b"patched weights"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def reply(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path == "/version":
            return self.reply({{"version": "fake"}})
        self.reply({{"data": [{{"id": name, "root": root, "max_model_len": max_len}}]}})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["messages"][-1]["content"].replace("$", "").replace("\\\\", "")
        item14 = "sin D = 0.7" in prompt
        finish = "stop"
        answer = "\\\\sqrt{{51}}" if item14 else "\\\\pi"
        content = "<think>seed %d</think>\\n\\nThe answer is $\\\\boxed{{%s}}$." % (req["seed"], answer)
        if not patched and not item14:  # a thinking loop, cut at the limit
            finish, content = "length", "<think>seed %d. Wait. Wait." % req["seed"]
        self.reply({{"model": req["model"], "choices": [{{
            "index": 0, "finish_reason": finish,
            "message": {{"role": "assistant", "content": content}}}}],
            "usage": {{"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59}}}})

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    bench = root / "bench"
    for sub in ("protocols", "settings", "data"):
        (bench / sub).mkdir(parents=True)
    shutil.copy(REAL / "requirements.txt", bench)
    shutil.copy(REAL / "settings" / "qwen3.5-0.8b-thinking.json", bench / "settings")
    (bench / "data" / "ids.json").write_text(json.dumps({"indices": IDS}))
    proto = json.loads((REAL / "protocols" / "math500-n197.json").read_text())
    proto.update(id="tiny", items=[], seeds=[1234],
                 selection={"file": "data/ids.json",
                            "sha256": core.sha256_file(bench / "data" / "ids.json")[0]})
    (bench / "protocols" / "tiny.json").write_text(json.dumps(proto))

    weights = "model.safetensors"
    models = root / "models" / "Qwen" / "Qwen3.5-0.8B"
    (models / "p-x" / "v1" / "files").mkdir(parents=True)
    dirs = {}
    for variant, data in (("base", b"base weights"), ("patched", b"patched weights")):
        d = root / "served" / variant
        d.mkdir(parents=True)
        (d / weights).write_bytes(data)
        (d / "config.json").write_text("{}")
        dirs[variant] = d
    (models / "base.json").write_text(json.dumps({"contract_version": 1, "profiles": [{
        "id": "bf16", "source": {"repo_id": "Qwen/Qwen3.5-0.8B", "revision": "0" * 40},
        "files": [{"path": weights, "sha256": core.sha256_file(dirs["base"] / weights)[0],
                   "size_bytes": 12}]}]}))
    (models / "p-x" / "patch.json").write_text(json.dumps({"kind": "replacement"}))
    (models / "p-x" / "v1" / "version.json").write_text(json.dumps({
        "target_profile": "bf16", "files": [{"name": "r.safetensors", "role": "source",
                                             "locators": [{"kind": "index-path",
                                                           "path": "files/r.safetensors"}]}]}))
    vllm = root / "fake-vllm"
    vllm.write_text(FAKE_VLLM.format(python=sys.executable))
    vllm.chmod(0o755)

    saved = {k: getattr(core, k) for k in ("REPO", "BENCH", "MODELS", "RESULTS")}
    core.REPO, core.BENCH, core.MODELS = root, bench, root / "models"
    core.RESULTS = bench / "results"
    yield {"root": root, "dirs": dirs, "vllm": str(vllm)}
    for k, v in saved.items():
        setattr(core, k, v)


def bench(*args: str) -> int:
    return cli.main(list(args))


ARGS = ("--entry", ENTRY, "--protocol", "tiny", "--settings", "qwen3.5-0.8b-thinking")


def run(repo, *extra: str, base=None, patched=None) -> int:
    return bench("run", *ARGS, "--vllm", repo["vllm"],
                 "--base-dir", str(base or repo["dirs"]["base"]),
                 "--patched-dir", str(patched or repo["dirs"]["patched"]), *extra)


def sessions(repo) -> list[Path]:
    return sorted((repo["root"] / RESULTS).iterdir())


def test_end_to_end(repo, capsys):
    assert bench("freeze", "--protocol", "tiny") == 0
    p = core.Protocol.load("tiny")
    assert p.ids == IDS and p.seeds == [1234]

    # Swapped folders are refused before any server starts.
    assert run(repo, "--items", "0:1", base=repo["dirs"]["patched"]) == 1
    assert "not the base profile's file" in capsys.readouterr().err
    assert bench("run", *ARGS, "--seed", "99", "--dry-run") == 1
    assert "not one of the protocol's seeds" in capsys.readouterr().err

    assert run(repo, "--items", "0:2", "--performed-by", "index") == 0  # items 2 and 14
    out = capsys.readouterr().out
    assert "seed 1234, 2 items (items 0:2): [2, 14]" in out
    assert "1/2 pairs with identical replies" in out
    assert run(repo, "--performed-by", "index") == 0  # random: only item 17 is open
    assert "1 items (random among pairs filed 0 time(s)): [17]" in capsys.readouterr().out

    first = sessions(repo)[0]
    assert sorted(f.name for f in first.iterdir()) == core.SESSION_FILES
    for variant in ("base", "patched"):
        sent = [json.loads(l)["request"] for l in (first / f"{variant}.requests.jsonl").read_text().splitlines()]
        assert {(r["seed"], r["temperature"], r["top_k"], r["presence_penalty"], r["max_tokens"])
                for r in sent} == {(1234, 1.0, 20, 1.5, 32768)}
        assert sent[0]["stop"] == [] and sent[0]["chat_template_kwargs"] == {"enable_thinking": True}
        # The settings carry the card's math prompt in place of the stock one.
        # Replies come back in any order, so look at every prompt.
        prompts = [r["messages"][-1]["content"] for r in sent]
        card = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        assert all(q.endswith(card) and "Problem:" not in q for q in prompts)
        assert any(q.endswith("fraction." + card) for q in prompts)  # item 2
    assert str(repo["root"]) not in (first / "session.json").read_text()

    capsys.readouterr()
    assert bench("check") == 1                       # sessions filed, runs not merged yet
    assert bench("merge") == 0
    assert bench("check") == 0
    root = repo["root"]
    patched = json.loads((root / f"{ENTRY}/runs/{RUN_ID}/run.json").read_text())
    base = json.loads((root / f"models/Qwen/Qwen3.5-0.8B/runs/{RUN_ID}.p-x-v1/run.json").read_text())
    assert patched["state"] == base["state"] == "COMPLETE"
    assert patched["performed_by"] == base["performed_by"] == "index"
    assert patched["score"]["value"] == pytest.approx(2 / 3)
    assert base["score"]["value"] == pytest.approx(1 / 3)
    assert patched["settings"]["paired_with"] == f"models/Qwen/Qwen3.5-0.8B/runs/{RUN_ID}.p-x-v1"

    capsys.readouterr()
    assert bench("compare", *ARGS) == 0
    c = json.loads(capsys.readouterr().out.split("\n\n")[0])
    assert c["complete"] and c["items"] == 3 and c["difference"] == pytest.approx(1 / 3)
    assert c["pairs_identical_replies"] == 1 and c["pairs_only_patched_right"] == 1


def test_repeat_counts_once(repo, capsys):
    # Item 2 again, both servers at once: same seed, same replies.
    assert run(repo, "--items", "0:1", "--parallel") == 0
    assert '"parallel": true' in (sessions(repo)[-1] / "session.json").read_text()
    assert bench("check") == 1                       # merged runs are stale
    assert bench("merge") == 0 and bench("check") == 0
    root = repo["root"]  # sessions by index and by a third party
    for path in (f"{ENTRY}/runs/{RUN_ID}", f"models/Qwen/Qwen3.5-0.8B/runs/{RUN_ID}.p-x-v1"):
        assert json.loads((root / path / "run.json").read_text())["performed_by"] == "third-party"
    capsys.readouterr()
    assert bench("compare", *ARGS) == 0
    c = json.loads(capsys.readouterr().out.split("\n\n")[0])
    assert c["pairs"] == 3 and c["repeats"] == 1 and c["repeats_identical_to_first"] == 1
    capsys.readouterr()
    assert bench("status", *ARGS) == 0
    assert "seed 1234: 3 filed, 0 open, 1 repeated" in capsys.readouterr().out


def edit_lines(path: Path, fn) -> str:
    before = path.read_text()
    path.write_text("".join(json.dumps(fn(json.loads(l))) + "\n" for l in before.splitlines()))
    return before


@pytest.mark.parametrize("file,fn,message", [
    ("patched.requests.jsonl", lambda x: {**x, "request": {**x["request"], "top_k": 40}}, "top_k"),
    ("patched.requests.jsonl", lambda x: {**x, "request": {**x["request"], "seed": 7}}, "seed"),
    ("base.requests.jsonl", lambda x: {**x, "request": {**x["request"], "logprobs": True}}, "unexpected"),
    ("base.samples.jsonl", lambda x: {**x, "doc_hash": "0" * 64}, "dataset content differs"),
    ("patched.samples.jsonl", lambda x: {**x, "resps": [["\\boxed{6}"]]}, "not the recorded reply"),
    ("base.requests.jsonl", lambda x: {**x, "status": 500}, "failed"),
])
def test_tampering_detected(repo, capsys, file, fn, message):
    path = sessions(repo)[0] / file
    before = edit_lines(path, fn)
    try:
        capsys.readouterr()
        assert bench("check") == 1
        assert message in capsys.readouterr().err
    finally:
        path.write_text(before)
    assert bench("check") == 0


def test_extra_exchange_detected(repo, capsys):
    path = sessions(repo)[1] / "patched.requests.jsonl"
    before = path.read_text()
    path.write_text(before + before)
    try:
        capsys.readouterr()
        assert bench("check") == 1
        assert "exchanges for 1 items" in capsys.readouterr().err
    finally:
        path.write_text(before)


def test_swapped_weights_detected(repo, capsys):
    path = sessions(repo)[0] / "session.json"
    before = path.read_text()
    rec = json.loads(before)
    rec["base"], rec["patched"] = rec["patched"], rec["base"]
    path.write_text(json.dumps(rec))
    try:
        capsys.readouterr()
        assert bench("check") == 1
        assert "not the base profile's file" in capsys.readouterr().err
    finally:
        path.write_text(before)
