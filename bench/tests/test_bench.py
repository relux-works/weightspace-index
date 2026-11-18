"""Offline controls for tensor safety, contracts, and the real harness path."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from bench import bench as b


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("NO_PROXY", "*")
    import requests
    original = requests.sessions.Session.request

    def local_only(self, method, url, **kwargs):
        assert b.urlsplit(url).hostname == "127.0.0.1", "test attempted nonlocal HTTP"
        return original(self, method, url, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", local_only)


@pytest.fixture
def cfg():
    return b.load_config(b.HERE / "config.json")[0]


@pytest.fixture
def synthetic(tmp_path):
    base = tmp_path / "source"
    base.mkdir()
    save_file({"layer.weight": torch.zeros(2, 3), "unchanged": torch.ones(1)}, base / "model.safetensors", metadata={"format": "pt"})
    (base / "config.json").write_text('{}')
    (base / "tokenizer_config.json").write_text('{}')
    patch = tmp_path / "replacement.safetensors"
    save_file({"layer.weight": torch.ones(2, 3)}, patch)
    return base, patch


# Proves valid replacement and each incompatible tensor are handled without base mutation/partial publication.
@pytest.mark.parametrize("invalid", [None, "shape", "dtype", "name", "empty", "duplicate"])
def test_patch_apply(tmp_path, synthetic, invalid):
    base, patch = synthetic
    before = b.inventory(base)
    if invalid in ("shape", "dtype", "name", "empty"):
        tensors = {"layer.weight": torch.ones(2, 3)}
        if invalid == "shape":
            tensors["layer.weight"] = torch.ones(3, 2)
        elif invalid == "dtype":
            tensors["layer.weight"] = torch.ones(2, 3, dtype=torch.float16)
        elif invalid == "name":
            tensors = {"missing": torch.ones(2, 3)}
        else:
            tensors = {}
        save_file(tensors, patch)
    destination = tmp_path / "edited"
    patches = [patch, patch] if invalid == "duplicate" else [patch]
    if invalid:
        with pytest.raises(ValueError, match={"shape": "shape", "dtype": "dtype", "name": "name", "empty": "empty", "duplicate": "duplicate"}[invalid]):
            b.apply_patch(base, patches, destination)
        assert not destination.exists()
    else:
        b.apply_patch(base, patches, destination)
        actual = load_file(destination / "model.safetensors")
        assert torch.equal(actual["layer.weight"], torch.ones(2, 3))
        assert torch.equal(actual["unchanged"], torch.ones(1))
    assert b.inventory(base) == before


# Proves validation requires explicit settings and the fixed output budget, with a nearby valid config.
def test_config(tmp_path, cfg):
    assert cfg["generation"]["max_gen_toks"] == 32768
    for mutate in (lambda c: c.pop("seed"), lambda c: c["generation"].pop("temperature"),
                   lambda c: c.update(repetitions=0), lambda c: c["generation"].update(max_gen_toks=2048),
                   lambda c: c["server"].update(trust_remote_code=True),
                   lambda c: c["generation"].update(top_p=2), lambda c: c["generation"].update(temperature=float("nan"))):
        changed = copy.deepcopy(cfg)
        mutate(changed)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            b.load_config(path)


# Proves exact numeric loopback acceptance and refusal of nonlocal, authenticated, ambiguous or zero-port URLs.
def test_localhost():
    assert b.validate_url("http://127.0.0.1:8000/v1/chat/completions")
    for url in ("http://localhost:8000/v1/chat/completions", "https://127.0.0.1:8000/v1/chat/completions",
                "http://example.invalid:8000/v1/chat/completions", "http://127.0.0.1.evil:8000/v1/chat/completions",
                "http://user:pass@127.0.0.1:8000/v1/chat/completions", "http://127.0.0.1:8000/v1/chat/completions?x=1",
                "http://127.0.0.1:0/v1/chat/completions"):
        with pytest.raises(ValueError):
            b.validate_url(url)


# Proves the shipped selections have 197 unique in-range rows and a pinned source, and corrupt IDs are refused.
def test_frozen_ids(tmp_path):
    for benchmark in b.TASKS:
        frozen = b.load_ids(b.HERE / "data" / f"{benchmark}-ids.json")
        assert len(frozen["indices"]) == 197
        for indices in (frozen["indices"][:-1], [frozen["indices"][0]] * 197, [-1] * 197):
            changed = dict(frozen, indices=indices)
            path = tmp_path / "ids.json"
            b.write_json(path, changed)
            with pytest.raises(ValueError):
                b.load_ids(path)


# Proves generated separator/glued stem refusals preserve bytes, with token-component and empty-value controls.
def test_export(tmp_path):
    path = tmp_path / "result.json"
    allowed = {"max_tokens": 32768, "tokenizer_config": {"value": "normal"},
               "maxTokens": 32768, "tokenizerConfig": {"value": "normal"},
               "tokens": 1, "tokenizer": "normal", "tokenization": "normal",
               "key": "normal", "monkey": "normal", "value": "normal", "model": "normal",
               "max-tokens": 1, "max..tokens": 1, "tokenizer.config": "normal"}
    b.write_json(path, allowed)
    assert b.read_json(path) == allowed
    before = path.read_bytes()
    keys = ("api_key", "apiKey", "apikey", "APIKEY", "access_token", "accessToken", "accesstoken",
            "ACCESSTOKEN", "secret", "password", "passwd", "passWord", "authorization", "Authorization",
            "credential", "cookie", "private_key", "privateKey", "privatekey", "PRIVATEKEY", "token",
            "clientSecret", "clientAPIKey", "client_apikey", "sessionAccessToken", "session_accesstoken",
            "aPi_kEy", "pAsSwOrD", "pRiVaTe_KeY", "aCcEsS_tOkEn", "token_count", "eos_token_id",
            "credentials", "credentials_file", "cookies", "secrets", "tokenCount", "eosTokenId",
            "token-count", "eos.token.id", "token__count", "tOkEn", "my_tOkEn")
    keys += tuple(spelling
                  for left, right in (("api", "key"), ("master", "password"), ("refresh", "token"),
                                      ("client", "secret"), ("private", "key"), ("session", "cookie"),
                                      ("secret", "key"), ("access", "token"), ("pass", "wd"),
                                      ("pass", "word"), ("user", "credential"), ("authorization", "header"))
                  for separator in ("", "_", "-", ".", "__", "..", "-._", "/", " ")
                  for spelling in (left + separator + right, (left + separator + right).upper(),
                                   (left + separator + right).title(), "backup" + left + separator + right + "file"))
    for key in keys:
        with pytest.raises(ValueError, match="secret-named field refused"):
            b.write_json(path, {"nested": [{key: "fixture"}]})
        assert path.read_bytes() == before
    for prefix in ("hf_", "ghp_", "github_pat_", "sk-", "glpat-", "xoxb-"):
        with pytest.raises(ValueError, match="token-shaped value refused"):
            b.write_json(path, {"text": prefix + "x" * 24})
        assert path.read_bytes() == before
    for empty in (None, "", [], {}, 0, False):
        allowed_empty = {key: empty for key in keys}
        b.write_json(path, allowed_empty)
        assert b.read_json(path) == allowed_empty


def frozen_fixture(benchmark):
    original = b.load_ids(b.HERE / "data" / f"{benchmark}-ids.json")
    return dict(original, n=2, indices=[2, 0], population=3)


def documents(benchmark):
    if benchmark == "math500":
        return [{"problem": f"fixture-{name}: What is one plus one?", "solution": r"The result is \boxed{2}.", "answer": "2"}
                for name in ("first", "excluded", "second")]
    return [{"question": f"fixture-choice fixture-{name}", "options": ["zero", "two"], "answer": "B", "category": "math", "cot_content": "The answer is (B)."}
            for name in ("first", "excluded", "second")]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def lane(tmp_path, synthetic, monkeypatch, cfg):
    """Wire tiny local fixtures into the actual CLI path; only downloads/server binary are substituted."""
    source, patch = synthetic
    entry = tmp_path / "models" / "fixture" / "patch" / "v1"
    entry.mkdir(parents=True)
    b.write_json(entry.parent / "patch.json", {"kind": "replacement"})
    b.write_json(entry / "version.json", {"target_profile": "bf16", "files": [
        {"name": "replacement.safetensors", "role": "source", "sha256": b.digest(patch), "size_bytes": patch.stat().st_size,
         "locators": [{"kind": "index-path", "path": "replacement.safetensors"}]}]})
    b.shutil.copyfile(patch, entry / "replacement.safetensors")
    b.write_json(entry.parent.parent / "base.json", {"profiles": [{"id": "bf16", "source": {"repo_id": "fixture/base", "revision": "a" * 40},
        "files": [{"path": "model.safetensors", "sha256": b.digest(source / "model.safetensors"), "size_bytes": (source / "model.safetensors").stat().st_size}]}]})
    here = tmp_path / "package"
    (here / "data").mkdir(parents=True)
    b.write_json(here / "config.json", cfg)
    for benchmark in b.TASKS:
        b.write_json(here / "data" / f"{benchmark}-ids.json", frozen_fixture(benchmark))
    monkeypatch.setattr(b, "HERE", here)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: b.shutil.copytree(source, kw["local_dir"], dirs_exist_ok=True))
    from datasets import Dataset
    import datasets
    state = {"benchmark": "math500", "mode": "complete", "argv": []}

    def dataset_fixture(**kwargs):
        assert kwargs["revision"] == frozen_fixture_from_file(state["benchmark"], here)["revision"]
        return {"test": Dataset.from_list(documents(state["benchmark"]))}

    monkeypatch.setattr(datasets, "load_dataset", dataset_fixture)

    def stub_argv(model, port, kv, config):
        argv = [sys.executable, str(Path(__file__).with_name("stub_server.py")), "--port", str(port),
                "--variant", model.name, "--mode", state["mode"]]
        state["argv"].append(argv)
        return argv

    monkeypatch.setattr(b, "server_argv", stub_argv)
    args = ["run", "--entry", str(entry), "--benchmark", state["benchmark"], "--workdir", str(tmp_path / "work"), "--port", str(free_port())]
    return args, state, tmp_path / "work", here


def frozen_fixture_from_file(benchmark, here):
    return b.load_ids(here / "data" / f"{benchmark}-ids.json")


# Proves real simple_evaluate + stock HTTP backend + scorers + two-variant publication on tiny pinned-row fixtures.
# These are fixture scores only; they say nothing about real model quality or Mac/vLLM support.
@pytest.mark.parametrize("benchmark", ["math500", "mmlu-pro"])
def test_end_to_end(lane, benchmark, capsys):
    args, state, work, here = lane
    state["benchmark"] = benchmark
    args[args.index("--benchmark") + 1] = benchmark
    assert b.main(args) == 0
    result = b.read_json(work / "results" / f"{benchmark}-compare.json")
    assert (result["base"], result["edited"], result["delta"], result["n"]) == (1.0, 0.0, -1.0, 2)
    assert "delta" in capsys.readouterr().out
    for variant in ("base", "edited"):
        path = work / "results" / f"{benchmark}-{variant}.json"
        record = b.read_json(path)
        assert {item["id"] for item in record["items"]} == {0, 2}
        for item in record["items"]:
            doc = item["sample"]["doc"]
            assert ("fixture-second" in doc.get("problem", doc.get("question", ""))) == (item["id"] == 2)
        assert len(record["items"]) == 2 and all(i["finish_reason"] == "unknown" for i in record["items"])
        assert record["settings_verbatim"] == (here / "config.json").read_text()
        assert record["server"]["argv"] == state["argv"][0 if variant == "base" else 1]
        with pytest.raises(ProcessLookupError):
            os.kill(record["server"]["pid"], 0)
        # Replay saved raw generations through the same stock filters/scorer; saved score corruption is corrected.
        for item in record["items"]:
            item["score"] = 1 - item["score"]
        assert b.rescore(record)["aggregate"] == (1.0 if variant == "base" else 0.0)
    assert b.main(["--rescore", str(path), "--output", str(work / "rescored.json")]) == 0


# The passing positive above accompanies missing-generation and HTTP-error plants; neither may publish a score/table.
@pytest.mark.parametrize("mode", ["incomplete", "error"])
def test_incomplete_refused(lane, mode, capsys):
    args, state, work, _ = lane
    state["mode"] = mode
    assert b.main(args) == 1
    assert not list((work / "results").glob("*.json"))
    captured = capsys.readouterr()
    assert "delta" not in captured.out and "benchmark failed" in captured.err
    assert ("missing generation" if mode == "incomplete" else "HTTPError") in captured.err


# Proves duplicate/missing saved rows cannot yield a final comparison, beside a valid fixture-file calculation.
def test_compare_files(tmp_path, cfg):
    record = dict(benchmark="math500", frozen={"n": 2, "indices": [0, 2]}, settings=cfg,
                  settings_verbatim=json.dumps(cfg), identity={"base": "fixture", "edits": []}, lm_eval_version="fixture",
                  variant="base", items=[{"id": 0, "repetition": 0, "generation": "a", "score": 1},
                                         {"id": 2, "repetition": 0, "generation": "b", "score": 0}])
    other = copy.deepcopy(record)
    other.update(variant="edited")
    other["items"][1]["score"] = 1
    b.write_json(tmp_path / "base.json", record)
    b.write_json(tmp_path / "edited.json", other)
    actual = b.compare(b.read_json(tmp_path / "base.json"), b.read_json(tmp_path / "edited.json"))
    assert (actual["base"], actual["edited"], actual["delta"], actual["n"]) == (0.5, 1.0, 0.5, 2)
    for changed in ([], [record["items"][0]] * 2):
        record["items"] = changed
        with pytest.raises(ValueError, match="incomplete"):
            b.compare(record, other)


# Proves a short result from the real harness cannot slip past count validation (the full path passes above).
def test_wrong_count_refused(lane, monkeypatch, capsys):
    from lm_eval import evaluator
    original = evaluator.simple_evaluate

    def missing_row(**kwargs):
        result = original(**kwargs)
        next(iter(result["samples"].values())).pop()
        return result

    monkeypatch.setattr(evaluator, "simple_evaluate", missing_row)
    args, _, work, _ = lane
    assert b.main(args) == 1
    assert "incomplete evaluation" in capsys.readouterr().err
    assert not list((work / "results").glob("*.json"))


# Proves repetitions retain all outputs; cache reuse succeeds, and edited/base corruption is refused offline.
def test_repetition_and_cache(lane, capsys):
    args, _, work, here = lane
    cfg = b.read_json(here / "config.json")
    cfg["repetitions"] = 2
    b.write_json(here / "config.json", cfg)
    assert b.main(args) == 0
    record = b.read_json(work / "results" / "math500-base.json")
    assert len(record["items"]) == 4
    assert {(i["id"], i["repetition"]) for i in record["items"]} == {(0, 0), (0, 1), (2, 0), (2, 1)}
    # Keep the old records; reruns must refuse before changing prepared models.
    before = b.inventory(work / "edited")
    assert b.main(args + ["--skip-download", "--skip-apply"]) == 1
    assert "result exists" in capsys.readouterr().err
    assert b.inventory(work / "edited") == before
    (work / "results").rename(work / "previous-results")
    assert b.main(args + ["--skip-download", "--skip-apply"]) == 0
    (work / "results").rename(work / "second-results")
    (work / "edited" / "config.json").write_text('{"changed": true}')
    assert b.main(args + ["--skip-download", "--skip-apply"]) == 1
    assert "edited cache incomplete/changed" in capsys.readouterr().err
    (work / "base" / "config.json").write_text('{"changed": true}')
    assert b.main(args + ["--skip-download", "--skip-apply"]) == 1
    assert "base cache is incomplete/changed" in capsys.readouterr().err


# Proves occupied ports/startup failures never touch the other listener, and failure cleanup reaps the owned PID.
def test_server_lifecycle(tmp_path):
    argv = [sys.executable, str(Path(__file__).with_name("stub_server.py")), "--port", str(free_port())]
    port = int(argv[-1])
    with b.serve(argv, port, 3, tmp_path / "server.log") as pid:
        with pytest.raises(OSError):
            with b.serve(argv, port, 1, tmp_path / "other.log"):
                pytest.fail("occupied port admitted")
        os.kill(pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    port = free_port()
    with pytest.raises(TimeoutError, match="health timeout"):
        with b.serve([sys.executable, "-c", "import time; time.sleep(30)"], port, 1, tmp_path / "timeout.log"):
            pytest.fail("unhealthy child admitted")
    with pytest.raises(ValueError, match="fixture failure"):
        argv[-1] = str(port)
        with b.serve(argv, port, 3, tmp_path / "failure.log") as failed_pid:
            raise ValueError("fixture failure")
    with pytest.raises(ProcessLookupError):
        os.kill(failed_pid, 0)


# Proves localhost/KV launch knobs reach both supported installed entry forms and no remote host is selected.
def test_server_argv(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(b.importlib.util, "find_spec", lambda name: object())
    argv = b.server_argv(tmp_path / "base", 8123, 1048576, cfg)
    assert argv[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == "1048576"
    monkeypatch.setattr(b.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(b.shutil, "which", lambda name: "vllm")
    argv = b.server_argv(tmp_path / "base", 8123, None, cfg)
    assert argv[:2] == ["vllm", "serve"] and "--kv-cache-memory-bytes" not in argv
    monkeypatch.setattr(b.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="install vLLM"):
        b.server_argv(tmp_path / "base", 8123, None, cfg)
