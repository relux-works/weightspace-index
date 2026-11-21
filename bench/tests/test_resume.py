"""A patched-only run reuses a verified base half for the exact same inputs."""

import json

import pytest

from bench import core, runner


def test_reuse_base_checks_the_pairing_and_keeps_the_original_files(tmp_path, monkeypatch):
    source = tmp_path / "old-session"
    source_stage = source / "session"
    source_stage.mkdir(parents=True)
    stage = tmp_path / "new-session"
    stage.mkdir()
    for kind in ("requests", "samples"):
        (source_stage / f"base.{kind}.jsonl").write_text("{}\n")
    common = {"protocol_sha256": "protocol", "settings_sha256": "settings",
              "items": [1, 2], "seed": 1234, "profile": "bf16", "concurrency": 8,
              "hardware": "test host", "parallel": False}
    base = {"model_files": {"model.safetensors": "digest"}, "server": {"managed": True, "args": []}}
    checkpoint = {"run": common, "base": base, "task_config_sha256": "task"}
    (source / "base.checkpoint.json").write_text(json.dumps(checkpoint))

    called = []
    monkeypatch.setattr(runner, "derive_variant", lambda *args, **kwargs: called.append(args[2:4]))
    with pytest.raises(core.BenchError, match="seed"):
        runner.reuse_base(source, stage, {**common, "seed": 0}, base["model_files"], [], None, None)
    assert list(stage.iterdir()) == []
    assert runner.reuse_base(source, stage, common, base["model_files"], [], None, None) == (base, "task")
    assert called == [([1, 2], 1234)]
    for kind in ("requests", "samples"):
        original = source_stage / f"base.{kind}.jsonl"
        assert (stage / original.name).read_bytes() == original.read_bytes()
