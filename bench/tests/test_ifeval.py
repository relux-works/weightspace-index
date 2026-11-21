"""Exercise strict IFEval grading through the same verifier used by run and check."""
import json
from pathlib import Path

import pytest

from bench import comparison, core, harness, sessions
from bench.runs import Collected


DOC = {
    "key": 1,
    "prompt": "Say hello. Include the word 'hello' and do not use commas.",
    "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
    "kwargs": [{"keywords": ["hello"]}, {}],
}


@pytest.fixture
def paired_reply():
    doc_hash = core.sha256_bytes(json.dumps(DOC, indent=2, ensure_ascii=False).encode())
    target_hash = core.sha256_bytes(b"0")
    protocol = core.Protocol(Path("ifeval.json"), "", {
        "task": "ifeval", "primary_metric": "prompt_level_strict_acc",
        "items": [[0, doc_hash, target_hash]], "num_fewshot": 0, "gen_kwargs": {"until": []},
        "dataset": {"repo_id": "google/IFEval", "revision": "a" * 40,
                    "split": "train", "pin_revision": True},
    })
    settings = core.Settings(Path("qwen.json"), "", {
        "gen_kwargs": {"max_gen_toks": 32768, "temperature": 1.0},
        "model_args": {"think_end_token": "</think>"},
    })
    messages = [{"role": "user", "content": DOC["prompt"]}]
    sample = {
        "doc_id": 0, "doc": dict(DOC), "doc_hash": doc_hash,
        "target": "0", "target_hash": target_hash,
        "arguments": {"gen_args_0": {"arg_0": json.dumps(messages)}},
    }
    choice = {"message": {}, "finish_reason": "stop"}
    exchange = {
        "status": 200,
        "request": {**core.expected_payload(protocol, settings, "bench", 1234), "messages": messages},
        "response": {"choices": [choice], "usage": {"prompt_tokens": 21, "completion_tokens": 10}},
    }
    return protocol, settings, sample, exchange, choice


def record_reply(paired_reply, text, lm_score, finish="stop"):
    protocol, settings, sample, _, choice = paired_reply
    sample["resps"] = [[sessions.lm_eval_answer(text, settings.think_end)]]
    sample[protocol.metric] = lm_score
    choice["message"]["content"] = text
    choice["finish_reason"] = finish


def verify(paired_reply):
    protocol, settings, sample, exchange, _ = paired_reply
    return sessions.derive_variant(protocol, settings, [0], 1234, [sample], [exchange])[0]


@pytest.mark.parametrize(("text", "lm_score", "finish", "answer", "correct"), [
    ("Think, think.</think>\nHello friend.", True, "stop", "Hello friend.", True),
    ("Thoughts</think>Hello, friend.", False, "stop", "Hello, friend.", False),
    ("Thoughts</think>Good morning.", False, "stop", "Good morning.", False),
    ("hello", True, "length", None, False),
    ("Thinking</think> \n", False, "stop", None, False),
    ("Thinking</think>Hello", True, "length", "Hello", True),
])
def test_verifies_strict_visible_reply(paired_reply, text, lm_score, finish, answer, correct):
    record_reply(paired_reply, text, lm_score, finish)
    outcome = verify(paired_reply)
    assert outcome["answer"] == answer
    assert outcome["correct"] is correct
    assert outcome["score"] == float(lm_score)
    assert outcome["finish_reason"] == finish
    assert "correct_is_equiv" not in outcome


def test_refuses_tampered_instructions(paired_reply):
    record_reply(paired_reply, "Thinking</think>Hello", True)
    paired_reply[2]["doc"]["kwargs"] = [{"keywords": ["goodbye"]}, {}]
    with pytest.raises(core.BenchError, match="logged document"):
        verify(paired_reply)


def test_refuses_tampered_prompt_in_both_logs(paired_reply):
    record_reply(paired_reply, "Thinking</think>Hello", True)
    messages = [{"role": "user", "content": "Say hello."}]
    paired_reply[2]["arguments"]["gen_args_0"]["arg_0"] = json.dumps(messages)
    paired_reply[3]["request"]["messages"] = messages
    with pytest.raises(core.BenchError, match="stock IFEval prompt"):
        verify(paired_reply)


def test_refuses_tampered_lm_eval_score(paired_reply):
    record_reply(paired_reply, "Thinking</think>Hello", False)
    with pytest.raises(core.BenchError, match="strict score"):
        verify(paired_reply)


def test_refuses_separate_reasoning_field(paired_reply):
    record_reply(paired_reply, "Hello", True)
    paired_reply[4]["message"]["reasoning"] = "Thinking"
    with pytest.raises(core.BenchError, match="think-end token"):
        verify(paired_reply)


def test_pins_revision_and_keeps_stock_prompt(paired_reply, tmp_path):
    protocol, settings, *_ = paired_reply
    path = Path(harness.task_arg(protocol, settings, tmp_path))
    override = path.read_text()
    assert "dataset_kwargs:" in override and protocol.data["dataset"]["revision"] in override
    assert "doc_to_text:" not in override


def test_compares_paired_passes_and_cuts(paired_reply, monkeypatch):
    protocol, settings, *_ = paired_reply
    protocol.data["seeds"] = [1234, 5678]
    record_reply(paired_reply, "hello", True, "length")
    base = verify(paired_reply)
    record_reply(paired_reply, "Thoughts</think>Hello.", True)
    patched = verify(paired_reply)
    pairs = {(0, 1234): ("pilot", base, patched)}
    monkeypatch.setattr(comparison, "collect", lambda *_: Collected([], pairs))

    result = comparison.compare(None, protocol, settings)
    assert result["pairs"] == 1 and not result["complete"]
    assert result["difference"] == 1 and result["lm_eval"]["difference"] == 0
    assert result["pairs_only_patched_right"] == 1
    assert result["base"]["cut_at_limit"] == 1 and result["patched"]["cut_at_limit"] == 0
    assert "score_is_equiv" not in result["base"]
    assert "interval_95" not in result
