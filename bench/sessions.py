"""Verify the evidence from a paired benchmark session."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import ifeval
from .core import (SESSION_FILES, VARIANTS, BenchError, Protocol, Settings, Version,
                   canonical, check_weights, expected_payload, non_weights, read_json,
                   read_jsonl, rel, require, sha256_bytes)
from .grading import MATH_TASK, final_answer, grade


def lm_eval_answer(content: str, think_end: str | None) -> str:
    """lm-eval 0.4.13 postprocess_generated_text with stop=None."""
    return content.split(think_end)[-1].lstrip() if think_end else content


def sample_messages(sample: dict) -> list:
    arg = sample["arguments"]["gen_args_0"]["arg_0"]
    if isinstance(arg, list):
        require(len(arg) == 1, "unexpected sample arguments")
        arg = arg[0]
    return json.loads(arg) if isinstance(arg, str) else arg


def derive_variant(p: Protocol, s: Settings, ids: list[int], seed: int, samples: list[dict],
                   exchanges: list[dict]) -> dict[int, dict]:
    """Outcomes of one model's half of a session, by item.

    Each lm-eval sample must be one of the session's frozen items with the
    frozen hashes, and must match exactly one recorded exchange whose
    request carried exactly the expected parameters and seed, and whose
    reply, cut the way lm-eval cuts it, is the text lm-eval scored. The
    protocol's grader scores the final answer using the verified sample."""
    require(p.task in (MATH_TASK, ifeval.TASK), f"bench does not grade {p.task}")
    if p.task == ifeval.TASK:
        require(p.metric == ifeval.METRIC, f"IFEval grades only {ifeval.METRIC}")
    frozen = p.items
    by_messages: dict[str, list[dict]] = {}
    served = None
    for x in exchanges:
        require(x.get("status") == 200, f"a request failed ({x.get('status')}); rerun")
        req = x["request"]
        served = served or req.get("model")
        want = expected_payload(p, s, served, seed)
        extra = set(req) - set(want) - {"messages"}
        require(not extra, f"request carried unexpected {sorted(extra)}")
        for key, value in want.items():
            require(key in req and canonical(req[key]) == canonical(value),
                    f"request {key}={req.get(key)!r}, expected {value!r}")
        by_messages.setdefault(canonical(req["messages"]), []).append(x)
    require(len(exchanges) == len(ids), f"{len(exchanges)} exchanges for {len(ids)} items")

    out: dict[int, dict] = {}
    for sample in samples:
        i = sample["doc_id"]
        require(i in ids and i not in out, f"sample {i} is not in the session or repeats")
        require((sample["doc_hash"], sample["target_hash"]) == frozen[i],
                f"item {i}: dataset content differs from the frozen protocol")
        scored = sample["resps"][0][0]
        messages = sample_messages(sample)
        matches = by_messages.get(canonical(messages), [])
        require(len(matches) == 1, f"item {i}: {len(matches)} recorded exchanges match its prompt")
        resp = matches[0]["response"]
        require(len(resp["choices"]) == 1, f"item {i}: expected one choice")
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        require(isinstance(content, str) and isinstance(scored, str)
                and lm_eval_answer(content, s.think_end) == scored,
                f"item {i}: scored text is not the recorded reply")
        target = sample["target"]
        require(isinstance(target, str) and sha256_bytes(target.encode()) == sample["target_hash"],
                (f"item {i}: target is not the frozen gold answer" if p.task == MATH_TASK
                 else f"item {i}: target differs from the frozen IFEval target"))
        if p.task == ifeval.TASK:
            try:
                verdict = ifeval.grade_sample(sample, messages, choice, scored, s.think_end)
            except BenchError as e:
                raise BenchError(f"item {i}: {e}") from None
        else:
            score = sample.get(p.metric)
            require(isinstance(score, (int, float)) and not isinstance(score, bool),
                    f"item {i}: no numeric {p.metric}")
            answer = final_answer(content, s.think_end)
            correct, correct_is_equiv = grade(answer, target)
            verdict = {"score": float(score), "answer": answer, "correct": correct,
                       "correct_is_equiv": correct_is_equiv}
        out[i] = {**verdict, "finish_reason": choice["finish_reason"],
                  "prompt_tokens": resp["usage"]["prompt_tokens"],
                  "completion_tokens": resp["usage"]["completion_tokens"],
                  "content_sha256": sha256_bytes(content.encode())}
    require(set(out) == set(ids), f"items {sorted(set(ids) - set(out))} not scored")
    return out


def task_config_sha256(config: dict) -> str:
    """lm-eval's resolved task config without `metadata` (local install
    path, endpoint) and without the memory addresses lm-eval prints for
    function references (`<function f at 0x...>`)."""
    text = canonical({k: v for k, v in config.items() if k != "metadata"})
    return sha256_bytes(re.sub(r" at 0x[0-9a-f]+>", ">", text).encode())


@dataclass
class Session:
    name: str
    record: dict
    outcomes: dict[str, dict[int, dict]]   # variant -> item -> outcome


def derive_session(v: Version, p: Protocol, s: Settings, d: Path) -> Session:
    """Verifies one session folder; the same check `run` applies before
    filing it and `check` applies to every filed one."""
    try:
        names = sorted(f.name for f in d.iterdir())
        require(names == SESSION_FILES, f"expected exactly {', '.join(SESSION_FILES)}")
        rec = read_json(d / "session.json")
        require(rec["protocol_sha256"] == p.sha256 and rec["settings_sha256"] == s.sha256,
                f"run under another version of {rel(p.path)} or {rel(s.path)}")
        require(rec["profile"] == v.profile["id"], "profile differs from the version's")
        if p.task == ifeval.TASK:
            require(rec["harness"]["packages"] == ifeval.prepare(),
                    "IFEval session has a different set of scorer packages")
        seed, ids = rec["seed"], rec["items"]
        require(seed in p.seeds, f"seed {seed} is not one of the protocol's seeds {p.seeds}")
        require(ids and ids == sorted(set(ids)) and set(ids) <= set(p.ids),
                "items must be sorted, unique protocol items")
        outcomes = {}
        for variant in VARIANTS:
            check_weights(v.profile, rec[variant]["model_files"], variant)
            outcomes[variant] = derive_variant(
                p, s, ids, seed, read_jsonl(d / f"{variant}.samples.jsonl"),
                read_jsonl(d / f"{variant}.requests.jsonl"))
        require(non_weights(rec["base"]["model_files"]) == non_weights(rec["patched"]["model_files"]),
                "base and patched model folders differ outside the weights files")
    except (BenchError, KeyError, TypeError) as e:
        raise BenchError(f"{rel(d)}: {e if isinstance(e, BenchError) else f'malformed: {e!r}'}") from None
    return Session(d.name, rec, outcomes)
