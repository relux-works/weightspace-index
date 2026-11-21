"""Collect paired sessions and build the index's run records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import core, ifeval
from .core import (SESSION_NAME, VARIANTS, Protocol, Settings, Version, canonical, dumps,
                   gen_kwargs, read_json, rel, require, sha256_bytes)
from .sessions import Session, derive_session


@dataclass
class Collected:
    sessions: list[Session]
    # (item, seed) -> (session name, base outcome, patched outcome); the
    # first session in name order (UTC start time) that ran a cell counts.
    pairs: dict[tuple[int, int], tuple[str, dict, dict]] = field(default_factory=dict)
    # Later sessions that ran an already-counted cell: (cell, session, base, patched).
    repeats: list[tuple[tuple[int, int], str, dict, dict]] = field(default_factory=list)

    def complete(self, p: Protocol) -> bool:
        return set(self.pairs) == set(p.cells())


def session_dirs(results: Path) -> list[Path]:
    out = []
    for d in sorted(results.iterdir()) if results.is_dir() else []:
        require(d.is_dir() and SESSION_NAME.match(d.name),
                f"{rel(d)}: unexpected path (sessions are <UTC>-<hex> folders)")
        out.append(d)
    return out


def collect(v: Version, p: Protocol, s: Settings) -> Collected:
    results = v.results_dir(p, s)
    dirs = session_dirs(results)
    require(dirs, f"{rel(results)}: no sessions filed")
    c = Collected([derive_session(v, p, s, d) for d in dirs])
    first = c.sessions[0].record
    for sess in c.sessions:
        rec = sess.record
        require(rec["harness"] == first["harness"],
                f"{rel(results / sess.name)}: harness differs from {c.sessions[0].name}")
        for variant in VARIANTS:
            for key in ("model_files", "served_model"):
                require(rec[variant][key] == first[variant][key],
                        f"{rel(results / sess.name)}: {variant} {key} differs from {c.sessions[0].name}")
        for i in rec["items"]:
            cell = (i, rec["seed"])
            b, q = sess.outcomes["base"][i], sess.outcomes["patched"][i]
            if cell in c.pairs:
                c.repeats.append((cell, sess.name, b, q))
            else:
                c.pairs[cell] = (sess.name, b, q)
    return c


def item_means(p: Protocol, pairs: dict, side: int, value=lambda o: o["score"]) -> dict[int, float]:
    """Per item, the mean over its filed seeds."""
    acc: dict[int, list[float]] = {}
    for (i, _seed), pair in pairs.items():
        acc.setdefault(i, []).append(value(pair[side]))
    return {i: sum(xs) / len(xs) for i, xs in sorted(acc.items())}


def is_cut(outcome: dict) -> bool:
    """A reply cut at the token limit, as the server reported it."""
    return outcome["finish_reason"] == "length"


def headline(outcome: dict) -> float:
    """One reply's headline value under the protocol's final-answer grader."""
    return float(outcome["correct"])


def lm_eval_score(outcome: dict) -> float:
    """One reply's score as lm-eval gave it. lm-eval scores the whole
    reply when the think-end token is missing, and math_verify then falls
    back to any number it finds there."""
    return outcome["score"]


def is_equiv_score(outcome: dict) -> float:
    """One reply's value under the original MATH checker, on the same
    final answer."""
    return float(outcome["correct_is_equiv"])


def mean_score(p: Protocol, pairs: dict, side: int, value) -> float:
    """Mean over the filed items of each item's mean over its seeds."""
    means = item_means(p, pairs, side, value)
    return sum(means.values()) / len(means)


def cut_at_limit(pairs: dict, side: int) -> float:
    """Share of one side's replies cut at the token limit, over every
    filed (item, seed) pair. The one definition: `compare` reports it
    and a completed run records it beside its score."""
    outs = [pair[side] for pair in pairs.values()]
    return sum(map(is_cut, outs)) / len(outs)


def finished_without_answer(pairs: dict, side: int) -> float:
    """Share of one side's replies that stopped on their own but give no
    final answer, over every filed (item, seed) pair."""
    outs = [pair[side] for pair in pairs.values()]
    return sum(not is_cut(o) and o["answer"] is None for o in outs) / len(outs)


def mean_completion_tokens(pairs: dict, side: int) -> float:
    outs = [pair[side] for pair in pairs.values()]
    return sum(o["completion_tokens"] for o in outs) / len(outs)


HEADLINE_UNIT = ("mean over the protocol's frozen items of the mean over its seeds; a reply "
                 "scores 1 when its final answer, the last \\boxed{} after the think-end token, "
                 "equals the gold answer under math-verify, else 0")


def build_runs(v: Version, p: Protocol, s: Settings) -> dict[str, tuple[str, str]]:
    """{variant: (run.json text, outcomes.jsonl text)} from the filed sessions."""
    c = collect(v, p, s)
    results = v.results_dir(p, s)
    dirs = v.run_dirs(p, s)
    complete = c.complete(p)
    first = c.sessions[0].record
    out = {}
    for side, variant in enumerate(VARIANTS, start=1):
        other = VARIANTS[2 - side]
        lines = "".join(
            canonical({"id": i, "seed": seed, "session": pair[0], **pair[side]}) + "\n"
            for (i, seed), pair in sorted(c.pairs.items()))
        if complete:
            score = {"metric": p.metric,
                     "value": mean_score(p, c.pairs, side, headline),
                     "value_lm_eval": mean_score(p, c.pairs, side, lm_eval_score),
                     "cut_at_limit": cut_at_limit(c.pairs, side),
                     "finished_without_answer": finished_without_answer(c.pairs, side),
                     "mean_completion_tokens": mean_completion_tokens(c.pairs, side),
                     "unit": ifeval.HEADLINE_UNIT if p.task == ifeval.TASK else HEADLINE_UNIT,
                     "denominator": len(item_means(p, c.pairs, side))}
        else:
            score = {"state": "UNKNOWN", "owner": "bench contributors",
                     "reason": f"{len(c.pairs)} of {len(p.cells())} item-seed pairs filed"}
        run = {
            "contract_version": 1,
            "profile": v.profile["id"],
            "protocol": p.named(),
            "dataset": p.dataset_named(),
            "settings": {
                "file": rel(s.path),
                "sha256": s.sha256,
                "gen_kwargs": gen_kwargs(p, s),
                "model_args": s.data["model_args"],
                "num_fewshot": p.data["num_fewshot"],
                "seeds": p.seeds,
                "harness": first["harness"],
                "served_model": first[variant]["served_model"],
                "model_files": first[variant]["model_files"],
                "paired_with": rel(dirs[other]),
                "sessions": {sess.name: {k: sess.record[k] for k in
                                         ("seed", "hardware", "concurrency")}
                             | {"items": len(sess.record["items"]),
                                "parallel": sess.record.get("parallel", False),
                                "server": sess.record[variant]["server"]}
                             for sess in c.sessions},
                "results": rel(results),
            },
            "state": "COMPLETE" if complete else "INCOMPLETE",
            "score": score,
            "outcomes": {"sha256": sha256_bytes(lines.encode()), "size_bytes": len(lines.encode()),
                         "locators": [{"kind": "index-path", "path": "outcomes.jsonl"}]},
        }
        if p.task == ifeval.TASK:
            run["settings"]["ifeval_grader"] = ifeval.GRADER
        out[variant] = (dumps(run), lines)
    return out


def all_results() -> list[tuple[Version, Protocol, Settings]]:
    """Every run folder under bench/results/<version>/<protocol>.<settings>.<profile>/."""
    out = []
    for d in sorted({x.parent.parent for x in core.RESULTS.glob("**/session.json")}):
        # Settings ids may contain dots; protocol and profile ids do not.
        head, _, profile = d.name.rpartition(".")
        protocol, _, settings = head.partition(".")
        v = Version.resolve(str(core.MODELS / d.parent.relative_to(core.RESULTS)))
        require(v.profile["id"] == profile, f"{rel(d)}: profile differs from the version's")
        out.append((v, Protocol.load(protocol), Settings.load(settings)))
    return out


def coverage(v: Version, p: Protocol, s: Settings) -> dict[tuple[int, int], int]:
    """How many filed sessions ran each protocol cell (from session.json only)."""
    counts = {cell: 0 for cell in p.cells()}
    results = v.results_dir(p, s)
    for d in session_dirs(results):
        rec = read_json(d / "session.json")
        for i in rec.get("items", []):
            if (i, rec.get("seed")) in counts:
                counts[(i, rec["seed"])] += 1
    return counts
