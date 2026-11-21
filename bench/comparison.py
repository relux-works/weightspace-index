"""Paired comparisons and discrepancies between task scorers."""

from __future__ import annotations

import math

from . import ifeval
from .core import VARIANTS, Protocol, Settings, Version, canonical, read_jsonl
from .grading import MATH_TASK, boxed
from .runs import (collect, cut_at_limit, finished_without_answer, headline, is_cut,
                   is_equiv_score, item_means, lm_eval_score, mean_completion_tokens, mean_score)
from .sessions import sample_messages


def compare(v: Version, p: Protocol, s: Settings) -> dict:
    """Paired base-versus-patched comparison over the filed pairs.

    The unit of analysis is the item: per item, the patched-minus-base
    score averaged over its seeds. IFEval reports the observed difference
    for the prompts filed, which need not be randomly sampled."""
    c = collect(v, p, s)
    pairs = c.pairs

    def difference(value) -> dict:
        b, q = item_means(p, pairs, 1, value), item_means(p, pairs, 2, value)
        diffs = [q[i] - b[i] for i in b]
        n = len(diffs)
        mean = sum(diffs) / n
        if p.task == ifeval.TASK:
            return {"difference": mean}
        se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1) / n) if n > 1 else 0.0
        return {"difference": mean, "standard_error": se,
                "interval_95": [mean - 1.96 * se, mean + 1.96 * se]}

    def stats(side: int) -> dict:
        outs = [pair[side] for pair in pairs.values()]
        scores = {
            "score": mean_score(p, pairs, side, headline),
            "score_lm_eval": mean_score(p, pairs, side, lm_eval_score),
            "cut_at_limit": cut_at_limit(pairs, side),
            "finished_without_answer": finished_without_answer(pairs, side),
            "answered_then_cut": sum(is_cut(o) and o["answer"] is not None for o in outs),
            "right_only_by_lm_eval": sum(o["score"] > 0 and not o["correct"] for o in outs),
            "right_only_by_final_answer": sum(o["correct"] and o["score"] == 0 for o in outs),
            "mean_completion_tokens": mean_completion_tokens(pairs, side),
        }
        if p.task == MATH_TASK:
            scores.update(score_is_equiv=mean_score(p, pairs, side, is_equiv_score),
                          checkers_disagree=sum(o["correct"] != o["correct_is_equiv"] for o in outs))
        return scores

    cells = list(pairs.values())
    right = [(headline(x[1]), headline(x[2])) for x in cells]
    return {
        "metric": p.metric, "complete": c.complete(p),
        "items": len(item_means(p, pairs, 1)), "pairs": len(pairs), "protocol_pairs": len(p.cells()),
        "base": stats(1), "patched": stats(2),
        **difference(headline),
        "lm_eval": difference(lm_eval_score),
        "pairs_both_right": sum(b == 1 and q == 1 for b, q in right),
        "pairs_only_base_right": sum(b > q for b, q in right),
        "pairs_only_patched_right": sum(q > b for b, q in right),
        "pairs_identical_replies": sum(x[1]["content_sha256"] == x[2]["content_sha256"] for x in cells),
        "repeats": len(c.repeats),
        "repeats_identical_to_first": sum(
            pairs[cell][1]["content_sha256"] == rb["content_sha256"]
            and pairs[cell][2]["content_sha256"] == rq["content_sha256"]
            for cell, _, rb, rq in c.repeats),
    }


# ---------------------------------------------------------------- disagreements


def disagreements(v: Version, p: Protocol, s: Settings) -> list[dict]:
    """Show the evidence for filed replies that the graders scored differently."""
    c = collect(v, p, s)
    results = v.results_dir(p, s)
    math_task = p.task == MATH_TASK
    out = []
    for side, variant in enumerate(VARIANTS, start=1):
        by_session: dict[str, list] = {}
        for (i, seed), pair in sorted(c.pairs.items()):
            o = pair[side]
            if (o["score"] > 0) != o["correct"] or (math_task and o["correct"] != o["correct_is_equiv"]):
                by_session.setdefault(pair[0], []).append((i, seed, o))
        for name, cells in sorted(by_session.items()):
            d = results / name
            samples = {x["doc_id"]: x for x in read_jsonl(d / f"{variant}.samples.jsonl")}
            # `collect` already proved each sample matches exactly one exchange.
            replies = {canonical(x["request"]["messages"]): x["response"]["choices"][0]["message"]["content"]
                       for x in read_jsonl(d / f"{variant}.requests.jsonl")}
            for i, seed, o in cells:
                sample = samples[i]
                content = replies[canonical(sample_messages(sample))]
                scored = sample["resps"][0][0]
                row = {
                    "variant": variant, "item": i, "seed": seed, "session": name,
                    "answer": o["answer"], "lm_eval": o["score"] > 0,
                    "finish_reason": o["finish_reason"],
                    "completion_tokens": o["completion_tokens"],
                    "thinking_finished": bool(s.think_end) and s.think_end in content,
                    "excerpt": scored[-400:],
                }
                if math_task:
                    found = boxed(scored)
                    last = found[-1] if found else None
                    row.update(gold=sample["target"], math_verify=o["correct"],
                               is_equiv=o["correct_is_equiv"], lm_eval_text_boxes=len(found),
                               lm_eval_text_last_box=last[2] if last else None)
                    if last:
                        row["excerpt"] = scored[max(0, last[0] - 240):last[1] + 160]
                else:
                    row["final_answer_correct"] = o["correct"]
                out.append(row)
    return out
