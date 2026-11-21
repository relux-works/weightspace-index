"""python -m bench <command>; see bench/README.md."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

from . import core, ifeval
from .comparison import compare, disagreements
from .core import (BenchError, Protocol, Settings, Version, check_settings_fit, dumps,
                   read_json, read_jsonl, rel, require, sha256_file)
from .harness import (check_results, harness_packages, lm_eval_argv, lm_eval_outputs,
                      run_logged, task_arg, work_dir)
from .runner import check_server_args, default_vllm, run
from .runs import all_results, build_runs, coverage


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


def cmd_freeze(a) -> None:
    """Fill a draft protocol's items with lm-eval's own doc/target hashes."""
    p = Protocol.load(a.protocol)
    require(not p.data["items"], f"{rel(p.path)} is already frozen")
    sel = p.data["selection"]
    if sel.get("all") is True:
        ids, expected = None, list(range(sel["count"]))
    else:
        sel_path = core.BENCH / sel["file"]
        require(sha256_file(sel_path)[0] == sel["sha256"], f"{sel['file']} differs from its pinned sha256")
        ids = expected = sorted(read_json(sel_path)["indices"])
    harness_packages(p)
    work = work_dir("freeze", f"{p.id}-{time.time_ns()}")
    task = task_arg(p, None, work)
    run_logged(lm_eval_argv(p, None, ids, None, None, None, work / "out", task=task),
               work / "lm_eval.log", "lm_eval")
    samples, results = lm_eval_outputs(work / "out", p.task)
    check_results(p, results, "dummy")
    hashes = {x["doc_id"]: [x["doc_hash"], x["target_hash"]] for x in read_jsonl(samples)}
    require(sorted(hashes) == expected, "lm_eval returned other items than the selection")
    p.data["items"] = [[i, *hashes[i]] for i in expected]
    text = dumps(p.data)
    # One item per line, and the seeds on one line, keep diffs readable.
    text = re.sub(r'\[\n\s+(\d+),\n\s+("[0-9a-f]{64}"),\n\s+("[0-9a-f]{64}")\n\s+\]',
                  r"[\1, \2, \3]", text)
    text = re.sub(r'"seeds": \[[^\]]*\]', lambda m: '"seeds": ' + json.dumps(p.seeds), text)
    p.path.write_text(text)
    print(f"froze {len(expected)} items into {rel(p.path)}")


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
    run(a, v, p, s, seed, ids, how)


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
    b, q, lm = c["base"], c["patched"], c["lm_eval"]
    title = "strict prompt-level accuracy" if p.task == ifeval.TASK else "final answers right"
    print(f"\n{title} over {c['items']} items ({state}): "
          f"base {b['score']:.3f}, patched {q['score']:.3f}, difference {c['difference']:+.3f}{interval(c)}")
    print(f"cut at the token limit: base {b['cut_at_limit']:.3f}, patched {q['cut_at_limit']:.3f}; "
          f"finished without an answer: base {b['finished_without_answer']:.3f}, "
          f"patched {q['finished_without_answer']:.3f}; mean reply tokens: "
          f"base {b['mean_completion_tokens']:.0f}, patched {q['mean_completion_tokens']:.0f}")
    print(f"lm-eval's own {p.metric}: base {b['score_lm_eval']:.3f}, "
          f"patched {q['score_lm_eval']:.3f}, difference {lm['difference']:+.3f}{interval(lm)}; "
          f"right only by lm-eval: base {b['right_only_by_lm_eval']}, "
          f"patched {q['right_only_by_lm_eval']}")
    if p.task == ifeval.TASK:
        return
    print(f"original MATH checker on the same answers: base {b['score_is_equiv']:.3f}, "
          f"patched {q['score_is_equiv']:.3f}; disagrees with math-verify: "
          f"base {b['checkers_disagree']}, patched {q['checkers_disagree']}")


def interval(stats: dict) -> str:
    if "interval_95" not in stats:
        return ""
    low, high = stats["interval_95"]
    return f" (95% {low:+.3f} .. {high:+.3f})"


def cmd_disagreements(a) -> None:
    v, p, s = load(a)
    found = disagreements(v, p, s)
    for x in found:
        print(json.dumps(x, ensure_ascii=False))
    if p.task == ifeval.TASK:
        print(f"\n{len(found)} replies where lm-eval and the visible reply grade disagree", file=sys.stderr)
        return
    lm_only = [x for x in found if x["lm_eval"] and not x["math_verify"]]
    print(f"\n{len(found)} replies where the graders disagree: "
          f"{len(lm_only)} right only by lm-eval ({sum(x['finish_reason'] == 'length' for x in lm_only)} "
          f"of them cut at the token limit, {sum(x['answer'] is None for x in lm_only)} with no final answer), "
          f"{sum(x['math_verify'] and not x['lm_eval'] for x in found)} right only by the final answer, "
          f"{sum(x['math_verify'] != x['is_equiv'] for x in found)} where math-verify and "
          f"the original MATH checker disagree", file=sys.stderr)


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
    c.add_argument("--vllm", default=default_vllm(),
                   help="vllm executable (default: the one beside this Python, else PATH's)")
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

    c = sub.add_parser("disagreements", help="replies that the protocol's graders scored differently")
    entry_args(c)
    c.set_defaults(fn=cmd_disagreements)

    a = parser.parse_args(argv)
    try:
        a.fn(a)
    except (BenchError, OSError) as e:
        print(f"bench: {e}", file=sys.stderr)
        return 1
    return 0
