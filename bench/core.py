"""Protocols, settings, paired sessions, and the run records built from them.

A session is one `python -m bench run`: the base model and the patched copy
answer the same items under the same sampling seed, one after the other.
Everything here is a pure function of files in the repository, so any
checkout rebuilds the same run.json and outcomes.jsonl bytes from the filed
sessions (`python -m bench check`).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench"
MODELS = REPO / "models"
RESULTS = BENCH / "results"
HARNESS_MODEL = "local-chat-completions"
VARIANTS = ("base", "patched")
SESSION_FILES = sorted(["session.json"] + [f"{v}.{kind}.jsonl" for v in VARIANTS
                                           for kind in ("samples", "requests")])
SESSION_NAME = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
WEIGHTS_SUFFIXES = (".safetensors", ".gguf")


class BenchError(ValueError):
    pass


def require(ok: object, message: str) -> None:
    if not ok:
        raise BenchError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BenchError(f"{rel(path)}: {e}") from None


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError) as e:
        raise BenchError(f"{rel(path)}: {e}") from None


def dumps(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- inputs


@dataclass
class Protocol:
    path: Path
    sha256: str
    data: dict

    @classmethod
    def load(cls, name: str) -> "Protocol":
        path = BENCH / "protocols" / f"{name}.json"
        data = read_json(path)
        require(data.get("id") == name and "." not in name, f"{rel(path)}: id must be {name}, no dots")
        for key in ("harness", "task", "task_version", "num_fewshot", "gen_kwargs",
                    "primary_metric", "dataset", "selection", "seeds", "items"):
            require(key in data, f"{rel(path)}: missing {key}")
        require(data["apply_chat_template"] is True, f"{rel(path)}: chat template must be on")
        seeds = data["seeds"]
        require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds)
                and all(isinstance(x, int) and not isinstance(x, bool) for x in seeds),
                f"{rel(path)}: seeds must be a non-empty list of distinct integers")
        ids = [item[0] for item in data["items"]]
        require(ids == sorted(set(ids)), f"{rel(path)}: items must be sorted, unique")
        # The digest leaves out `seeds`, so seeds can be appended later
        # without changing what an already-filed pair was run under.
        body = {k: v for k, v in data.items() if k != "seeds"}
        return cls(path, sha256_bytes(canonical(body).encode()), data)

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def task(self) -> str:
        return self.data["task"]

    @property
    def metric(self) -> str:
        return self.data["primary_metric"]

    @property
    def seeds(self) -> list[int]:
        return self.data["seeds"]

    @property
    def ids(self) -> list[int]:
        return [item[0] for item in self.data["items"]]

    @property
    def items(self) -> dict[int, tuple[str, str]]:
        return {i: (doc, target) for i, doc, target in self.data["items"]}

    def cells(self) -> list[tuple[int, int]]:
        """Every (item, seed) pair the protocol asks for."""
        return [(i, seed) for seed in self.seeds for i in self.ids]

    def named(self) -> dict:
        return {"name": rel(self.path), "sha256": self.sha256}

    def dataset_named(self) -> dict:
        d = self.data["dataset"]
        lines = "".join(f"{i} {doc} {target}\n" for i, doc, target in self.data["items"])
        return {"name": f"{d['repo_id']}@{d['revision']}:{d['split']}",
                "sha256": sha256_bytes(lines.encode())}


@dataclass
class Settings:
    path: Path
    sha256: str
    data: dict

    @classmethod
    def load(cls, name: str) -> "Settings":
        path = BENCH / "settings" / f"{name}.json"
        data = read_json(path)
        require(data.get("id") == name, f"{rel(path)}: id must be {name}")
        g = data.get("gen_kwargs", {})
        for key in ("max_gen_toks", "temperature", "do_sample"):
            require(key in g, f"{rel(path)}: gen_kwargs.{key} must be explicit")
        require(isinstance(data.get("model_args"), dict), f"{rel(path)}: missing model_args")
        require("seed" not in data["model_args"], f"{rel(path)}: the seed comes from the protocol")
        require(isinstance(data.get("min_max_model_len"), int), f"{rel(path)}: missing min_max_model_len")
        for task, o in data.get("task_overrides", {}).items():
            require(isinstance(o, dict) and isinstance(o.get("doc_to_text"), str)
                    and set(o) <= {"doc_to_text", "note"},
                    f"{rel(path)}: task_overrides.{task} may set only doc_to_text (and a note)")
        return cls(path, sha256_file(path)[0], data)

    def doc_to_text(self, task: str) -> str | None:
        """The card's prompt for `task`, if the card gives one; else the
        stock task's prompt is used."""
        return self.data.get("task_overrides", {}).get(task, {}).get("doc_to_text")

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def think_end(self) -> str | None:
        return self.data["model_args"].get("think_end_token")


def gen_kwargs(p: Protocol, s: Settings) -> dict:
    """What lm-eval gets on --gen_kwargs: the card's settings plus the
    protocol's task overrides. A key set by both is an error."""
    both = set(p.data["gen_kwargs"]) & set(s.data["gen_kwargs"])
    require(not both, f"{', '.join(sorted(both))} set by both protocol and settings")
    return {**s.data["gen_kwargs"], **p.data["gen_kwargs"]}


def expected_payload(p: Protocol, s: Settings, served: str, seed: int) -> dict:
    """Every key lm-eval's local-chat-completions sends besides messages,
    derived from the same gen_kwargs and seed (lm-eval 0.4.13
    LocalChatCompletion._create_payload)."""
    g = dict(gen_kwargs(p, s))
    g.pop("do_sample", None)
    until = g.pop("until", [])
    until = [until] if isinstance(until, str) else list(until)
    return {"model": served, "max_tokens": g.pop("max_gen_toks"),
            "temperature": g.pop("temperature"), "stop": until[:4], "seed": seed, **g}


# ---------------------------------------------------------------- entries


@dataclass
class Version:
    """A patch version folder and the base profile it targets."""
    folder: Path         # relative to models/, e.g. Qwen/Qwen3.5-0.8B/<slug>/v1
    base_folder: Path    # relative to models/
    profile: dict
    record: dict         # version.json
    kind: str            # patch.json kind

    @classmethod
    def resolve(cls, path: str) -> "Version":
        abs_path = Path(path) if Path(path).is_absolute() else REPO / path
        abs_path = abs_path.resolve()
        try:
            folder = abs_path.relative_to(MODELS.resolve())
        except ValueError:
            raise BenchError(f"{path}: not a folder under models/") from None
        require((abs_path / "version.json").is_file(),
                f"{path}: not a patch version folder (models/<ns>/<model>/<patch>/v<N>)")
        record = read_json(abs_path / "version.json")
        kind = read_json(abs_path.parent / "patch.json")["kind"]
        base_folder = folder.parent.parent
        target = record["target_profile"]
        found = [x for x in read_json(MODELS / base_folder / "base.json")["profiles"]
                 if x["id"] == target]
        require(found, f"{path}: base has no profile {target}")
        require("." not in target, "profile ids with dots are not supported")
        return cls(folder, base_folder, found[0], record, kind)

    @property
    def slug(self) -> str:
        return f"{self.folder.parent.name}-{self.folder.name}"

    def run_ids(self, p: Protocol, s: Settings) -> dict[str, str]:
        patched = f"{p.id}.{s.id}.{self.profile['id']}"
        return {"patched": patched, "base": f"{patched}.{self.slug}"}

    def run_dirs(self, p: Protocol, s: Settings) -> dict[str, Path]:
        ids = self.run_ids(p, s)
        return {"base": MODELS / self.base_folder / "runs" / ids["base"],
                "patched": MODELS / self.folder / "runs" / ids["patched"]}

    def results_dir(self, p: Protocol, s: Settings) -> Path:
        return RESULTS / self.folder / self.run_ids(p, s)["patched"]

    def source_file(self) -> Path:
        """The version's hosted source file, the input to `weightspace apply`."""
        require(self.kind == "replacement",
                f"{self.folder}: bench builds only replacement patches so far (kind {self.kind})")
        found = [loc["path"] for f in self.record["files"] if f.get("role") == "source"
                 for loc in f.get("locators", []) if loc.get("kind") == "index-path"]
        require(len(found) == 1, f"{self.folder}: expected one hosted source file")
        return MODELS / self.folder / found[0]


def check_settings_fit(v: Version, s: Settings) -> None:
    card = s.data["source"]["repo_id"]
    base = v.profile["source"]["repo_id"]
    require(card == base, f"{rel(s.path)} is the card of {card}, not of {base}")


def check_weights(profile: dict, files: dict[str, str], variant: str) -> None:
    """Base weights must be the profile's exact files; patched weights must
    be the same set of paths with every file changed."""
    want = {f["path"]: f["sha256"] for f in profile["files"]}
    weights = {k for k in files if k.endswith(WEIGHTS_SUFFIXES)}
    require(weights == set(want), f"{variant} model folder weights {sorted(weights)} are not "
                                  f"the profile's {sorted(want)}")
    for path, digest in want.items():
        if variant == "base":
            require(files[path] == digest, f"{path} is not the base profile's file")
        else:
            require(files[path] != digest, f"{path} in the patched folder is the unpatched base file")


def non_weights(files: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in files.items() if not k.endswith(WEIGHTS_SUFFIXES)}


# ---------------------------------------------------------------- sessions


def lm_eval_answer(content: str, think_end: str | None) -> str:
    """lm-eval 0.4.13 postprocess_generated_text with stop=None."""
    return content.split(think_end)[-1].lstrip() if think_end else content


# ---------------------------------------------------------------- grading

# Tasks whose replies end in a \boxed{} final answer, graded by `grade`.
GRADED_TASKS = ("minerva_math500",)
BOXED = "\\boxed{"


def pinned_packages() -> dict[str, str]:
    """Installed versions of every pin in bench/requirements.txt; any
    difference is an error, since another scorer version can score the
    same reply differently."""
    out = {}
    for line in (BENCH / "requirements.txt").read_text().splitlines():
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


def boxed(text: str) -> list[tuple[int, int, str]]:
    """Every \\boxed{...} in `text`, braces balanced: (start, end, inside)."""
    out, start = [], 0
    while (j := text.find(BOXED, start)) >= 0:
        k, depth = j + len(BOXED), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        if depth == 0:
            out.append((j, k, text[j + len(BOXED):k - 1]))
        start = j + len(BOXED)
    return out


def final_answer(content: str, think_end: str | None) -> str | None:
    """A reply's final answer: the inside of its last \\boxed{...} after
    its last think-end token. None, which counts as wrong, when the reply
    never closed its thinking, when no box follows, or when the last box
    is unclosed or empty. Nothing else in the text is read as an answer
    (owner decision: Timur, 2026-11-20; docs/benchmarking.md)."""
    if think_end:
        _, found, content = content.rpartition(think_end)
        if not found:
            return None
    start = content.rfind(BOXED)
    last = boxed(content[start:]) if start >= 0 else []
    return (last[0][2].strip() or None) if last else None


class Timeouts(logging.Handler):
    """Counts math-verify's timeout warnings. A timed-out parse or
    comparison comes back as 'not equal', which would make a grade depend
    on the machine's speed; `grade` refuses it instead."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += record.getMessage().startswith("Timeout during")


TIMEOUTS = Timeouts()


@functools.cache
def grader():
    """math-verify, and lm-eval's copy of the original MATH checker
    (Hendrycks et al.'s is_equiv), at the versions bench/requirements.txt
    pins. Imported on first use, so commands that grade nothing run
    without them."""
    pinned_packages()
    import math_verify  # noqa: PLC0415
    from lm_eval.tasks.hendrycks_math.utils import is_equiv  # noqa: PLC0415
    log = logging.getLogger("math_verify")
    log.setLevel(logging.WARNING)
    log.addHandler(TIMEOUTS)
    return math_verify, is_equiv


def grade(answer: str | None, gold: str) -> tuple[bool, bool]:
    """(math-verify, original MATH checker) verdicts on one final answer.
    math-verify compares the two boxed answers and nothing else,
    verify(parse('\\boxed{gold}'), parse('\\boxed{answer}')), with the
    library's defaults; the MATH checker is is_equiv(answer, gold)."""
    if answer is None:
        return False, False
    mv, is_equiv = grader()
    before = TIMEOUTS.count
    g, a = mv.parse(f"\\boxed{{{gold}}}"), mv.parse(f"\\boxed{{{answer}}}")
    right = bool(g and a and mv.verify(g, a))
    require(TIMEOUTS.count == before,
            f"math-verify timed out grading {answer[:80]!r} against {gold!r}; "
            "a timed-out grade would depend on the machine's speed")
    require(g, f"gold answer {gold!r} does not parse")
    return right, bool(is_equiv(answer, gold))


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
    reply's final answer is graded against the sample's target, which must
    be the frozen gold answer."""
    require(p.task in GRADED_TASKS, f"bench grades only {', '.join(GRADED_TASKS)} so far ({p.task})")
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
        score = sample.get(p.metric)
        require(isinstance(score, (int, float)) and not isinstance(score, bool),
                f"item {i}: no numeric {p.metric}")
        scored = sample["resps"][0][0]
        matches = by_messages.get(canonical(sample_messages(sample)), [])
        require(len(matches) == 1, f"item {i}: {len(matches)} recorded exchanges match its prompt")
        resp = matches[0]["response"]
        require(len(resp["choices"]) == 1, f"item {i}: expected one choice")
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        require(isinstance(content, str) and isinstance(scored, str)
                and lm_eval_answer(content, s.think_end) == scored,
                f"item {i}: scored text is not the recorded reply")
        gold = sample["target"]
        require(isinstance(gold, str) and sha256_bytes(gold.encode()) == sample["target_hash"],
                f"item {i}: target is not the frozen gold answer")
        answer = final_answer(content, s.think_end)
        correct, correct_is_equiv = grade(answer, gold)
        out[i] = {"score": float(score), "answer": answer, "correct": correct,
                  "correct_is_equiv": correct_is_equiv, "finish_reason": choice["finish_reason"],
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


# ---------------------------------------------------------------- runs


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
    """One reply's headline value: 1 if its final answer is right under
    math-verify, else 0 (see `final_answer` and `grade`; owner decision:
    Timur, 2026-11-20; docs/benchmarking.md)."""
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
                     "unit": HEADLINE_UNIT,
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
        out[variant] = (dumps(run), lines)
    return out


def all_results() -> list[tuple[Version, Protocol, Settings]]:
    """Every run folder under bench/results/<version>/<protocol>.<settings>.<profile>/."""
    out = []
    for d in sorted({x.parent.parent for x in RESULTS.glob("**/session.json")}):
        # Settings ids may contain dots; protocol and profile ids do not.
        head, _, profile = d.name.rpartition(".")
        protocol, _, settings = head.partition(".")
        v = Version.resolve(str(MODELS / d.parent.relative_to(RESULTS)))
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


# ---------------------------------------------------------------- comparison


def compare(v: Version, p: Protocol, s: Settings) -> dict:
    """Paired base-versus-patched comparison over the filed pairs.

    The unit of analysis is the item: per item, the patched-minus-base
    score averaged over its seeds; the standard error is over items. The
    headline grades each reply's final answer; the same figures under
    lm-eval's own score sit under `lm_eval`."""
    c = collect(v, p, s)
    pairs = c.pairs

    def difference(value) -> dict:
        b, q = item_means(p, pairs, 1, value), item_means(p, pairs, 2, value)
        diffs = [q[i] - b[i] for i in b]
        n = len(diffs)
        mean = sum(diffs) / n
        se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1) / n) if n > 1 else 0.0
        return {"difference": mean, "standard_error": se,
                "interval_95": [mean - 1.96 * se, mean + 1.96 * se]}

    def stats(side: int) -> dict:
        outs = [pair[side] for pair in pairs.values()]
        return {
            "score": mean_score(p, pairs, side, headline),
            "score_lm_eval": mean_score(p, pairs, side, lm_eval_score),
            "score_is_equiv": mean_score(p, pairs, side, is_equiv_score),
            "cut_at_limit": cut_at_limit(pairs, side),
            "finished_without_answer": finished_without_answer(pairs, side),
            "answered_then_cut": sum(is_cut(o) and o["answer"] is not None for o in outs),
            "right_only_by_lm_eval": sum(o["score"] > 0 and not o["correct"] for o in outs),
            "right_only_by_final_answer": sum(o["correct"] and o["score"] == 0 for o in outs),
            "checkers_disagree": sum(o["correct"] != o["correct_is_equiv"] for o in outs),
            "mean_completion_tokens": mean_completion_tokens(pairs, side),
        }

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
    """Every filed reply on which lm-eval's score, math-verify on the
    final answer, and the original MATH checker do not all agree, with
    the final answer, the gold answer, and the boxes in the text lm-eval
    scored (the whole reply when the think-end token is missing)."""
    c = collect(v, p, s)
    results = v.results_dir(p, s)
    out = []
    for side, variant in enumerate(VARIANTS, start=1):
        by_session: dict[str, list] = {}
        for (i, seed), pair in sorted(c.pairs.items()):
            o = pair[side]
            if len({o["score"] > 0, o["correct"], o["correct_is_equiv"]}) > 1:
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
                found = boxed(scored)
                last = found[-1] if found else None
                out.append({
                    "variant": variant, "item": i, "seed": seed, "session": name,
                    "gold": sample["target"], "answer": o["answer"],
                    "lm_eval": o["score"] > 0, "math_verify": o["correct"],
                    "is_equiv": o["correct_is_equiv"],
                    "finish_reason": o["finish_reason"],
                    "completion_tokens": o["completion_tokens"],
                    "thinking_finished": bool(s.think_end) and s.think_end in content,
                    "lm_eval_text_boxes": len(found),
                    "lm_eval_text_last_box": last[2] if last else None,
                    "excerpt": scored[max(0, last[0] - 240):last[1] + 160] if last else scored[-400:],
                })
    return out
