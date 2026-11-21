"""Repository paths, protocol and settings inputs, and model weight identities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
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


def pinned_packages(*extra_files: str) -> dict[str, str]:
    """Verify the scorer's common and task-specific package pins."""
    out = {}
    for file in ("requirements.txt", *extra_files):
        for line in (BENCH / file).read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            name, _, version = line.partition("==")
            name = re.sub(r"\[.*\]", "", name).strip()
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                raise BenchError(f"{name} is not installed; pip install -r bench/{file}") from None
            require(installed == version, f"{name} {installed} installed, bench/{file} pins {version}")
            out[name] = installed
    return out
