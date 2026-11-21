"""Score IFEval replies with the pinned lm-eval instruction checkers."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from .core import BenchError, pinned_packages, require, sha256_bytes, sha256_file

TASK = "ifeval"
METRIC = "prompt_level_strict_acc"
HEADLINE_UNIT = ("mean over the protocol's frozen prompts of the mean over its seeds, "
                 "with a score of 1 when the visible reply passes every strict instruction check")
PUNKT_ENGLISH = {
    "abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
    "sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
}


@functools.cache
def prepare() -> dict[str, str]:
    """Use the same dependencies and language detector in lm-eval and replay."""
    packages = pinned_packages("requirements-ifeval.txt")
    import langdetect
    import nltk

    try:
        english = Path(str(nltk.data.find("tokenizers/punkt_tab/english/")))
        observed = {name: sha256_file(english / name)[0] for name in PUNKT_ENGLISH}
    except (LookupError, OSError):
        observed = {}
    require(observed == PUNKT_ENGLISH,
            "IFEval needs the pinned NLTK punkt_tab tables; see bench/README.md")
    langdetect.DetectorFactory.seed = 0
    return packages


def grade_sample(sample: dict, messages: list, choice: dict, scored: str, think_end: str | None) -> dict:
    """Verify the harness verdict and grade the text that reached the user."""
    doc = sample["doc"]
    doc_hash = sha256_bytes(json.dumps(doc, indent=2, ensure_ascii=False).encode())
    require(doc_hash == sample["doc_hash"], "logged document differs from the frozen IFEval document")
    require(messages == [{"role": "user", "content": doc["prompt"]}],
            "request differs from the stock IFEval prompt")
    require(choice["finish_reason"] in ("stop", "length"), "unexpected IFEval finish reason")
    message = choice["message"]
    if think_end:
        require(not (message.get("reasoning") or message.get("reasoning_content")),
                "IFEval needs the think-end token in raw reply content; disable the server's reasoning parser")

    prepare()
    from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict

    verdict = test_instruction_following_strict(InputExample(**doc), scored).follow_all_instructions
    require(verdict is sample.get(METRIC), "logged lm-eval strict score differs from the replayed score")
    completed_thinking = not think_end or think_end in message["content"]
    answer = scored if completed_thinking and scored.strip() else None
    return {"score": float(verdict), "answer": answer, "correct": answer is not None and verdict}


if __name__ == "__main__":
    try:
        prepare()
    except BenchError as e:
        raise SystemExit(str(e)) from None
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
