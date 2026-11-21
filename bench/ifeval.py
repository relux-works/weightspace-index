"""Score IFEval replies with the pinned lm-eval instruction checkers."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from string import ascii_letters

from .core import BenchError, pinned_packages, require, sha256_bytes, sha256_file

TASK = "ifeval"
METRIC = "prompt_level_strict_acc"
GRADER = "literal-character-frequency-v1"
LETTER_FREQUENCY = "keywords:letter_frequency"
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
    install_literal_frequency_checker()
    return packages


def is_literal_character(letter: str | None) -> bool:
    return isinstance(letter, str) and len(letter) == 1 and letter not in ascii_letters


def install_literal_frequency_checker() -> None:
    from lm_eval.tasks.ifeval import instructions_registry
    from lm_eval.tasks.ifeval.instructions import LetterFrequencyChecker

    class LiteralFrequencyChecker(LetterFrequencyChecker):
        def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
            if not is_literal_character(letter):
                return super().build_description(letter=letter, let_frequency=let_frequency,
                                                 let_relation=let_relation)
            # Initialize the stock threshold and relation without its random
            # replacement of a supplied non-letter character.
            super().build_description(letter="a", let_frequency=let_frequency, let_relation=let_relation)
            self._letter = letter.lower()
            return self._description_pattern.format(letter=self._letter, let_frequency=self._frequency,
                                                    let_relation=self._comparison_relation)

    instructions_registry.INSTRUCTION_DICT[LETTER_FREQUENCY] = LiteralFrequencyChecker


def verify_legacy_score(sample: dict, doc: dict, replayed: list[bool]) -> None:
    random_checks = {i for i, (instruction, kwargs) in enumerate(zip(doc["instruction_id_list"], doc["kwargs"]))
                     if instruction == LETTER_FREQUENCY and is_literal_character(kwargs.get("letter"))}
    if not random_checks:
        require(all(replayed) is sample.get(METRIC), "logged lm-eval strict score differs from the replayed score")
        return
    # The original checker did not record the random letter it counted.
    # Verify all other instruction verdicts and the recorded aggregation.
    logged = sample.get("inst_level_strict_acc")
    require(isinstance(logged, list) and len(logged) == len(replayed)
            and all(isinstance(passed, bool) for passed in logged)
            and all(logged) is sample.get(METRIC)
            and all(passed is logged[i] for i, passed in enumerate(replayed) if i not in random_checks),
            "logged lm-eval strict score is inconsistent with the replayed stable instructions")


def grade_sample(sample: dict, messages: list, choice: dict, scored: str, think_end: str | None,
                 grader: str | None) -> dict:
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

    replayed = test_instruction_following_strict(InputExample(**doc), scored)
    if grader is None:
        verify_legacy_score(sample, doc, replayed.follow_instruction_list)
    else:
        require(grader == GRADER, f"unsupported IFEval grader {grader!r}")
        require(replayed.follow_all_instructions is sample.get(METRIC),
                "logged lm-eval strict score differs from the replayed score")
    completed_thinking = not think_end or think_end in message["content"]
    answer = scored if completed_thinking and scored.strip() else None
    return {"score": float(sample[METRIC]), "answer": answer,
            "correct": answer is not None and replayed.follow_all_instructions}


if __name__ == "__main__":
    try:
        prepare()
    except BenchError as e:
        raise SystemExit(str(e)) from None
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
