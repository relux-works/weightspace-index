"""Final-answer grading for MATH-500."""

import functools
import logging

from .core import pinned_packages, require

MATH_TASK = "minerva_math500"
BOXED = "\\boxed{"


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
