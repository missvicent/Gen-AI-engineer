"""TODO: make the golden set testable, then make the golden set pass.

Two jobs in this lesson, in this order:

  1. Implement the checks below. Use `rapidfuzz` — you are not writing string
     algorithms:

         from rapidfuzz import fuzz
         from rapidfuzz.utils import default_process

         fuzz.token_set_ratio(a, b, processor=default_process)   # same question, reworded
         fuzz.partial_ratio(q, chunk, processor=default_process) # question lifted from a chunk

     `processor=default_process` lowercases and strips punctuation. Without it a
     capital letter is enough to hide a true duplicate.

  2. Fix `evals/golden.jsonl` until `validate()` returns an empty list. The
     shipped file has real problems in it — your checks are how you find them,
     and `python -m src.dataset` prints the report.

Reference: ../after/src/dataset.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SLICES = ("semantic", "exact", "multi_hop", "unanswerable", "adversarial")

MIN_UNANSWERABLE = 5
DUPLICATE_THRESHOLD = 92.0
LEAKAGE_THRESHOLD = 95.0


@dataclass(frozen=True)
class GoldenRow:
    id: str
    slice: str
    question: str
    ground_truth: str
    expects_abstention: bool = False
    supporting_doc_ids: list[str] = field(default_factory=list)
    source: str = ""
    labeled_by: str = ""
    labeled_on: str = ""


def load_golden(path: str | Path) -> list[GoldenRow]:
    """TODO 1: read the jsonl into GoldenRow objects; raise on duplicate ids."""
    raise NotImplementedError


def load_corpus(path: str | Path) -> list[str]:
    """TODO 2: return the chunk texts — the leakage check needs them."""
    raise NotImplementedError


def near_duplicates(
    rows: list[GoldenRow], threshold: float = DUPLICATE_THRESHOLD
) -> list[tuple[str, str]]:
    """TODO 3: every pair of ids whose questions are the same question, reworded."""
    raise NotImplementedError


def leaked_questions(
    rows: list[GoldenRow], chunks: list[str], threshold: float = LEAKAGE_THRESHOLD
) -> list[str]:
    """TODO 4: ids whose question is copy-pasted out of a chunk."""
    raise NotImplementedError


def slice_counts(rows: list[GoldenRow]) -> dict[str, int]:
    """TODO 5: how many rows in each of SLICES (zeros included — that's the point)."""
    raise NotImplementedError


def missing_provenance(rows: list[GoldenRow]) -> list[str]:
    """TODO 6: ids lacking source / labeled_by / labeled_on."""
    raise NotImplementedError


def unsupported_answerable(rows: list[GoldenRow]) -> list[str]:
    """TODO 7: answerable ids with no supporting_doc_ids (no judge-free metrics without it)."""
    raise NotImplementedError


def validate(rows: list[GoldenRow], chunks: list[str]) -> list[str]:
    """TODO 8: collect every problem as a human-readable string.

    Cover: empty slices, too few unanswerable rows (< MIN_UNANSWERABLE),
    near-duplicates, leakage, missing provenance, answerable rows with no
    supporting docs, unknown slice names, and `expects_abstention` disagreeing
    with the `unanswerable` slice.
    """
    raise NotImplementedError


def report(rows: list[GoldenRow], chunks: list[str]) -> str:
    """TODO 9: counts per slice plus the problem list — the PR-comment view."""
    raise NotImplementedError


if __name__ == "__main__":
    print(report(load_golden("evals/golden.jsonl"), load_corpus("evals/corpus.jsonl")))
