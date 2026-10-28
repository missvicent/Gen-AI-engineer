"""TODO: make the golden set testable, then make the golden set pass.

Two jobs in this lesson, in this order:

  1. Implement the checks below. Use `rapidfuzz` — you are not writing string
     algorithms:

         from rapidfuzz import fuzz
         from rapidfuzz.utils import default_process

         fuzz.token_set_ratio(a, b, processor=default_process)    # reworded question
         fuzz.partial_ratio(q, chunk, processor=default_process)  # lifted from a chunk

     `processor=default_process` lowercases and strips punctuation. Without it a
     capital letter is enough to hide a true duplicate.

  2. Fix `evals/golden.jsonl` until `validate()` returns an empty list. The
     shipped file has real problems in it — your checks are how you find them,
     and `python -m src.dataset` prints the report.

Reference: ../after/src/dataset.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

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
    rows = [
        GoldenRow(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate row ids found in golden set")
    return rows


def load_corpus(path: str | Path) -> list[str]:
    return [
        json.loads(line)["text"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def near_duplicates(
    rows: list[GoldenRow], threshold: float = DUPLICATE_THRESHOLD
) -> list[tuple[str, str]]:
    """Pairs of questions that are the same question, reworded."""
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if (
                fuzz.token_set_ratio(a.question, b.question, processor=default_process)
                >= threshold
            ):
                pairs.append((a.id, b.id))
    return pairs


def leaked_questions(
    rows: list[GoldenRow], chunks: list[str], threshold: float = LEAKAGE_THRESHOLD
) -> list[str]:
    """Rows whose question is copy-pasted out of a chunk.

    Those rows score well for the wrong reason. The retriever is matching the
    question's own source text, so you're measuring string overlap and calling it
    retrieval quality.
    """
    return [
        r.id
        for r in rows
        if any(
            fuzz.partial_ratio(r.question, chunk, processor=default_process)
            >= threshold
            for chunk in chunks
        )
    ]


def slice_counts(rows: list[GoldenRow]) -> dict[str, int]:
    return {s: sum(1 for r in rows if r.slice == s) for s in SLICES}


def missing_provenance(rows: list[GoldenRow]) -> list[str]:
    return [r.id for r in rows if not all((r.source, r.labeled_by, r.labeled_on))]


def unsupported_answerable(rows: list[GoldenRow]) -> list[str]:
    """Answerable rows must name the docs that support them.

    That single field is what lets you compute retrieval metrics with no judge:
    did the supporting docs show up in the top-k? is a set membership test.
    """
    return [r.id for r in rows if not r.expects_abstention and not r.supporting_doc_ids]


def validate(rows: list[GoldenRow], chunks: list[str]) -> list[str]:
    """Collect every problem as a human-readable string.

    Cover: empty slices, too few unanswerable rows (< MIN_UNANSWERABLE),
    near-duplicates, leakage, missing provenance, answerable rows with no
    supporting docs, unknown slice names, and `expects_abstention` disagreeing
    with the `unanswerable` slice.
    """
    problems: list[str] = []

    counts = slice_counts(rows)
    problems += [f"slice {s} is empty" for s, n in counts.items() if n == 0]

    unanswerable = sum(r.expects_abstention for r in rows)
    if unanswerable < MIN_UNANSWERABLE:
        problems.append(
            f"only {unanswerable} unanswerable rows (need >= {MIN_UNANSWERABLE}): "
            "the abstain path goes untested"
        )

    problems += [
        f"near-duplicate questions: {a} ~ {b}" for a, b in near_duplicates(rows)
    ]
    problems += [
        f"question leaked from chunk: {rid}" for rid in leaked_questions(rows, chunks)
    ]
    problems += [f"missing provenance: {rid}" for rid in missing_provenance(rows)]
    problems += [
        f"answerable rows with no supporting docs: {rid}"
        for rid in unsupported_answerable(rows)
    ]

    for row in rows:
        if row.slice not in SLICES:
            problems.append(f"unknown slice: {row.slice} on {row.id}")
        if row.expects_abstention != (row.slice == "unanswerable"):
            problems.append(
                f"{row.id}: expects_abstention and slice 'unanswerable' must agree"
            )

    return problems


def report(rows: list[GoldenRow], chunks: list[str]) -> str:
    """Markdown block to paste into the PR that grows the golden set."""
    counts = slice_counts(rows)
    problems = validate(rows, chunks)
    total = len(rows)

    status = "✅ valid" if not problems else f"❌ {len(problems)} problem(s)"
    lines = [
        f"**Golden set: {total} rows, {status}**",
        "",
        "| slice | rows | share |",
        "|---|---:|---:|",
        *(f"| `{s}` | {n} | {n / (total or 1):.0%} |" for s, n in counts.items()),
    ]
    if problems:
        lines += ["", "**Problems**", *(f"- {p}" for p in problems)]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(load_golden("evals/golden.jsonl"), load_corpus("evals/corpus.jsonl")))
