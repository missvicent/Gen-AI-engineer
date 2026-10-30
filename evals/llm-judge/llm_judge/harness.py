"""Build the harness so that everything EXCEPT the judge is testable offline.

The judge is injected. That one decision is what lets `make test` cover the grading
logic with no model, no key and no network — and it is the difference between an
eval suite you run on every push and one you run when you remember to.

Two rules the tests will hold you to:

  * Abstention rows never reach the judge. "Did the system refuse?" is a string
    check; sending it to an LLM adds cost, latency and noise to the slice you least
    want noise in.
  * Aggregate per slice as well as overall. An overall 0.86 can hide the
    unanswerable slice at 0.40.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

METRICS = ("faithfulness", "context_recall")

ABSTENTION_MARKERS = (
    "not in the docs",
    "not in the documents",
    "i don't know",
    "i do not know",
    "no information",
    "cannot answer",
    "can't answer",
)


class Judge(Protocol):
    """The only component allowed to need a model."""

    def faithfulness(
        self, question: str, answer: str, contexts: list[str]
    ) -> float: ...

    def context_recall(
        self, question: str, contexts: list[str], reference: str
    ) -> float: ...

    def describe(self) -> dict[str, str]: ...


Pipeline = Callable[[str], tuple[str, list[str]]]


@dataclass(frozen=True)
class Row:
    id: str
    slice: str
    question: str
    ground_truth: str
    expects_abstention: bool = False
    supporting_doc_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoredRow:
    id: str
    slice: str
    scores: dict[str, float]
    judged: bool


@dataclass(frozen=True)
class SuiteResult:
    overall: dict[str, float]
    by_slice: dict[str, dict[str, float]]
    rows: list[ScoredRow]
    instrument: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def load_golden(path: str | Path) -> list[Row]:
    valid_fields = {f.name for f in fields(Row)}
    rows = [
        Row(**{k: v for k, v in json.loads(line).items() if k in valid_fields})
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate row ids found in golden set")
    return rows


def is_abstention(answer: str) -> bool:
    return any(marker.lower() in answer.lower() for marker in ABSTENTION_MARKERS)


def score_row(row: Row, answer: str, contexts: list[str], judge: Judge) -> ScoredRow:
    """Score one row.

    Abstention rows: 1.0 on every metric if the system refused, 0.0 if it invented
    an answer — and `judged=False`, because the judge was never called.
    Everything else: ask the judge, `judged=True`.
    """
    if row.expects_abstention:
        if is_abstention(answer):
            return ScoredRow(
                id=row.id,
                slice=row.slice,
                scores={m: 1.0 for m in METRICS},
                judged=False,
            )
        else:
            return ScoredRow(
                id=row.id,
                slice=row.slice,
                scores={m: 0.0 for m in METRICS},
                judged=False,
            )
    else:
        return ScoredRow(
            id=row.id,
            slice=row.slice,
            scores={
                "context_recall": judge.context_recall(
                    row.question, contexts, row.ground_truth
                ),
                "faithfulness": judge.faithfulness(row.question, answer, contexts),
            },
            judged=True,
        )


def mean_scores(rows: Iterable[ScoredRow]) -> dict[str, float]:
    """Mean of scores per metric, rounded to 3 dp.
    An empty input is 0.0, not a crash.
    """
    rows = list[ScoredRow](rows)
    if not rows:
        return {m: 0.0 for m in METRICS}
    return {
        m: round(sum(row.scores[m] for row in rows) / len(rows), 3) for m in METRICS
    }


def run_suite(rows: list[Row], pipeline: Pipeline, judge: Judge) -> SuiteResult:
    """Score every row, then aggregate overall AND per slice.

    Record `judge.describe()` as the instrument — a score without its instrument is
    not a measurement.
    """
    scored = [score_row(row, *pipeline(row.question), judge=judge) for row in rows]
    slices = sorted({r.slice for r in scored})
    return SuiteResult(
        overall=mean_scores(scored),
        by_slice={s: mean_scores(r for r in scored if r.slice == s) for s in slices},
        rows=scored,
        instrument=judge.describe(),
    )


def format_table(result: SuiteResult) -> str:
    return "\n".join(
        [f"{s}: {result.by_slice[s]}" for s in result.by_slice]
        + [f"Overall: {result.overall}"]
        + [f"judged rows: {len([r for r in result.rows if r.judged])}"]
    )
