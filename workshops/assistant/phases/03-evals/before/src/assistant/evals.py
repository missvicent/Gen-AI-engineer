"""Eval-suite layer — put the RAG core on trial and block merges that regress it.

The layer every later workshop plugs into: memory adds recall rows, hardening adds
red-team rows, deploy wires `gate()` into the pipeline. Three rules the reference
follows, and you should too:

  * **The judge is injected** (`Judge` below). Offline runs use `KeywordJudge`, an
    honestly-named lexical stub, so the whole suite runs with no model and no network.
  * **Abstention is scored without a judge** — "did it refuse?" is a string check on
    the slice the business cares about most.
  * **Score per slice.** An overall mean hides a collapsed slice, and the collapsed
    slice is always the interesting one.

Reference: ../../../after/src/assistant/evals.py
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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

BARS = {"faithfulness": 0.85, "context_recall": 0.80}
COLLAPSE_FLOOR = 0.5


@dataclass(frozen=True)
class GoldenRow:
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


class Judge(Protocol):
    """The only component allowed to need a model."""

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float: ...

    def context_recall(self, question: str, contexts: list[str], reference: str) -> float: ...


AnswerFn = Callable[[str], tuple[str, list[str]]]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if len(w) > 3}


def _overlap(text: str, contexts: list[str]) -> float:
    """Share of `text`'s words that appear in the joined contexts."""
    t = words(text)
    c = words(" ".join(contexts))
    if not t or not c:
        return 0.0
    return len(t & c) / len(t)


class KeywordJudge:
    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        return _overlap(answer, contexts)

    def context_recall(self, question: str, contexts: list[str], reference: str) -> float:
        return _overlap(reference, contexts)


def is_abstention(answer: str) -> bool:
    return any(marker.lower() in answer.lower() for marker in ABSTENTION_MARKERS)


def score_row(row: GoldenRow, answer: str, contexts: list[str], judge: Judge) -> ScoredRow:
    refused = is_abstention(answer)
    if row.expects_abstention:
        if refused:
            return ScoredRow(row.id, row.slice, {"faithfulness": 1.0, "context_recall": 1.0}, False)
        else:
            return ScoredRow(row.id, row.slice, {"faithfulness": 0.0, "context_recall": 0.0}, False)
    scores = {
        "faithfulness": judge.faithfulness(row.question, answer, contexts),
        "context_recall": judge.context_recall(row.question, contexts, row.ground_truth),
    }
    return ScoredRow(row.id, row.slice, scores, True)


def mean_scores(rows: Iterable[ScoredRow]) -> dict[str, float]:
    rows = list(rows)
    scores = {}
    if not rows:
        return dict.fromkeys(METRICS, 0.0)
    for row in rows:
        for metric, score in row.scores.items():
            scores[metric] = scores.get(metric, 0.0) + score
    return {metric: round(scores[metric] / len(rows), 3) for metric in scores}


def run_suite(rows: list[GoldenRow], answer_fn: AnswerFn, judge: Judge) -> SuiteResult:
    scored = [score_row(row, *answer_fn(row.question), judge) for row in rows]
    overall = mean_scores(scored)
    by_slice = {
        row.slice: mean_scores([r for r in scored if r.slice == row.slice]) for row in scored
    }
    return SuiteResult(overall, by_slice, scored)


def gate(
    result: SuiteResult,
    baseline: dict[str, dict[str, float]],
    bars: dict[str, float] | None = None,
    tolerance: float = 0.03,
) -> list[str]:
    blocks = []
    for name, base_scores in baseline.items():
        for metric, before in base_scores.items():
            now = result.by_slice.get(name, {}).get(metric, 0.0)
            if now < round(before - tolerance, 3):
                blocks.append(f"{name} {metric} {now} < baseline {before}")
            if before > COLLAPSE_FLOOR and now < COLLAPSE_FLOOR:
                blocks.append(f"{name} {metric} COLLAPSED: {before} → {now}")
    bars = bars or BARS
    for metric, bar in bars.items():
        now_overall = result.overall.get(metric, 0.0)
        if now_overall < bar:
            blocks.append(f"overall {metric} {now_overall} < bar {bar}")
    return blocks


# --- trajectory: score what the agent DID, not only its final message --------
#
# The agent loop (agent.py) audits every executed tool as "ran: <name>" and
# every containment pause as "paused for approval: <name>". That audit trail is
# a trace, and traces are structures — so all of this runs in the fast tier.

RAN_PREFIX = "ran: "


def tools_run(audit: Iterable[str]) -> list[str]:
    """TODO 10: executed tools in order — the audit entries that start with
    RAN_PREFIX, with the prefix stripped."""
    raise NotImplementedError


def tool_choice_f1(executed: list[str], expected: list[str]) -> float:
    """TODO 11: F1 over tool names as multisets (collections.Counter and `&`).
    Both empty -> 1.0; no overlap -> 0.0; round to 3."""
    raise NotImplementedError


def goal_completion(answer: str, must_mention: list[str]) -> float:
    """TODO 12: fraction of required facts the final answer carries
    (case-insensitive substring per fact). Nothing required -> 1.0; round to 3."""
    raise NotImplementedError


def containment_ok(
    audit: Iterable[str],
    gated: frozenset[str] | set[str],
    approvals: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """TODO 13: True when no gated tool executed without an approval on file.
    This must ALWAYS hold — one unapproved firing is an incident, whatever the
    final answer said."""
    raise NotImplementedError


def agreement(human: list[str], judge: list[str]) -> float:
    """TODO 7: share of rows where the two raters said the same thing."""
    raise NotImplementedError


def cohen_kappa(human: list[str], judge: list[str], labels: tuple[str, ...]) -> float:
    """TODO 8: agreement corrected for chance — the number you actually report.

        kappa = (observed - expected) / (1 - expected)

    where `expected` sums, over each label, the product of how often each rater used
    it. Raw agreement flatters a rubber stamp; kappa doesn't.

    (At work you would import `sklearn.metrics.cohen_kappa_score` — see
    `phase3-evals/03-judge-calibration`. It is inlined here to keep this workshop
    dependency-free.)
    """
    raise NotImplementedError


def format_table(result: SuiteResult) -> str:
    """TODO 9: one line per slice, then an OVERALL line. Read this, not the average."""
    raise NotImplementedError
