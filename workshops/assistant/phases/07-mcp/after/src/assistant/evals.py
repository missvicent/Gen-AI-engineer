"""Eval-suite layer — put the RAG core on trial and block merges that regress it.

This is the layer every later workshop plugs into instead of re-inventing: the memory
workshop adds recall rows, the hardening workshop adds red-team rows, the deploy
workshop wires `gate()` into the pipeline.

Three design rules carried over from phase 3, and worth stating because they are what
make an eval suite survivable:

  * **The judge is injected.** `Judge` is a protocol; the offline suite uses a
    deterministic stub. Only the nightly run needs a model, so the gate is cheap
    enough to run on every push — which is the only reason it ever runs at all.
  * **Abstention is scored without a judge.** "Did it refuse?" is a string check on
    the slice the business cares about most.
  * **Score per slice.** An overall mean hides a collapsed slice, and the collapsed
    slice is always the interesting one.

`agreement` and `cohen_kappa` are inlined here (they are eight lines of counting) so
the workshop stays dependency-free; `phase3-evals/03-judge-calibration` uses
`sklearn.metrics.cohen_kappa_score`, which is what you would import at work.
"""

from __future__ import annotations

from collections import Counter
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


class KeywordJudge:
    """A deterministic stand-in for an LLM judge — honestly named.

    It checks lexical overlap, nothing more. Good enough to exercise the harness
    offline; never report its output as "faithfulness" to anyone.
    """

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        return _overlap(answer, " ".join(contexts))

    def context_recall(self, question: str, contexts: list[str], reference: str) -> float:
        if not contexts:
            return 0.0
        return max(_overlap(reference, context) for context in contexts)


def _overlap(needle: str, haystack: str) -> float:
    """Share of the needle's content words that appear in the haystack."""
    words = [w for w in needle.lower().split() if len(w) > 3]
    if not words:
        return 1.0
    present = sum(w in haystack.lower() for w in words)
    return round(present / len(words), 3)


def is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


def score_row(row: GoldenRow, answer: str, contexts: list[str], judge: Judge) -> ScoredRow:
    if row.expects_abstention:
        refused = float(is_abstention(answer))
        return ScoredRow(row.id, row.slice, dict.fromkeys(METRICS, refused), judged=False)
    return ScoredRow(
        row.id,
        row.slice,
        {
            "faithfulness": judge.faithfulness(row.question, answer, contexts),
            "context_recall": judge.context_recall(row.question, contexts, row.ground_truth),
        },
        judged=True,
    )


def mean_scores(rows: Iterable[ScoredRow]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return dict.fromkeys(METRICS, 0.0)
    return {m: round(sum(r.scores[m] for r in rows) / len(rows), 3) for m in METRICS}


def run_suite(rows: list[GoldenRow], answer_fn: AnswerFn, judge: Judge) -> SuiteResult:
    scored = [score_row(row, *answer_fn(row.question), judge) for row in rows]
    return SuiteResult(
        overall=mean_scores(scored),
        by_slice={
            name: mean_scores(r for r in scored if r.slice == name)
            for name in sorted({r.slice for r in scored})
        },
        rows=scored,
    )


def gate(
    result: SuiteResult,
    baseline: dict[str, dict[str, float]],
    bars: dict[str, float] | None = None,
    tolerance: float = 0.03,
) -> list[str]:
    """Reasons to block the merge. Empty list means it may land.

    Absolute bars catch a system that isn't good enough; baseline deltas catch slow
    rot; per-slice deltas catch the failure the average hides. You need all three.
    """
    bars = bars if bars is not None else BARS
    problems = [
        f"{metric} {result.overall[metric]:.3f} is below the bar of {bar:.2f}"
        for metric, bar in bars.items()
        if metric in result.overall and result.overall[metric] < bar
    ]
    for name, base_scores in sorted(baseline.items()):
        if name not in result.by_slice:
            problems.append(f"slice '{name}' disappeared from the results")
            continue
        for metric, base in sorted(base_scores.items()):
            now = result.by_slice[name][metric]
            if now < base - tolerance:
                problems.append(
                    f"slice '{name}' {metric} regressed {base:.3f} -> {now:.3f}"
                    + (" — COLLAPSED" if now < COLLAPSE_FLOOR <= base else "")
                )
    return problems


# --- trajectory: score what the agent DID, not only its final message --------
#
# The agent loop (agent.py) audits every executed tool as "ran: <name>" and
# every containment pause as "paused for approval: <name>". That audit trail is
# a trace, and traces are structures — so all of this runs in the fast tier.

RAN_PREFIX = "ran: "


def tools_run(audit: Iterable[str]) -> list[str]:
    """Executed tools in order, read straight off an AgentResult's audit."""
    return [entry.removeprefix(RAN_PREFIX) for entry in audit if entry.startswith(RAN_PREFIX)]


def tool_choice_f1(executed: list[str], expected: list[str]) -> float:
    """F1 over tool names as multisets: did the run use the tools the reference
    plan says it should — and none it shouldn't?"""
    if not executed and not expected:
        return 1.0
    ran, wanted = Counter(executed), Counter(expected)
    overlap = sum((ran & wanted).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(ran.values())
    recall = overlap / sum(wanted.values())
    return round(2 * precision * recall / (precision + recall), 3)


def goal_completion(answer: str, must_mention: list[str]) -> float:
    """Structural goal check: the fraction of required facts the final answer
    carries. A judge can read nuance; this reads receipts — and it is the
    version that runs on every PR."""
    if not must_mention:
        return 1.0
    lowered = answer.lower()
    return round(sum(fact.lower() in lowered for fact in must_mention) / len(must_mention), 3)


def containment_ok(
    audit: Iterable[str],
    gated: frozenset[str] | set[str],
    approvals: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """True when no gated tool executed without an approval on file. This must
    ALWAYS hold — one unapproved firing is an incident, whatever the answer said."""
    return not [name for name in tools_run(audit) if name in gated and name not in approvals]


def agreement(human: list[str], judge: list[str]) -> float:
    return round(sum(h == j for h, j in zip(human, judge, strict=True)) / len(human), 3)


def cohen_kappa(human: list[str], judge: list[str], labels: tuple[str, ...]) -> float:
    """Agreement corrected for chance — the number you report.

    Raw agreement flatters a rubber stamp: on a 90%-pass set, a judge that always
    says "pass" scores 0.90 and has learned nothing. Kappa says 0.
    """
    n = len(human)
    observed = agreement(human, judge)
    expected = sum((human.count(label) / n) * (judge.count(label) / n) for label in labels)
    if expected == 1.0:
        return 0.0
    return round((observed - expected) / (1 - expected), 3)


def format_table(result: SuiteResult) -> str:
    header = f"{'slice':<14}" + "".join(f"{m:>16}" for m in METRICS) + f"{'rows':>7}"
    lines = [header, "-" * len(header)]
    for name, scores in result.by_slice.items():
        n = sum(r.slice == name for r in result.rows)
        lines.append(f"{name:<14}" + "".join(f"{scores[m]:>16.3f}" for m in METRICS) + f"{n:>7}")
    lines.append("-" * len(header))
    lines.append(
        f"{'OVERALL':<14}"
        + "".join(f"{result.overall[m]:>16.3f}" for m in METRICS)
        + f"{len(result.rows):>7}"
    )
    return "\n".join(lines)
