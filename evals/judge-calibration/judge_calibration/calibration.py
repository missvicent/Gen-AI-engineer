"""TODO: measure how much your judge deserves to be trusted.

You have 40 rows in `evals/labeled.jsonl` that a human (you) labeled `pass`/`fail`,
each with the judge's raw score. Turn that into a decision: can this judge gate
merges, at what threshold, and with what regression tolerance?

Report both numbers, and know why:

  * **agreement** — share of rows where you and the judge said the same thing. A
    liar under class imbalance: with 90% passes, a judge that always says "pass"
    scores 0.90.
  * **Cohen's kappa** — the same overlap corrected for chance agreement. Use
    `sklearn.metrics.cohen_kappa_score(human, judge, labels=["pass", "fail"])`.
    Do not hand-roll inter-rater statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

LABELS = ("pass", "fail")

# Landis & Koch's bands for inter-rater agreement: a convention for deciding how far
# to trust a second rater, not a law of nature.
BANDS = (
    (0.20, "the judge is measuring something else entirely — rewrite the rubric"),
    (0.40, "weak signal: fine for smoke tests, not for gating merges"),
    (0.60, "usable with a margin — gate on regressions, not on absolutes"),
    (1.01, "substantial agreement — you can defend gating merges on this"),
)

GATING_KAPPA = 0.60


@dataclass(frozen=True)
class LabeledRow:
    id: str
    question: str
    answer: str
    judge_score: float
    human: str
    labeled_by: str = ""
    labeled_on: str = ""


@dataclass(frozen=True)
class Calibration:
    n: int
    threshold: float
    agreement: float
    kappa: float
    disagreements: list[str]
    judge_pass_rate: float
    human_pass_rate: float

    @property
    def gatable(self) -> bool:
        return self.kappa >= GATING_KAPPA

    @property
    def tolerance(self) -> float:
        """TODO 7: the smallest SUITE-LEVEL move worth failing CI over.

        Per-row disagreement is `1 - agreement`. Averaged over n rows that noise
        shrinks like `disagreement / sqrt(n)` — anything smaller than that is the
        judge having a different opinion than you on a couple of rows, not a
        regression. Floor it at 0.01 and round to 2 dp.
        """
        raise NotImplementedError


def load_labeled(path: str | Path) -> list[LabeledRow]:
    valid_fields = {f.name for f in fields(LabeledRow)}
    rows = [
        LabeledRow(**{k: v for k, v in json.loads(line).items() if k in valid_fields})
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate row ids found in labeled set")

    problems: list[str] = []
    for row in rows:
        if row.human not in LABELS:
            problems.append(f"unknown human labels: {row.human!r} for row {row.id!r}")
    if problems:
        raise ValueError("invalid labeled rows:\n" + "\n".join(problems))
    return rows


def verdicts(rows: list[LabeledRow], threshold: float) -> list[str]:
    return ["pass" if row.judge_score >= threshold else "fail" for row in rows]


def calibrate(rows: list[LabeledRow], threshold: float = 0.5) -> Calibration:
    human = [r.human for r in rows]
    judge_verdicts = verdicts(rows, threshold)
    agreement = sum(h == j for h, j in zip(human, judge_verdicts)) / len(rows)
    kappa = cohen_kappa_score(human, judge_verdicts, labels=LABELS)
    disagreements = [r.id for r, h, j in zip(rows, human, judge_verdicts) if h != j]
    judge_pass_rate = sum(j == "pass" for j in judge_verdicts) / len(rows)
    human_pass_rate = sum(r.human == "pass" for r in rows) / len(rows)

    return Calibration(
        n=len(rows),
        threshold=threshold,
        agreement=agreement,
        kappa=kappa,
        disagreements=disagreements,
        judge_pass_rate=judge_pass_rate,
        human_pass_rate=human_pass_rate,
    )


def best_threshold(rows: list[LabeledRow], step: float = 0.05) -> Calibration:
    return max(
        (
            calibrate(rows, threshold=round(i * step, 2))
            for i in range(int(1 / step) + 1)
        ),
        key=lambda x: x.kappa,
    )


def interpret(kappa: float) -> str:
    """TODO 5: the first BANDS advice whose edge the kappa falls under."""
    raise NotImplementedError


def disagreement_rows(
    rows: list[LabeledRow], calibration: Calibration
) -> list[LabeledRow]:
    """TODO 8: the rows you and the judge disagreed on — the real deliverable.

    Each one is a bad rubric, a bad label, or a question too ambiguous to belong in
    the golden set. Reading them is the work; the kappa is just the receipt.
    """
    raise NotImplementedError


def report(rows: list[LabeledRow]) -> str:
    """TODO 9: default vs best threshold, the verdict, the gate decision, tolerance.

    Must mention "gate on this judge?", "regression tolerance:" and "best kappa" —
    this text is what you keep next to your scores.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print(report(load_labeled("evals/labeled.jsonl")))
