"""Tier 2: the REAL RAGAS metrics with a pinned judge. Needs Ollama running.

make test-integration
"""

import pytest

pytestmark = pytest.mark.integration

GOLDEN = "evals/golden.jsonl"


def _pipeline(q: str):
    from tests.test_quality import _pipeline as p

    return p(q)


def test_real_ragas_scores_and_gates():
    """The live judge, asserted on its numbers.

    This used to check that `"faithfulness" in scores` and that `gate` returned a
    bool and a list — true of a function that returns `{"faithfulness": 0.0}` and
    of one that returns nothing at all. It passed while proving only that the
    imports resolved.

    A judge's exact values are not predictable even at temperature 0, so what is
    asserted here is what a judge must manage on this input rather than a specific
    score. `_pipeline` answers three questions using their own ground truth as the
    only retrieved context: every claim in the answer is supported and the
    reference is fully covered. If a pinned judge cannot clear the bars on that, the
    finding is about the judge, and finding that out is the point of running it.

    The arithmetic between a verdict and a merge decision is tested in
    `test_gate.py`, offline and by the number — including the boundary cases and
    the NaN a live run will not produce on demand.
    """
    from src.harness import load_golden
    from src.ragas_eval import CONTEXT_RECALL_BAR, FAITHFULNESS_BAR, gate, run_ragas

    scores = run_ragas(load_golden(GOLDEN)[:3], _pipeline)
    assert set(scores) == {"faithfulness", "context_recall"}
    for metric, value in scores.items():
        assert 0.0 <= value <= 1.0, f"{metric}={value} is not a RAGAS score"
    assert scores["faithfulness"] >= FAITHFULNESS_BAR, (
        f"the judge scored a perfectly grounded answer at {scores['faithfulness']:.2f}"
    )
    assert scores["context_recall"] >= CONTEXT_RECALL_BAR, (
        f"the judge missed a reference that WAS the only context: {scores}"
    )
    ok, reasons = gate(scores)
    assert ok is True, f"the gate failed on a grounded pipeline: {reasons}"


def test_the_installed_ragas_is_the_one_this_lesson_was_written_against():
    """A pin in a pyproject is an intention; this is the check.

    RAGAS moved its metric classes into `ragas.metrics.collections` at 0.4 and
    left the old import path importable with a DeprecationWarning — so code
    written against the old surface keeps working long enough to be copied,
    and then diverges from what lesson 3.2 teaches. Fail loudly instead."""
    from importlib.metadata import version

    from ragas.llms import llm_factory  # noqa: F401  (the 0.4 judge builder)
    from ragas.metrics.collections import ContextRecall, Faithfulness  # noqa: F401

    major, minor, *_ = version("ragas").split(".")
    assert (int(major), int(minor)) == (0, 4), (
        f"this lesson is written against ragas 0.4.x, found {version('ragas')}"
    )


def test_the_scores_travel_with_the_instrument_that_produced_them():
    """A number without its judge is not comparable to next week's number."""
    from src.ragas_eval import describe

    stamp = describe()
    assert stamp["judge_temperature"] == "0.0"
    assert stamp["judge_model"] and stamp["ragas_version"]
