"""Tier 2's arithmetic, tested without tier 2's judge.

`test_integration.py` needs Ollama, a multi-gigabyte model and a few minutes, so
it runs nightly and asserts almost nothing: that the keys exist and that `gate`
returns a bool and a list. That is a connectivity check wearing the costume of a
quality gate. Anything it *could* assert about the numbers would be
non-deterministic anyway — a judge is a language model, and pinning its
temperature to 0 makes it repeatable, not predictable.

So the numbers get tested here instead, on the two pure functions that stand
between a judge's verdict and a merge decision. No judge, no network, no RAGAS
install: `src/ragas_eval.py` imports ragas lazily inside the functions that need
it, which is what makes this file possible and is worth copying.

What each test below is protecting:

  * `aggregate` — a mean over rows. Cheap to get wrong in a way that reads fine.
  * `as_score` — the guard on a judge's output. The NaN case is the one that
    matters: a NaN average compares False against every bar, so a gate holding
    one is a gate that has stopped gating and still prints a number.
  * `gate` — the bars. Tested AT the boundary in both directions, because `<` and
    `<=` look identical in every passing run and differ on exactly the value a
    team argues about.
"""

import math

import pytest

from src.ragas_eval import (
    CONTEXT_RECALL_BAR,
    FAITHFULNESS_BAR,
    aggregate,
    as_score,
    gate,
)


def rows(*pairs: tuple[float, float]) -> list[dict[str, float]]:
    return [{"faithfulness": f, "context_recall": c} for f, c in pairs]


# --- aggregate ----------------------------------------------------------------


def test_the_mean_is_the_mean_for_both_metrics():
    """Both numbers, computed independently. Two metrics averaged with one
    accumulator is a real bug and every row of a passing run hides it."""
    scores = aggregate(rows((1.0, 0.5), (0.5, 1.0), (0.9, 0.6)))
    assert scores["faithfulness"] == pytest.approx(0.8)
    assert scores["context_recall"] == pytest.approx(0.7)


def test_a_run_that_scored_nothing_is_zero_and_not_perfect():
    """A nightly job that died before its first row must FAIL the gate.

    `sum([]) / len([])` raises, and the tempting fix — skip the metric, or default
    it to 1.0 — turns a crashed eval into a green one. Zero is the honest reading
    of "no evidence", and zero is below every bar.
    """
    assert aggregate([]) == {"faithfulness": 0.0, "context_recall": 0.0}
    assert gate(aggregate([]))[0] is False


def test_one_bad_row_moves_the_mean_by_its_share():
    """The arithmetic behind the 30-row assignment, made concrete.

    Six rows, one of them zero: faithfulness lands at 0.833, under the 0.85 bar.
    With thirty rows the same single failure lands at 0.967 and passes. Neither
    number is more correct — the point is that a slice mean over a handful of rows
    is dominated by whichever row went wrong, which is why the fixture is a fixture.
    """
    six = aggregate(rows(*([(1.0, 1.0)] * 5 + [(0.0, 1.0)])))
    assert six["faithfulness"] == pytest.approx(5 / 6)
    assert gate(six)[0] is False

    thirty = aggregate(rows(*([(1.0, 1.0)] * 29 + [(0.0, 1.0)])))
    assert thirty["faithfulness"] == pytest.approx(29 / 30)
    assert gate(thirty)[0] is True


# --- as_score -----------------------------------------------------------------


def test_a_valid_verdict_passes_through_as_a_float():
    assert as_score(0.0, "faithfulness") == 0.0
    assert as_score(1, "faithfulness") == 1.0
    assert as_score(0.875, "context_recall") == 0.875


def test_nan_is_rejected_because_no_threshold_can_catch_it():
    """The whole reason this function exists.

    A NaN compares False against everything, so `nan < 0.85` is False and the gate
    passes. Worse, `mean([0.9, nan])` is `nan`, so one unparsed row can silence a
    gate over a whole run — while the report prints a number and looks measured.
    """
    assert math.isnan(float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        as_score(float("nan"), "faithfulness")
    # and the thing it protects against, stated directly
    assert not (float("nan") < FAITHFULNESS_BAR)


def test_a_missing_verdict_is_an_error_and_not_a_zero():
    """`None` means the judge did not answer, which is different from the judge
    saying "unfaithful". Silently reading it as 0.0 would turn a broken judge into
    a quality regression, and the team would go looking at the retriever."""
    with pytest.raises(ValueError, match="no score"):
        as_score(None, "context_recall")


@pytest.mark.parametrize("value", [-0.01, 1.01, 100.0])
def test_a_score_outside_zero_to_one_is_rejected(value: float):
    """The 0..100 mistake, caught. `rapidfuzz` returns 0..100 and RAGAS returns
    0..1; mixing them yields a faithfulness of 87.0, which clears a 0.85 bar by
    two orders of magnitude and looks like a triumph in the report."""
    with pytest.raises(ValueError, match="outside 0..1"):
        as_score(value, "faithfulness")


# --- gate ---------------------------------------------------------------------


def test_scores_exactly_on_the_bar_pass():
    """`>=`, not `>`. Asserted because this is the one value where the two
    implementations differ, and a team that has agreed on "0.85 or better" will
    argue about a build that fails at exactly 0.85."""
    ok, reasons = gate(
        {"faithfulness": FAITHFULNESS_BAR, "context_recall": CONTEXT_RECALL_BAR}
    )
    assert ok is True
    assert reasons == []


def test_a_hair_under_either_bar_fails_and_says_which():
    """A gate that fails without naming the metric sends someone to read the
    harness instead of the pipeline."""
    ok, reasons = gate(
        {
            "faithfulness": FAITHFULNESS_BAR - 0.01,
            "context_recall": CONTEXT_RECALL_BAR,
        }
    )
    assert ok is False
    assert len(reasons) == 1
    assert "faithfulness" in reasons[0] and str(FAITHFULNESS_BAR) in reasons[0]

    ok, reasons = gate(
        {
            "faithfulness": FAITHFULNESS_BAR,
            "context_recall": CONTEXT_RECALL_BAR - 0.01,
        }
    )
    assert ok is False
    assert "context_recall" in reasons[0]


def test_both_failures_are_reported_together():
    """Not the first one found. Fixing faithfulness and then discovering recall
    was also failing is two builds for one piece of information."""
    ok, reasons = gate({"faithfulness": 0.1, "context_recall": 0.1})
    assert ok is False
    assert len(reasons) == 2


def test_a_metric_the_judge_never_produced_fails_closed():
    """An absent key reads as 0.0 and fails, rather than being skipped.

    `scores.get(metric, 0.0)` is a one-word decision with an outcome: a run whose
    faithfulness metric crashed is a run with no faithfulness evidence, and the
    gate's job is to refuse it. Defaulting to a pass — or to `None` and a
    TypeError three frames later — is how a missing measurement becomes a green
    build.
    """
    ok, reasons = gate({"context_recall": 1.0})
    assert ok is False
    assert any("faithfulness" in r for r in reasons)


def test_a_passing_run_and_a_failing_run_end_to_end():
    """The two examples the exercise asks for, from per-row scores to verdict.

    Deliberately the whole path — `aggregate` then `gate` — because that is the
    composition a nightly build runs, and each function passing on its own does
    not prove the pair agrees about which metric is which.
    """
    passing = aggregate(rows((0.95, 0.90), (0.90, 0.85), (0.88, 0.95)))
    assert passing["faithfulness"] == pytest.approx(0.91)
    assert passing["context_recall"] == pytest.approx(0.90)
    assert gate(passing) == (True, [])

    failing = aggregate(rows((0.95, 0.40), (0.90, 0.30), (0.88, 0.50)))
    assert failing["faithfulness"] == pytest.approx(0.91)
    assert failing["context_recall"] == pytest.approx(0.40)
    ok, reasons = gate(failing)
    assert ok is False
    # faithfulness was fine — a gate that blamed both would be useless here
    assert reasons == [f"context_recall 0.40 < {CONTEXT_RECALL_BAR}"]
