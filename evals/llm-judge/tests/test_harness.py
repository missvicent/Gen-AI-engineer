"""The fast tier: the whole harness, no model, no network.

The fake judges here are not a shortcut — they are how you test grading logic at
all. A test whose expected value depends on what a model felt like saying is not a
test.
"""

from pathlib import Path

from llm_judge.harness import (
    Row,
    ScoredRow,
    format_table,
    is_abstention,
    load_golden,
    mean_scores,
    run_suite,
    score_row,
)

GOLDEN = Path(__file__).parents[2] / "golden-set" / "evals" / "golden.jsonl"


class FixedJudge:
    """Returns the same verdict for everything. Deterministic by construction."""

    def __init__(self, value: float = 0.9) -> None:
        self.value = value
        self.calls = 0

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        self.calls += 1
        return self.value

    def context_recall(
        self, question: str, contexts: list[str], reference: str
    ) -> float:
        self.calls += 1
        return self.value

    def describe(self) -> dict[str, str]:
        return {"judge_model": "fixed-fake", "judge_temperature": "0"}


class ExplodingJudge(FixedJudge):
    """Fails loudly if it is called at all — proves a row stayed judge-free."""

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        raise AssertionError("the judge must not be called for this row")

    def context_recall(
        self, question: str, contexts: list[str], reference: str
    ) -> float:
        raise AssertionError("the judge must not be called for this row")


def good_pipeline(question: str) -> tuple[str, list[str]]:
    rows = {r.question: r for r in load_golden(GOLDEN)}
    row = rows[question]
    if row.expects_abstention:
        return "Not in the documents.", []
    return row.ground_truth, [row.ground_truth]


def test_golden_set_loads_with_slices_and_abstention_flags():
    rows = load_golden(GOLDEN)
    assert len(rows) >= 20
    assert {r.slice for r in rows} >= {"semantic", "exact", "unanswerable"}
    assert sum(r.expects_abstention for r in rows) >= 5


def test_abstention_rows_never_reach_the_judge():
    row = next(r for r in load_golden(GOLDEN) if r.expects_abstention)
    scored = score_row(row, "Not in the documents.", [], ExplodingJudge())
    assert scored.judged is False
    assert scored.scores["faithfulness"] == 1.0


def test_an_invented_answer_fails_an_abstention_row():
    row = next(r for r in load_golden(GOLDEN) if r.expects_abstention)
    scored = score_row(row, "The cap is 40,000 EUR per supplier.", [], ExplodingJudge())
    assert scored.scores["faithfulness"] == 0.0


def test_answerable_rows_are_judged():
    row = next(r for r in load_golden(GOLDEN) if not r.expects_abstention)
    judge = FixedJudge(0.75)
    scored = score_row(row, row.ground_truth, [row.ground_truth], judge)
    assert scored.judged is True
    assert scored.scores == {"faithfulness": 0.75, "context_recall": 0.75}
    assert judge.calls == 2


def test_slice_breakdown_exposes_what_the_average_hides():
    rows = load_golden(GOLDEN)

    def broken_abstain(question: str) -> tuple[str, list[str]]:
        row = {r.question: r for r in rows}[question]
        if row.expects_abstention:
            return "The answer is definitely 40,000 EUR.", []  # invents an answer
        return row.ground_truth, [row.ground_truth]

    result = run_suite(rows, broken_abstain, FixedJudge(1.0))
    assert result.by_slice["unanswerable"]["faithfulness"] == 0.0
    assert result.by_slice["semantic"]["faithfulness"] == 1.0
    # The overall average stays comfortably high while a whole slice is at zero:
    assert result.overall["faithfulness"] > 0.75


def test_instrument_is_recorded_with_the_scores():
    result = run_suite(load_golden(GOLDEN)[:3], good_pipeline, FixedJudge())
    assert result.instrument["judge_model"] == "fixed-fake"
    assert '"judge_model": "fixed-fake"' in result.to_json()


def test_mean_of_nothing_is_zero_not_a_crash():
    assert mean_scores([]) == {"faithfulness": 0.0, "context_recall": 0.0}


def test_abstention_detection_covers_the_phrasings_we_ship():
    assert is_abstention("Not in the documents.")
    assert is_abstention("I don't know — the corpus doesn't cover it.")
    assert not is_abstention("The refund window is five business days.")


def test_table_shows_every_slice_and_the_judged_count():
    rows = load_golden(GOLDEN)
    table = format_table(run_suite(rows, good_pipeline, FixedJudge()))
    for name in {r.slice for r in rows}:
        assert name in table
    assert "judged rows:" in table


def test_a_pipeline_that_retrieves_nothing_is_visible_per_slice():
    rows = [r for r in load_golden(GOLDEN) if not r.expects_abstention][:4]
    result = run_suite(rows, lambda q: ("I don't know", []), FixedJudge(0.1))
    assert result.overall["context_recall"] == 0.1


def test_row_equality_is_structural():
    a = Row(id="x", slice="semantic", question="q", ground_truth="g")
    b = Row(id="x", slice="semantic", question="q", ground_truth="g")
    assert a == b


def test_mean_accepts_a_generator_and_counts_abstention_rows():
    rows = [
        ScoredRow(
            "a", "unanswerable", {"faithfulness": 1.0, "context_recall": 1.0}, False
        ),
        ScoredRow("b", "semantic", {"faithfulness": 0.5, "context_recall": 0.0}, True),
    ]
    assert mean_scores(r for r in rows) == {"faithfulness": 0.75, "context_recall": 0.5}
    assert mean_scores(r for r in []) == {"faithfulness": 0.0, "context_recall": 0.0}
