"""The bench, exercised with zero network and a fake clock.

Every provider in these tests is a dict lookup and every millisecond is a number
we chose. That is what makes the assertions about ranking and failure handling
meaningful — nothing here can pass or fail because a vendor was slow today.
"""

from __future__ import annotations

import json

import pytest
from bench.core import (
    Candidate,
    Reply,
    Usage,
    cost,
    percentile,
    rank,
    run_bench,
    run_case,
)

# Per case below every runner sends 100 input and 20 output tokens, so the
# per-call price is easy to reason about: MID is ~2x cheaper than DEAR.
DEAR = Candidate("dear", "frontier-1", price_in=5.00, price_out=30.00)  # 1100e-6 / call
MID = Candidate("mid", "middle-1", price_in=2.00, price_out=16.00)  # 520e-6 / call
FREE = Candidate("local", "qwen3.5:9b", price_in=0.0, price_out=0.0)

GOOD = '{"vendor": "Acme", "date": "2026-01-01", "total": 10.0}'
BAD = "Sure! Here is your invoice: vendor is Acme, total about ten dollars."


class FakeClock:
    """Advances a fixed amount every time it is read, so latency is scripted."""

    def __init__(self, step_s: float = 0.010) -> None:
        self.now = 0.0
        self.step = step_s

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def scripted(replies: dict[str, str | None], tokens: tuple[int, int] = (100, 20)):
    """A runner that answers from a table. `None` in the table means 'blow up'."""

    def runner(candidate: Candidate, prompt: str) -> Reply:
        text = replies[candidate.name]
        if text is None:
            raise TimeoutError("provider did not answer")
        return Reply(
            text=text, usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1])
        )

    return runner


def walking(scripts: dict[str, list[str]]):
    """A runner that walks each candidate's script, one entry per call."""
    calls: dict[str, int] = {}

    def runner(candidate: Candidate, prompt: str) -> Reply:
        i = calls.get(candidate.name, 0)
        calls[candidate.name] = i + 1
        return Reply(
            text=scripts[candidate.name][i],
            usage=Usage(input_tokens=100, output_tokens=20),
        )

    return runner


def is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def test_cost_comes_from_usage_not_from_an_estimate():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost(DEAR, usage) == pytest.approx(35.00)
    assert cost(FREE, usage) == 0.0


def test_a_cache_hit_bills_at_a_tenth_of_input():
    fresh = Usage(input_tokens=1_000_000, output_tokens=0)
    cached = Usage(
        input_tokens=1_000_000, output_tokens=0, cache_read_input_tokens=1_000_000
    )
    assert cost(DEAR, cached) == pytest.approx(cost(DEAR, fresh) * 0.1)


def test_percentile_is_nearest_rank_with_no_interpolation():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 100) == 40
    assert percentile([], 50) == 0.0


def test_percentile_rank_ties_use_ceil_not_round():
    """Regression: round() rounds ties to even, so P50 of two samples would
    report the max. Nearest-rank is ceil(p/100 * n) - 1, full stop."""
    assert percentile([10, 20], 50) == 10
    assert percentile([10, 20, 30, 40, 50, 60], 50) == 30


def test_a_dead_provider_becomes_a_row_not_a_crash():
    result = run_case(
        DEAR, "prompt", scripted({"dear": None}), is_json_object, clock=FakeClock()
    )
    assert result.ok is False
    assert result.error is not None
    assert "TimeoutError" in result.error
    assert result.cost_usd == 0.0


def test_a_shape_violation_is_a_failure_even_though_the_call_succeeded():
    result = run_case(
        DEAR, "p", scripted({"dear": BAD}), is_json_object, clock=FakeClock()
    )
    assert result.ok is False
    assert result.error == "schema violation"
    # It still cost money. That is exactly why `ok` and `cost_usd` are separate.
    assert result.cost_usd > 0


def test_cheaper_per_token_loses_when_it_cannot_hold_the_schema():
    """The headline lesson of the whole bench.

    MID is half DEAR's per-token price but only lands one answer in four. Ranked
    on price it wins; ranked on cost per *working* answer it is the worse buy.
    """
    run = run_bench(
        candidates=[MID, DEAR],
        cases=["a", "b", "c", "d"],
        runner=walking(
            {
                "mid": [GOOD, BAD, BAD, BAD],
                "dear": [GOOD, GOOD, GOOD, GOOD],
            }
        ),
        validate=is_json_object,
        clock=FakeClock(),
    )
    rows = {r.candidate: r for r in run.rows}
    assert rows["mid"].cost_usd < rows["dear"].cost_usd  # cheaper to run
    assert rows["mid"].success_rate == 0.25
    assert rows["dear"].success_rate == 1.0
    assert rows["mid"].cost_per_success > rows["dear"].cost_per_success  # worse to buy
    assert [r.candidate for r in rank(run)] == ["dear", "mid"]


def test_wasted_spend_shows_up_as_cost_per_success():
    """Half the calls failing doubles the price of every answer you can use."""
    run = run_bench(
        candidates=[MID],
        cases=["a", "b", "c", "d"],
        runner=walking({"mid": [GOOD, GOOD, BAD, BAD]}),
        validate=is_json_object,
        clock=FakeClock(),
    )
    row = run.rows[0]
    per_case = row.cost_usd / row.cases
    assert row.cost_per_success == pytest.approx(per_case * 2)


def test_a_candidate_that_never_succeeds_sinks_to_the_bottom():
    run = run_bench(
        candidates=[FREE, DEAR],
        cases=["a"],
        runner=scripted({"local": BAD, "dear": GOOD}),
        validate=is_json_object,
        clock=FakeClock(),
    )
    ranked = rank(run)
    # Free per token, but zero working answers: infinitely expensive per success.
    assert ranked[-1].candidate == "local"
    assert ranked[-1].cost_per_success == float("inf")
    assert ranked[0].candidate == "dear"


def test_rows_aggregate_tokens_and_latency_across_cases():
    run = run_bench(
        candidates=[DEAR],
        cases=["a", "b", "c"],
        runner=scripted({"dear": GOOD}, tokens=(100, 20)),
        validate=is_json_object,
        clock=FakeClock(step_s=0.010),
    )
    row = run.rows[0]
    assert (row.cases, row.ok) == (3, 3)
    assert (row.tokens_in, row.tokens_out) == (300, 60)
    assert row.p50_ms == pytest.approx(10.0)
    assert row.max_ms == pytest.approx(10.0)
    assert run.total_cost == pytest.approx(row.cost_usd)
