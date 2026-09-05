"""A report nobody can diff is a report that catches nothing next month."""

from __future__ import annotations

import json

from bench.core import BenchRun, Row
from bench.report import table, to_json


def rows() -> list[Row]:
    return [
        Row(
            "mid",
            "middle-1",
            cases=4,
            ok=1,
            tokens_in=400,
            tokens_out=80,
            cost_usd=0.00208,
            p50_ms=90,
            max_ms=120,
            errors=["schema violation"] * 3,
        ),
        Row(
            "dear",
            "frontier-1",
            cases=4,
            ok=4,
            tokens_in=400,
            tokens_out=80,
            cost_usd=0.0044,
            p50_ms=800,
            max_ms=1500,
        ),
        Row(
            "local",
            "qwen3.5:9b",
            cases=4,
            ok=0,
            tokens_in=400,
            tokens_out=80,
            cost_usd=0.0,
            p50_ms=2200,
            max_ms=4000,
            errors=["schema violation"] * 4,
        ),
    ]


def run() -> BenchRun:
    return BenchRun(task="invoice-extraction", rows=rows())


def test_the_table_is_ranked_by_cost_per_success():
    lines = table(run()).splitlines()
    body = [line.split()[0] for line in lines[2:5]]
    assert body == ["dear", "mid", "local"]


def test_the_table_shows_a_dash_rather_than_infinity():
    assert "inf" not in table(run())
    assert "—" in table(run())


def test_the_table_reports_total_spend():
    assert "total spend: $0.006480" in table(run())


def test_json_carries_the_derived_numbers_and_the_ranking():
    payload = json.loads(to_json(run()))
    assert [r["candidate"] for r in payload["rows"]] == ["dear", "mid", "local"]
    assert payload["rows"][0]["success_rate"] == 1.0
    assert payload["rows"][1]["cost_per_success"] == 0.00208
    # A JSON `Infinity` is not valid JSON, so a total failure serialises as null.
    assert payload["rows"][2]["cost_per_success"] is None
    assert payload["total_cost_usd"] == 0.00648


def test_an_empty_run_still_renders():
    assert "candidate" in table(BenchRun(task="none", rows=[]))
