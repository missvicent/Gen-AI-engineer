"""Offline tests: the schema AND the extraction path, both without a live model.

A fake extractor stands in for the Instructor-patched client. That is the only
reason `extract_all` and `violation_rate` can be tested at all — and it is the
reason this test file fails in `before/` until those functions exist.
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from src.extract import (
    Invoice,
    compare_providers,
    extract_all,
    extract_invoice,
    field_accuracy,
    violation_rate,
)

GOOD = "Acme Corp billed us $4,231.50 on July 3rd 2026."
HOPELESS = "see attached"


class FakeExtractor:
    """Answers from a script; raises for the cases a small model would fumble."""

    def __init__(self, replies: dict[str, Invoice | Exception]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Invoice:
        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"]
        reply = self.replies[text]
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_invoice_schema_shape():
    inv = Invoice(vendor="Acme", date="2026-07-03", total=4231.50)
    assert inv.total == 4231.50
    assert inv.vendor == "Acme"


def test_invoice_rejects_bad_total():
    with pytest.raises(ValidationError):
        Invoice(vendor="Acme", date="2026-07-03", total="not a number")  # type: ignore[arg-type]


def test_invoice_rejects_a_negative_total():
    with pytest.raises(ValidationError):
        Invoice(vendor="Acme", date="2026-07-03", total=-1.0)


def test_extraction_asks_for_the_schema_not_for_json_in_prose():
    """The schema goes in `response_model`. If it only appears in the prompt, you
    have a suggestion rather than a constraint."""
    want = Invoice(vendor="Acme", date="2026-07-03", total=4231.50)
    fake = FakeExtractor({GOOD: want})
    got = extract_invoice(GOOD, client=fake)
    assert got == want
    assert fake.calls[0]["response_model"] is Invoice


def test_a_case_the_model_cannot_shape_is_counted_not_crashed():
    fake = FakeExtractor(
        {
            GOOD: Invoice(vendor="Acme", date="2026-07-03", total=4231.50),
            HOPELESS: ValueError("could not satisfy schema after retries"),
        }
    )
    invoices, violations = extract_all([GOOD, HOPELESS], client=fake)
    assert len(invoices) == 1
    assert len(violations) == 1
    assert violation_rate([GOOD, HOPELESS], violations) == 0.5


def test_violation_rate_is_zero_for_an_empty_run():
    assert violation_rate([], []) == 0.0


def test_a_clean_run_reports_no_violations():
    texts = [GOOD]
    fake = FakeExtractor({GOOD: Invoice(vendor="Acme", date="2026-07-03", total=1.0)})
    invoices, violations = extract_all(texts, client=fake)
    assert len(invoices) == 1
    assert violation_rate(texts, violations) == 0.0


# The shoot-out's own evidence: two providers, the same three invoices, one of
# them read wrong by the cheaper model in the way cheaper models actually fail —
# a valid object with a wrong number in it.
SECOND = "Globex, 14 Feb 2026, total 900."
THIRD = "Initech — 2026-01-09 — $12.00"
TEXTS = [GOOD, SECOND, THIRD]
TRUTH = [
    Invoice(vendor="Acme Corp", date="2026-07-03", total=4231.50),
    Invoice(vendor="Globex", date="2026-02-14", total=900.0),
    Invoice(vendor="Initech", date="2026-01-09", total=12.0),
]


def test_a_valid_object_can_still_be_wrong():
    """Why the shoot-out needs two numbers, not one.

    This run has a 0% violation rate — every reply satisfied the schema — and
    two of its three dates are wrong. A comparison that stopped at validity
    would have reported the two providers as identical.
    """
    sloppy = [
        Invoice(vendor="Acme Corp", date="July 3rd 2026", total=4231.50),
        Invoice(vendor="Globex", date="14/02/2026", total=900.0),
        Invoice(vendor="Initech", date="2026-01-09", total=12.0),
    ]
    scores = field_accuracy(sloppy, TRUTH)
    assert scores["vendor"] == 1.0
    assert scores["total"] == 1.0
    assert scores["date"] == pytest.approx(1 / 3)


def test_totals_compare_within_a_cent_not_bit_for_bit():
    """Otherwise the test measures float formatting rather than the model."""
    got = [Invoice(vendor="Acme Corp", date="2026-07-03", total=4231.499)]
    assert field_accuracy(got, TRUTH[:1])["total"] == 1.0


def test_no_expected_rows_is_no_evidence_rather_than_a_perfect_score():
    assert field_accuracy([], []) == {"vendor": 0.0, "date": 0.0, "total": 0.0}


def test_the_comparison_runs_over_any_two_clients():
    """Provider-parameterized on purpose: the exercise claims a hosted-vs-local
    comparison, and a procedure that needs two API keys to execute is one nobody
    can check. Point it at real clients and it is the same function."""
    frontier = FakeExtractor(dict(zip(TEXTS, TRUTH, strict=True)))
    laptop = FakeExtractor({
        GOOD: Invoice(vendor="Acme Corp", date="2026-07-03", total=4231.50),
        SECOND: Invoice(vendor="Globex", date="14/02/2026", total=90.0),
        THIRD: ValueError("could not satisfy schema after retries"),
    })
    rows = compare_providers(TEXTS, TRUTH, {"frontier": frontier, "laptop": laptop})

    assert rows["frontier"] == {
        "violation_rate": 0.0,
        "vendor": 1.0,
        "date": 1.0,
        "total": 1.0,
    }
    # The laptop dropped one row entirely and misread another, so accuracy is
    # scored against what was ASKED FOR, not against what came back — three
    # expected rows, two answers, one of them wrong twice.
    assert rows["laptop"]["violation_rate"] == pytest.approx(1 / 3)
    assert rows["laptop"]["date"] == pytest.approx(1 / 3)
    assert rows["laptop"]["total"] == pytest.approx(1 / 3)
