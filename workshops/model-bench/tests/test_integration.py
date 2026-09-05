"""The opt-in tier: the same bench, against a model that really answers.

    make test-integration        # needs `ollama serve` and `ollama pull qwen3.5:9b`

Everything the fast tier proves is structural. This tier proves the one thing a
fake cannot: that a real small model, handed a real messy invoice, produces
something your schema accepts. Expect it to be slower and occasionally worse than
you hoped — that number is the point of the exercise.
"""

from __future__ import annotations

import pytest
from bench.core import run_bench
from bench.providers import CANDIDATES, live_runner
from bench.report import table
from bench.tasks import prompts, validate_invoice

pytestmark = pytest.mark.integration


def test_a_local_model_extracts_an_invoice_the_schema_accepts():
    run = run_bench(
        candidates=[CANDIDATES["local"]],
        cases=prompts()[:2],
        runner=live_runner,
        validate=validate_invoice,
        task="invoice-extraction",
    )
    row = run.rows[0]
    print("\n" + table(run))
    assert row.cases == 2
    assert row.tokens_in > 0, "no usage came back — the meter is reading nothing"
    assert row.ok >= 1, f"the local model failed every case: {row.errors}"
    assert row.cost_usd == 0.0  # local tokens are free; the bench must say so


def test_latency_is_measured_not_guessed():
    run = run_bench(
        candidates=[CANDIDATES["local"]],
        cases=prompts()[:1],
        runner=live_runner,
        validate=validate_invoice,
    )
    assert run.rows[0].p50_ms > 1.0, "a real call cannot take under a millisecond"
