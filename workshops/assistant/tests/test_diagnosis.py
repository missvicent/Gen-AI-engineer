"""The "operate" drill: diagnose a seeded production failure from the spans alone.

The scenario every on-call rotation eventually meets: the assistant got slow, and
nobody changed the assistant. These tests plant the failure — one tool quietly
degraded, one tool intermittently failing — and then diagnose it WITHOUT reading
the tool code, using only what the trace recorded. If a diagnosis step below needs
anything other than the spans, the instrumentation is not finished.

The discipline being drilled:

    symptom  -> agent.run got slow          (duration on the root span)
    localise -> which child ate the time    (time_by_tool / slowest_tool)
    confirm  -> is it failing or just slow  (failed_spans on the same trace)

Run it against the live stack and the same readers work on the OTLP copy of these
spans in a collector — the drill is the runbook's "diagnose" step, rehearsed.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from assistant.agent import Step, run
from assistant.observe import (
    duration_ms,
    failed_spans,
    percentile,
    recorder,
    slowest_tool,
    time_by_tool,
    traced_registry,
    traced_run,
)
from assistant.tools import Tool

SEEDED_DELAY_S = 0.05  # the "degradation": read_news suddenly takes 50ms per call


def degraded_registry() -> dict[str, Tool]:
    """The seeded incident: read_news degraded, everything else healthy."""

    def slow_news(**_: Any) -> str:
        time.sleep(SEEDED_DELAY_S)
        return "a headline, eventually"

    return {
        "read_emails": Tool("read_emails", lambda **_: ["one mail"], False, "read"),
        "read_news": Tool("read_news", slow_news, False, "read"),
        "send_telegram": Tool("send_telegram", lambda **_: {"sent": True}, True, "gated"),
    }


def script(*steps: Step):
    remaining = list(steps)

    def decide(goal: str, state: list[Any]) -> Step:
        return remaining.pop(0) if remaining else Step("", {}, is_final=True, answer="done")

    return decide


@pytest.fixture
def rec():
    return recorder("assistant-drill")


def test_the_slow_tool_is_identified_from_the_spans_alone(rec):
    """Step 1+2 of the drill: the run is slow, and the trace names the culprit."""
    traced_run(
        run,
        "brief me",
        rec.tracer,
        decide=script(
            Step("read_emails", {}),
            Step("read_news", {}),
            Step("", {}, is_final=True, answer="briefed"),
        ),
        registry=traced_registry(degraded_registry(), rec.tracer),
    )

    root = rec.named("agent.run")[0]
    assert duration_ms(root) >= SEEDED_DELAY_S * 1000  # the symptom is on the root span

    # The diagnosis is a lookup, not a hunch: the degraded tool dominates the clock.
    totals = time_by_tool(rec.spans())
    assert slowest_tool(rec.spans()) == "read_news"
    assert totals["read_news"] > totals["read_emails"] * 5


def test_the_tail_shows_the_regression_before_the_average_does(rec):
    """Operate lesson: one degraded call in ten barely moves the mean but owns the
    P99 — which is why the CI gate budgets the tail, not the average."""
    tools = traced_registry(degraded_registry(), rec.tracer)
    for _ in range(9):
        tools["read_emails"].fn()
    tools["read_news"].fn()  # the one degraded call

    durations = [duration_ms(s) for s in rec.spans()]
    p50, p99 = percentile(durations, 50), percentile(durations, 99)
    assert p99 >= SEEDED_DELAY_S * 1000
    assert p99 > p50 * 5  # the median stays innocent; the tail confesses


def test_slow_and_failing_are_distinguished_on_the_same_trace(rec):
    """Step 3: 'slow' and 'broken' need different fixes, and the trace separates
    them without a redeploy — status on the failing span, duration on the slow one."""

    def flaky(**_: Any) -> Any:
        raise ConnectionError("news upstream reset the connection")

    seeded = degraded_registry()
    seeded["read_emails"] = Tool("read_emails", flaky, False, "read")
    tools = traced_registry(seeded, rec.tracer)

    with pytest.raises(ConnectionError):
        tools["read_emails"].fn()
    tools["read_news"].fn()

    failures = failed_spans(rec.spans())
    assert [str((s.attributes or {}).get("tool.name")) for s in failures] == ["read_emails"]
    # The slow tool is NOT in the failure list — it needs a capacity fix, not a retry.
    assert slowest_tool(rec.spans()) == "read_news"
