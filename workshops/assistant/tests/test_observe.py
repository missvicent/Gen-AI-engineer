"""The agent under a span tree, asserted offline with no collector.

The point of these tests is not that OTel works. It is that the trace answers the
questions a latency number cannot: which tools ran, in what order, where the time
went, and whether an irreversible one fired.
"""
from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from assistant.agent import Step, run
from assistant.observe import (
    AGENT_APPROVED,
    AGENT_OUTCOME,
    AGENT_PAUSED,
    AGENT_PENDING_TOOL,
    AGENT_STEPS,
    TOOL_GATED,
    TOOL_NAME,
    duration_ms,
    failed_spans,
    gated_tool_calls,
    percentile,
    recorder,
    slowest_tool,
    time_by_tool,
    tool_calls,
    traced_registry,
    traced_run,
)
from assistant.tools import Tool


def registry(**overrides: Any) -> dict[str, Tool]:
    base = {
        "read_emails": Tool("read_emails", lambda **_: ["one mail"], False, "read"),
        "read_news": Tool("read_news", lambda **_: "a headline", False, "read"),
        "send_telegram": Tool("send_telegram", lambda **_: {"sent": True}, True, "gated"),
    }
    base.update(overrides)
    return base


def script(*steps: Step):
    """A deterministic 'brain': hand back the next scripted step each time."""
    remaining = list(steps)

    def decide(goal: str, state: list[Any]) -> Step:
        return remaining.pop(0) if remaining else Step("", {}, is_final=True, answer="done")

    return decide


@pytest.fixture
def rec():
    return recorder("assistant-test")


def test_wrapping_the_registry_traces_every_tool_without_editing_any_tool(rec):
    tools = traced_registry(registry(), rec.tracer)
    tools["read_emails"].fn()
    tools["read_news"].fn(url="x")

    assert tool_calls(rec.spans()) == ["read_emails", "read_news"]
    assert [s.name for s in rec.spans()] == ["tool.read_emails", "tool.read_news"]


def test_a_wrapped_tool_still_returns_what_it_returned_before(rec):
    tools = traced_registry(registry(), rec.tracer)
    assert tools["read_emails"].fn() == ["one mail"]
    assert tools["read_emails"].requires_approval is False


def test_a_failing_tool_is_marked_error_and_the_exception_still_propagates(rec):
    def boom(**_: Any) -> Any:
        raise ConnectionError("inbox unreachable")

    tools = traced_registry(registry(read_emails=Tool("read_emails", boom, False, "")), rec.tracer)
    with pytest.raises(ConnectionError):
        tools["read_emails"].fn()

    failures = failed_spans(rec.spans())
    assert len(failures) == 1
    assert failures[0].status.status_code is StatusCode.ERROR


def test_the_trace_records_which_gated_tools_actually_fired(rec):
    """A safety report, not a performance one — Phase 6 contained these, and now
    production has to prove which ones ran."""
    tools = traced_registry(registry(), rec.tracer)
    tools["read_emails"].fn()
    tools["send_telegram"].fn(chat_id="1", message="hi")

    assert gated_tool_calls(rec.spans()) == ["send_telegram"]
    gated = [s for s in rec.spans() if (s.attributes or {})[TOOL_NAME] == "send_telegram"]
    assert (gated[0].attributes or {})[TOOL_GATED] is True


def test_the_agent_run_gets_a_root_span_carrying_its_outcome(rec):
    result = traced_run(
        run,
        "summarise my inbox",
        rec.tracer,
        decide=script(Step("read_emails", {}), Step("", {}, is_final=True, answer="two mails")),
        registry=traced_registry(registry(), rec.tracer),
    )
    assert result.text == "two mails"

    root = rec.named("agent.run")[0]
    assert (root.attributes or {})[AGENT_PAUSED] is False
    assert (root.attributes or {})[AGENT_OUTCOME] == "completed"
    assert root.status.status_code is StatusCode.OK
    # The tool span is a child, so the tree shows the tool inside the run.
    tool_span = rec.named("tool.read_emails")[0]
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == root.context.span_id


def test_a_run_paused_for_approval_is_visible_in_the_trace(rec):
    """A run that stopped for a human and a run that hit the step cap look identical
    on a latency chart, and could not be more different in a review."""
    result = traced_run(
        run,
        "text my boss",
        rec.tracer,
        decide=script(Step("send_telegram", {"chat_id": "1", "message": "hi"})),
        registry=traced_registry(registry(), rec.tracer),
    )
    assert result.pending is not None

    root = rec.named("agent.run")[0]
    assert (root.attributes or {})[AGENT_PAUSED] is True
    assert (root.attributes or {})[AGENT_STEPS] == 1
    # The outcome and the tool it is waiting on are attributes, not archaeology.
    assert (root.attributes or {})[AGENT_OUTCOME] == "paused_for_approval"
    assert (root.attributes or {})[AGENT_PENDING_TOOL] == "send_telegram"
    # The gated tool never ran, so there is no span for it — containment, proven.
    assert gated_tool_calls(rec.spans()) == []


def test_the_root_span_records_which_approvals_the_run_started_with(rec):
    """The approval story lives on the trace: an auditor reading one span sees the
    grants that were live, not just that a gated tool happened to fire."""
    result = traced_run(
        run,
        "text my boss",
        rec.tracer,
        decide=script(
            Step("send_telegram", {"chat_id": "1", "message": "hi"}),
            Step("", {}, is_final=True, answer="sent"),
        ),
        approvals={"send_telegram": True, "delete_file": False},
        registry=traced_registry(registry(), rec.tracer),
    )
    assert result.pending is None

    root = rec.named("agent.run")[0]
    # Only the truthy grants count — a denied approval is not an approval.
    assert list((root.attributes or {})[AGENT_APPROVED]) == ["send_telegram"]
    assert (root.attributes or {})[AGENT_OUTCOME] == "completed"
    assert gated_tool_calls(rec.spans()) == ["send_telegram"]


def test_a_containment_breach_is_an_error_with_a_policy_violation_outcome(rec):
    """traced_run trusts the loop's own confession. If a gated tool fired without
    approval, the root span must say so loudly — ERROR status, explicit outcome —
    because this is the one trace a reviewer must never scroll past."""
    class Breach:
        audit = ["ran: send_telegram"]
        pending = None
        fired_irreversible_tool_without_approval = True

    traced_run(lambda goal, **_: Breach(), "text my boss", rec.tracer)

    root = rec.named("agent.run")[0]
    assert root.status.status_code is StatusCode.ERROR
    assert (root.attributes or {})[AGENT_OUTCOME] == "policy_violation"


def test_time_by_tool_finds_where_the_wall_clock_actually_went(rec):
    slow = Tool("read_news", lambda **_: _spin(), False, "")
    tools = traced_registry(registry(read_news=slow), rec.tracer)
    tools["read_emails"].fn()
    tools["read_news"].fn()

    totals = time_by_tool(rec.spans())
    assert set(totals) == {"read_emails", "read_news"}
    assert slowest_tool(rec.spans()) == "read_news"


def test_helpers_are_safe_on_an_empty_trace(rec):
    assert tool_calls(rec.spans()) == []
    assert slowest_tool(rec.spans()) is None
    assert time_by_tool(rec.spans()) == {}
    assert percentile([], 99) == 0.0


def test_percentile_is_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 100) == 40


def test_percentile_rank_ties_use_ceil_not_round():
    """Regression: round() rounds ties to even, so P50 of two samples would
    report the max. Nearest-rank is ceil(p/100 * n) - 1, full stop."""
    assert percentile([10, 20], 50) == 10
    assert percentile([10, 20, 30, 40, 50, 60], 50) == 30


def test_duration_is_read_off_the_span(rec):
    tools = traced_registry(registry(), rec.tracer)
    tools["read_emails"].fn()
    assert duration_ms(rec.spans()[0]) >= 0.0


def _spin() -> str:
    """Burn a measurable amount of wall clock without sleeping the suite."""
    total = 0
    for i in range(200_000):
        total += i
    return str(total)
