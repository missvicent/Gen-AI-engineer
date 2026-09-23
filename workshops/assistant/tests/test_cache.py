"""Caching an agent's answers — mostly a test suite about what must NOT be cached.

Every one of these refusals maps to a way a caching layer can quietly break an
assistant: skipping a message the user expected, answering a stale inbox summary, or
serving an answer from a run that never finished.
"""
from __future__ import annotations

import pytest

from assistant.agent import AgentResult, Pending
from assistant.cache import AnswerCache, answer_key, cached_run, is_cacheable


class Tick:
    """A clock we advance by hand, so TTL tests don't sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def answered(text: str = "two new mails") -> AgentResult:
    return AgentResult(text=text)


def test_the_key_covers_the_context_because_memory_changes_the_answer():
    assert answer_key("what did I miss", context="mem@v1") != answer_key(
        "what did I miss", context="mem@v2"
    )


def test_the_key_ignores_case_and_stray_whitespace():
    assert answer_key(" What Did I Miss ") == answer_key("what did i miss")


def test_a_completed_read_only_answer_is_cacheable():
    assert is_cacheable(answered())


def test_a_run_that_paused_for_approval_is_not_cacheable():
    """There is no answer yet — only a question waiting for a human."""
    assert not is_cacheable(AgentResult(pending=Pending("send_telegram", {})))


def test_a_run_that_fired_an_irreversible_tool_is_not_cacheable():
    """The one that matters. Replaying this answer skips the message the user
    expected to be sent — an invisible failure, and the worst kind."""
    assert not is_cacheable(answered(), gated_tools_fired=["send_telegram"])


def test_a_run_that_hit_the_step_cap_is_not_cacheable():
    assert not is_cacheable(AgentResult(text="stopped: max_steps"))


def test_an_empty_answer_is_not_cacheable():
    assert not is_cacheable(AgentResult(text=""))


def test_a_containment_breach_is_never_cached():
    breach = AgentResult(text="sent!", fired_irreversible_tool_without_approval=True)
    assert not is_cacheable(breach)


def test_offer_refuses_and_counts_rather_than_storing_silently():
    cache = AnswerCache(clock=Tick())
    stored = cache.offer("k", answered(), gated_tools_fired=["schedule_event"])
    assert stored is False
    assert cache.refused == 1
    assert cache.get("k") is None


def test_offer_stores_a_clean_answer():
    cache = AnswerCache(clock=Tick())
    assert cache.offer("k", answered("two new mails")) is True
    assert cache.get("k") == "two new mails"
    assert cache.hit_rate == 1.0


def test_a_stale_entry_expires_and_is_dropped():
    tick = Tick()
    cache = AnswerCache(ttl_s=120.0, clock=tick)
    cache.offer("k", answered())
    tick.now = 121.0
    assert cache.get("k") is None
    assert "k" not in cache.entries


def test_the_second_identical_request_never_reruns_the_agent():
    cache = AnswerCache(clock=Tick())
    calls: list[str] = []

    def run(goal: str) -> AgentResult:
        calls.append(goal)
        return answered("two new mails")

    first = cached_run("what did I miss", run, cache)
    second = cached_run("what did I miss", run, cache)

    assert (first.cached, first.stored) == (False, True)
    assert (second.cached, second.text) == (True, "two new mails")
    assert calls == ["what did I miss"]  # the agent ran once


def test_a_side_effecting_request_reruns_every_time_by_design():
    """The correct behaviour, not a missed optimization: 'text my boss' must reach
    the agent every time, because the point of the request is the side effect."""
    cache = AnswerCache(clock=Tick())
    calls: list[str] = []

    def run(goal: str) -> AgentResult:
        calls.append(goal)
        return answered("sent")

    for _ in range(3):
        out = cached_run("text my boss", run, cache, gated_tools_fired=lambda _: ["send_telegram"])
        assert out.cached is False
        assert out.stored is False

    assert len(calls) == 3
    assert cache.refused == 3


def test_the_safety_check_reads_the_trace_rather_than_trusting_the_agent():
    """`gated_tools_fired` is injected so the decision can come from spans. You
    cannot safely cache what you cannot see."""
    cache = AnswerCache(clock=Tick())
    seen: list[AgentResult] = []

    def from_trace(result: AgentResult) -> list[str]:
        seen.append(result)
        return []

    out = cached_run("what did I miss", lambda _: answered(), cache, gated_tools_fired=from_trace)
    assert out.stored is True
    assert len(seen) == 1


def test_hit_rate_is_zero_before_any_traffic():
    assert AnswerCache().hit_rate == pytest.approx(0.0)
