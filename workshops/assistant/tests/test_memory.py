"""The memory layer: provenance, expiry, correction, and a bounded window.

Offline and deterministic, like every other test in this workshop. Recall *quality* is
not asserted here — it belongs in the eval suite as golden-set rows, where a regression
shows up as a number instead of a red test nobody trusts.
"""

import pytest

from assistant.memory import (
    DAY_SECONDS,
    AssistantMemory,
    assemble,
    audit,
    classify,
    context_for,
    count_tokens,
)

NOW = 1_800_000_000.0
GUARDRAILS = ("never send email without approval",)


@pytest.fixture
def memory() -> AssistantMemory:
    store = AssistantMemory()
    store.write("semantic", "I work in UTC-5 and prefer meetings after 10:00",
                source="turn-3", now=NOW)
    store.write("procedural", "retry the calendar tool once on a 429", source="run-12", now=NOW)
    store.write("episodic", "the calendar tool failed when the event had no end time",
                source="run-12", now=NOW)
    return store


def test_a_remembered_fact_comes_back_with_its_source(memory: AssistantMemory) -> None:
    hits = memory.recall("semantic", "what timezone do I work in", now=NOW)
    assert hits and "UTC-5" in hits[0].text
    assert hits[0].source == "turn-3"


def test_kinds_stay_apart(memory: AssistantMemory) -> None:
    hits = memory.recall("semantic", "retry the calendar tool", now=NOW)
    assert all(hit.kind == "semantic" for hit in hits)


def test_a_memory_without_provenance_is_refused(memory: AssistantMemory) -> None:
    with pytest.raises(ValueError, match="source"):
        memory.write("semantic", "something true", source="", now=NOW)


def test_semantic_facts_expire_by_default(memory: AssistantMemory) -> None:
    """Facts about a person go stale quietly, so they get a lifetime unless told otherwise."""
    assert memory.recall("semantic", "when do I prefer meetings", now=NOW)
    two_years = NOW + 730 * DAY_SECONDS
    assert memory.recall("semantic", "when do I prefer meetings", now=two_years) == []
    assert memory.recall("procedural", "retry 429", now=two_years)  # no TTL by design


def test_a_corrected_fact_replaces_the_stale_one(memory: AssistantMemory) -> None:
    stale = memory.write("semantic", "Dana is my manager", source="turn-4", now=NOW)
    memory.correct(stale, "Ravi is my manager", source="turn-91", now=NOW)
    texts = [hit.text for hit in memory.recall("semantic", "who is my manager", now=NOW)]
    assert any("Ravi" in text for text in texts)
    assert not any("Dana" in text for text in texts)


def test_forget_all_clears_one_kind_only(memory: AssistantMemory) -> None:
    assert memory.forget_all("semantic") == 1
    assert memory.recall("semantic", "timezone", now=NOW) == []
    assert memory.recall("procedural", "retry 429", now=NOW)


@pytest.mark.parametrize(
    ("turn", "expected"),
    [
        ("I work in UTC-5", "semantic"),
        ("always retry the calendar tool once on a 429", "procedural"),
        ("the send step failed last week", "episodic"),
        ("what's on my calendar tomorrow?", None),  # most turns are not worth storing
    ],
)
def test_only_some_turns_are_worth_remembering(turn: str, expected: str | None) -> None:
    assert classify(turn) == expected


def test_remember_skips_the_turns_that_are_not_worth_it() -> None:
    store = AssistantMemory()
    assert store.remember("what's on my calendar tomorrow?", source="turn-1", now=NOW) is None
    assert store.remember("I prefer meetings after 10:00", source="turn-2", now=NOW) is not None
    assert len(store.all()) == 1


def test_the_context_never_exceeds_its_budget(memory: AssistantMemory) -> None:
    context = context_for(memory, "when should we meet", guardrails=GUARDRAILS,
                          budget_tokens=25, now=NOW)
    assert context.tokens <= 25


def test_guardrails_are_pinned_and_survive_pressure(memory: AssistantMemory) -> None:
    context = context_for(memory, "when should we meet", guardrails=GUARDRAILS,
                          budget_tokens=20, now=NOW)
    assert "never send email without approval" in context.render()


def test_pinning_more_than_the_budget_fails_loudly(memory: AssistantMemory) -> None:
    with pytest.raises(ValueError, match="budget"):
        context_for(memory, "when should we meet", guardrails=GUARDRAILS * 5,
                    budget_tokens=8, now=NOW)


def test_a_line_that_does_not_fit_is_parked_whole(memory: AssistantMemory) -> None:
    long_fact = memory.write("semantic", " ".join(["timezone"] * 40), source="turn-9", now=NOW)
    context = context_for(memory, "timezone", budget_tokens=30, now=NOW)
    assert long_fact in [line.id for line in context.parked]
    assert "timezone timezone timezone" not in context.render()


def test_every_context_line_carries_its_source(memory: AssistantMemory) -> None:
    context = context_for(memory, "when should we meet", guardrails=GUARDRAILS, now=NOW)
    assert all(f"[{line.source}]" in context.render() for line in context.lines)


def test_the_receipt_reports_what_was_spent(memory: AssistantMemory) -> None:
    context = context_for(memory, "when should we meet", budget_tokens=40, now=NOW)
    assert f"/{context.budget}" in context.receipt()


def test_expired_rows_stay_visible_to_the_audit(memory: AssistantMemory) -> None:
    two_years = NOW + 730 * DAY_SECONDS
    assert memory.recall("semantic", "timezone", now=two_years) == []
    assert "expired=1" in audit(memory, now=two_years)


def test_assemble_is_injectable_for_a_real_tokenizer(memory: AssistantMemory) -> None:
    """The counter is a parameter, so production can count in real tokens."""
    doubled = assemble("task", [], memory.all("semantic"), 200, count=lambda t: 2 * count_tokens(t))
    plain = assemble("task", [], memory.all("semantic"), 200)
    assert doubled.tokens == 2 * plain.tokens
