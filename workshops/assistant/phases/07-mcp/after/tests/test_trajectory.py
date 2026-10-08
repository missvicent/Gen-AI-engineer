"""The eval layer, proving the RAG layer. Offline, deterministic, no model."""

from assistant.agent import Step, run
from assistant.evals import (
    containment_ok,
    goal_completion,
    tool_choice_f1,
    tools_run,
)
from assistant.tools import REGISTRY

GATED = {name for name, tool in REGISTRY.items() if tool.requires_approval}


def scripted(*steps: Step):
    it = iter(steps)
    return lambda goal, state: next(it)


def test_the_audit_trail_reads_back_as_a_trajectory():
    result = run(
        "morning brief",
        scripted(
            Step(tool="read_emails", args={"limit": 1}),
            Step(tool="read_news", args={"url": "x"}),
            Step(tool="", args={}, is_final=True, answer="done"),
        ),
    )
    assert tools_run(result.audit) == ["read_emails", "read_news"]


def test_tool_choice_f1_flags_a_run_that_used_the_wrong_tool():
    assert tool_choice_f1(["read_emails", "read_news"], ["read_emails", "read_news"]) == 1.0
    assert tool_choice_f1(["read_emails", "delete_note"], ["read_emails", "read_news"]) == 0.5
    assert tool_choice_f1([], []) == 1.0


def test_goal_completion_reads_receipts_off_the_final_answer():
    answer = "You have 1 mail from Ana; the standup moved to 10:00."
    assert goal_completion(answer, ["1 mail", "10:00"]) == 1.0
    assert goal_completion(answer, ["1 mail", "budget approved"]) == 0.5


def test_a_gated_tool_never_fires_without_approval_even_when_the_brain_insists():
    """The trajectory check that must stay green through the hardening workshop:
    a 'brain' that keeps demanding the gated tool gets a pause, not a firing."""
    result = run(
        "message the team",
        scripted(Step(tool="send_telegram", args={"chat_id": "1", "message": "hi"})),
    )
    assert result.pending is not None
    assert containment_ok(result.audit, GATED)


def test_containment_passes_an_approved_firing_and_names_nothing():
    result = run(
        "message the team",
        scripted(
            Step(tool="send_telegram", args={"chat_id": "1", "message": "hi"}),
            Step(tool="", args={}, is_final=True, answer="sent"),
        ),
        approvals={"send_telegram": True},
    )
    assert containment_ok(result.audit, GATED, approvals={"send_telegram"})
    assert not containment_ok(result.audit, GATED)  # same trace, no approval on file
