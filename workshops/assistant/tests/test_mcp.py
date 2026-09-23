"""Tools that arrive by discovery, and the gate they arrive behind.

Discovery is the feature: add a tool to the server, restart, the assistant can
use it, with no agent code change. The gate is what makes that feature something
you can deploy — a tool nobody reviewed, described by the party that wants it
called, is not a tool to run on a planner's say-so.
"""
from assistant.agent import Step, run
from assistant.mcp_client import extend_assistant, gate
from assistant.tools import REGISTRY


def _fake_invoker(name, kwargs):
    return {"tool": name, "args": kwargs, "result": "ok"}


def test_discovered_tools_are_added_by_discovery_not_hardcoding():
    discovered = [
        {"name": "get_habit_streak", "description": "read a habit streak"},
        {"name": "log_habit", "description": "log a habit"},
    ]
    merged = extend_assistant(REGISTRY, discovered, _fake_invoker)
    # existing tools still present; new ones discovered
    assert "read_emails" in merged
    assert "get_habit_streak" in merged
    assert merged["log_habit"].requires_approval is True


def test_assistant_can_call_a_discovered_tool():
    discovered = [{"name": "get_habit_streak", "description": "read streak"}]
    merged = extend_assistant(
        REGISTRY, discovered, _fake_invoker, readonly_allowlist=["get_habit_streak"]
    )
    script = [
        Step(tool="get_habit_streak", args={"name": "reading"}),
        Step(tool="", args={}, is_final=True, answer="streak is 5"),
    ]
    it = iter(script)
    res = run("check my streak", lambda g, s: next(it), registry=merged)
    assert res.text == "streak is 5"


def test_adding_a_server_tool_needs_no_agent_code_change():
    # simulate the server gaining a new tool between runs — assistant picks it up
    v1 = extend_assistant(REGISTRY, [{"name": "a", "description": "x"}], _fake_invoker)
    v2 = extend_assistant(REGISTRY, [{"name": "a", "description": "x"},
                                     {"name": "b", "description": "y"}], _fake_invoker)
    assert "b" not in v1 and "b" in v2  # no code change — just re-discovery


# --- the gate: default deny, and only local policy opens it ---------------------


def test_a_tool_that_says_nothing_about_itself_is_gated():
    """The defect this replaces was one word long: `spec.get("requires_approval",
    False)`. An absent key read as permission, so a server ungated every tool it
    published by not mentioning the subject. Discovery is precisely the path by
    which tools arrive unreviewed; unreviewed and unrestricted must not be the
    same state."""
    assert gate({"name": "delete_everything", "description": "no comment"}) is True


def test_a_server_claiming_read_only_is_still_gated():
    """The rule, stated as a test: annotations inform policy, they never grant
    authority. `readOnlyHint` is an assertion by the party that benefits from it
    being believed, and there is nothing on the other end verifying it."""
    claims_harmless = {
        "name": "wipe_database", "description": "totally safe",
        "read_only": True, "destructive": False,
    }
    assert gate(claims_harmless) is True
    assert gate(claims_harmless, readonly_allowlist=["something_else"]) is True


def test_only_a_local_allowlist_entry_ungates_a_tool():
    spec = {"name": "lookup_fact", "description": "read a fact", "read_only": True}
    assert gate(spec, readonly_allowlist=["lookup_fact"]) is False


def test_an_allowlist_entry_is_enough_even_with_no_annotations_at_all():
    """The operator reviewed it. A server that declines to describe its own tools
    does not get to overrule that in either direction."""
    assert gate({"name": "list_topics"}, readonly_allowlist=["list_topics"]) is False


def test_a_hint_can_tighten_the_gate_it_cannot_open_it():
    """The asymmetry is the design. A tool the operator allowlisted, which then
    announces itself as destructive, is gated — the operator was working from
    the description and the description changed. Server input can only ever add
    caution."""
    turned_nasty = {"name": "lookup_fact", "read_only": False, "destructive": True}
    assert gate(turned_nasty, readonly_allowlist=["lookup_fact"]) is True

    # ...and a tool that merely declines to call itself a read is gated too
    hedging = {"name": "lookup_fact", "read_only": False}
    assert gate(hedging, readonly_allowlist=["lookup_fact"]) is True


def test_the_allowlist_flows_from_settings_into_the_registry():
    discovered = [
        {"name": "lookup_fact", "description": "read", "read_only": True},
        {"name": "send_invoice", "description": "send", "read_only": True},
    ]
    merged = extend_assistant(
        REGISTRY, discovered, _fake_invoker, readonly_allowlist=["lookup_fact"]
    )
    assert merged["lookup_fact"].requires_approval is False
    assert merged["send_invoice"].requires_approval is True, (
        "an identical self-description ungated a tool the operator never reviewed"
    )
