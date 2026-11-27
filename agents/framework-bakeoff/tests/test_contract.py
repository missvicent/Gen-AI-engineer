"""The shared contract (bakeoff/contract.py) — written before contract.py.

One result shape for all three frameworks, and one honest check that the agent
looked the fact up: used_the_tool() reads the tool calls, never the answer.
"""

from __future__ import annotations

import dataclasses

import pytest


def test_task_names_note_2():
    from bakeoff.contract import TASK

    assert TASK == "What does note 2 say? Use the tool."


def test_result_is_a_frozen_value():
    from bakeoff.contract import Result, ToolCall

    call = ToolCall(name="read_note", args={"note_id": "2"})
    r = Result(answer="call dentist", tool_calls=[call], framework="langgraph")

    assert r.answer == "call dentist"
    assert r.tool_calls == [ToolCall("read_note", {"note_id": "2"})]  # equal by value
    assert r.framework == "langgraph"
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.answer = "something else"
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.name = "delete_note"


def make(*calls, answer="Note 2 says: call dentist"):
    from bakeoff.contract import Result, ToolCall

    tool_calls = [ToolCall(name, args) for name, args in calls]
    return Result(answer=answer, tool_calls=tool_calls, framework="test")


def test_used_the_tool_for_read_note_2():
    from bakeoff.contract import used_the_tool

    assert used_the_tool(make(("read_note", {"note_id": "2"})))


def test_used_the_tool_when_the_right_call_is_among_others():
    from bakeoff.contract import used_the_tool

    r = make(("read_note", {"note_id": "1"}), ("read_note", {"note_id": "2"}))
    assert used_the_tool(r)


def test_right_answer_without_a_tool_call_does_not_count():
    """A model that guesses "call dentist" must come out False."""
    from bakeoff.contract import used_the_tool

    assert not used_the_tool(make())


@pytest.mark.parametrize(
    "name, args",
    [
        ("read_note", {"note_id": "1"}),  # wrong note
        ("read_note", {"note_id": 2}),  # int, not the string "2"
        ("read_note", {}),  # no args
        ("read_note", {"note_id": "2", "extra": "x"}),  # args must be exactly this
        ("delete_note", {"note_id": "2"}),  # wrong tool
    ],
)
def test_used_the_tool_is_false_for_any_other_call(name, args):
    from bakeoff.contract import used_the_tool

    assert not used_the_tool(make((name, args)))
