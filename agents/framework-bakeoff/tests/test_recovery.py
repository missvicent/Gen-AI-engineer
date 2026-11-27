"""Recovery: the lookup fails on its first call and works on the second.

"Fails" means it RAISES (a flaky store), not that it returns {"error": ...}.
An error dict is just data, and the scripted model would retry on its own in any
framework, so that would measure nothing. A raise reaches the framework: by
default LangGraph's ToolNode and Pydantic AI both re-raise it and the run dies.
The runner has to turn the error into a message the model can react to.
"""

from __future__ import annotations

import agent_tools.tools as notes
import pytest


def flaky_lookup():
    """read_note that raises on its first call. Returns (tool, calls seen)."""
    calls: list[str] = []

    def read_note(note_id: str) -> dict:
        """Read one note by its id. Use when the user wants to see an existing note.

        Args:
            note_id: the note's id, e.g. "1".
        Returns {"text": ...} on success or {"error": ...} if the id is unknown.
        """
        calls.append(note_id)
        if len(calls) == 1:
            raise ConnectionError("note store unreachable, try again")
        return notes.read_note(note_id)

    return read_note, calls


@pytest.mark.parametrize("framework", ["langgraph", "pydantic_ai"])
def test_recovers_from_a_failing_first_lookup(framework, lc_model, pai_model):
    from bakeoff.contract import TASK, used_the_tool
    from bakeoff.run_langgraph import run_langgraph
    from bakeoff.run_pydantic_ai import run_pydantic_ai

    run, model = {
        "langgraph": (run_langgraph, lc_model),
        "pydantic_ai": (run_pydantic_ai, pai_model),
    }[framework]
    lookup, calls = flaky_lookup()

    r = run(TASK, lookup, model=model)

    assert calls == ["2", "2"]  # failed once, asked again, worked
    assert "call dentist" in r.answer
    assert used_the_tool(r)
    assert len(r.tool_calls) == 2  # the failed attempt is visible too
