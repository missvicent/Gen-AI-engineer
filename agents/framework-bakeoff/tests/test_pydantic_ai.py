"""The agent on Pydantic AI (bakeoff/run_pydantic_ai.py) — written before the runner.

Offline: `pai_model` (tests/conftest.py) is a FunctionModel that calls
read_note(note_id="2") once, then answers with what the tool returned.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent_tools.tools as notes

RUNNER = Path(__file__).parent.parent / "bakeoff" / "run_pydantic_ai.py"


def test_answers_with_one_tool_call(pai_model):
    from bakeoff.contract import TASK, ToolCall, used_the_tool
    from bakeoff.run_pydantic_ai import run_pydantic_ai

    r = run_pydantic_ai(TASK, notes.read_note, model=pai_model)

    assert r.framework == "pydantic_ai"
    assert "call dentist" in r.answer
    assert r.tool_calls == [ToolCall(name="read_note", args={"note_id": "2"})]
    assert used_the_tool(r)


def test_runner_has_no_future_annotations():
    """Pydantic AI reads type hints at run time; stringified hints break tools
    defined inside a function. (Parsed, so the docstring's mention doesn't count.)"""
    tree = ast.parse(RUNNER.read_text())
    future = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        for alias in node.names
    ]
    assert "annotations" not in future
