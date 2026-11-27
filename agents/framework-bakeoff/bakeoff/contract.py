"""The shared contract: one result shape for all three frameworks, the task,
and the one honest check that the agent looked the fact up.

Spec: tests/test_contract.py (`make test`).
"""

from __future__ import annotations

from dataclasses import dataclass

# The answer, "call dentist", can't be guessed: only the tool knows it.
TASK = "What does note 2 say? Use the tool."


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, str]


@dataclass(frozen=True)
class Result:
    answer: str
    tool_calls: list[ToolCall]
    framework: str


def used_the_tool(result: Result) -> bool:
    return any(
        tool.name == "read_note" and tool.args == {"note_id": "2"}
        for tool in result.tool_calls
    )
