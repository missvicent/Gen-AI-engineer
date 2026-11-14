from typing import Any

from dotenv import load_dotenv
from react_from_scratch.agent import Decision

load_dotenv()

# decide() treats "plain text, no tool call" as the final answer, so the
# model must be told that rule on every turn.
INSTRUCTION = (
    "Call a tool if you need more information. When you know the answer, "
    "reply with the final answer as plain text, without calling a tool."
)


def render(goal: str, history: list[tuple[Decision, Any]]) -> str:
    lines = [f"Goal: {goal}", ""]
    if history:
        lines.append("Steps so far:")
    for i, (decision, result) in enumerate(history, start=1):
        lines.append(f"{i}. {decision.tool}({decision.args}) -> {result}")
    lines += ["", INSTRUCTION]
    return "\n".join(lines)


def decide(goal: str, history: list[tuple[Decision, Any]]) -> Decision:
    raise NotImplementedError
