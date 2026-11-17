import os
from typing import Any

from agent_tools.spec import spec_from_function
from agent_tools.tools import delete_note, draft_reply, read_note
from dotenv import load_dotenv
from react_from_scratch.agent import Decision, run_agent
from src import call_tool

load_dotenv()

# Which universal-client provider drives the agent: "claude", "gpt", "local".
PROVIDER = "local"

# The model can call only these. grant_approval is NOT here: approval is app state.
TOOLS = {fn.__name__: fn for fn in (read_note, draft_reply, delete_note)}

# `make live BLANK=draft_reply` hides that tool's docstring from the model only.
BLANK = os.environ.get("BLANK", "")
TOOL_SPECS = [spec_from_function(fn, blank=name == BLANK) for name, fn in TOOLS.items()]

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
    prompt = render(goal, history)
    reply = call_tool(prompt, TOOL_SPECS, provider="local")
    if isinstance(reply, list):
        # The loop runs one tool per step; extra parallel calls are asked again
        # next turn, with this step's result in the history.
        call = reply[0]
        return Decision(is_final=False, tool=call.name, args=call.args)
    return Decision(is_final=True, answer=reply)


if __name__ == "__main__":
    goal = "What is 15% of Tokyo's population?"

    def logged_decide(goal: str, history: list[tuple[Decision, Any]]) -> Decision:
        # Print each step so the run can be written down (task 5).
        if history:
            print(f"   -> {str(history[-1][1])[:200]}")
        d = decide(goal, history)
        print(f"{len(history) + 1}. {d.tool}({d.args})" if not d.is_final else "final")
        return d

    print(run_agent(goal, TOOLS, with_retry(logged_decide, TOOL_SPECS), max_steps=6))
