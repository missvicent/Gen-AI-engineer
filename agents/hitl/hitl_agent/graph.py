"""Your scratch agent (4.1 loop + 4.2 tools) as a LangGraph graph, with a pause
before the irreversible tool.

    decide ──final──► END
       │ ──delete_note──► approve (interrupt) ──► run_tool ──► decide
       │ ──any other tool──────────────────────► run_tool ──► decide

One node per step, so a resume replays only `approve`, never the model call.
State is plain data (dicts, lists, str), so the checkpointer can save it.

Spec: tests/test_graph.py (`make test-integration`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict


class State(TypedDict, total=False):
    goal: str  # what the user asked; comes from the first invoke
    history: list[dict]  # finished steps: {"tool", "args", "result"}
    pending: dict | None  # the call decide proposed: {"tool", "args"}
    approved: bool  # the human's answer for `pending`
    answer: str  # final answer; when it exists, the run is done


def build_graph(
    decide: Callable[[str, list[dict]], Any],
    tools: dict[str, Callable[..., Any]],
    checkpointer: Any,
) -> Any:
    from agent_tools.tools import grant_approval
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    def decide_node(state: State) -> dict:
        goal = state["goal"]
        history = state.get("history", [])
        d = decide(goal, history)
        if d.is_final:
            return {"answer": d.answer}
        else:
            return {"pending": {"tool": d.tool, "args": d.args}}

    def approve_node(state: State) -> dict:
        pending = state["pending"]
        ok = interrupt(pending)
        if ok:
            grant_approval(pending["args"]["note_id"])
        return {"approved": ok}

    def run_tool_node(state: State) -> dict:
        history = state.get("history", [])
        pending = state["pending"]
        is_approved = state.get("approved")
        tool, args = pending["tool"], pending["args"]
        if tool == "delete_note" and not is_approved:
            result = "rejected by human"
        else:
            result = tools[tool](**args)
        return {"history": history + [{"tool": tool, "args": args, "result": result}]}

    def route(state: State) -> str:
        if "answer" in state:
            return END
        pending = state["pending"]
        tool = pending["tool"]
        if tool == "delete_note":
            return "approve"
        else:
            return "run_tool"

    g = StateGraph(State)
    g.add_node("decide", decide_node)
    g.set_entry_point("decide")
    g.add_conditional_edges("decide", route)
    g.add_node("approve", approve_node)
    g.add_node("run_tool", run_tool_node)
    g.add_edge("approve", "run_tool")
    g.add_edge("run_tool", "decide")
    return g.compile(checkpointer=checkpointer)
