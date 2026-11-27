"""The agent on LangGraph.

Spec: tests/test_langgraph.py, tests/test_recovery.py (`make test`).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from bakeoff.contract import Result, ToolCall

load_dotenv()


def run_langgraph(
    query: str,
    lookup: Callable[..., dict],
    model: Any = None,
    checkpointer: Any = None,
) -> Result:
    if model is None:
        model = ChatOpenAI(
            base_url=os.getenv("BASE_URL"), model="qwen3.5:9b", api_key="ollama"
        )
    t = tool(lookup)
    bound = model.bind_tools([t])

    def agent(state: MessagesState) -> dict:
        that_reply = bound.invoke(state["messages"])
        return {"messages": [that_reply]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([t], handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    app = g.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final = app.invoke({"messages": [("user", query)]}, config)

    tool_calls = []
    for m in final["messages"]:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                tool_calls.append(ToolCall(name=c["name"], args=c["args"]))
    answer = final["messages"][-1].content

    return Result(answer=answer, tool_calls=tool_calls, framework="langgraph")
