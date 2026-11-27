"""The agent on CrewAI. Live only: there is no offline test for it.

Spec: tests/test_crewai.py (`make live`, asks first).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from crewai import LLM, Agent
from crewai.tools import tool
from dotenv import load_dotenv

from bakeoff.contract import Result, ToolCall

load_dotenv()


def run_crewai(query: str, lookup: Callable[..., dict], model: Any = None) -> Result:
    if model is None:
        model = LLM(
            model="openai/qwen3.5:9b",
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=os.getenv("BASE_URL"),
        )

    calls: list[ToolCall] = []  # CrewAI keeps no tool-call history: record our own

    def read_note(note_id: str) -> dict:
        """Read one note by its id, e.g. "1". Returns {"text": ...} or {"error": ...}"""
        calls.append(ToolCall(name="read_note", args={"note_id": note_id}))
        return lookup(note_id)

    t = tool("read_note")(read_note)

    agent = Agent(
        role="Assistant",
        goal="Answer the user's question.",
        backstory="You answer questions using the tools you have.",
        tools=[t],
        llm=model,
    )

    out = agent.kickoff(query)
    return Result(answer=out.raw, tool_calls=calls, framework="crewai")
