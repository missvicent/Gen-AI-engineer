"""The agent on Pydantic AI.

No `from __future__ import annotations` in this file, on purpose: Pydantic AI
reads type hints at run time, and stringified hints resolve against module
globals, where a function-local import isn't. tests/test_pydantic_ai.py checks it.

Spec: tests/test_pydantic_ai.py, tests/test_recovery.py (`make test`).
"""

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from bakeoff.contract import Result, ToolCall

load_dotenv()


def run_pydantic_ai(
    query: str, lookup: Callable[..., dict], model: Any = None
) -> Result:
    if model is None:
        provider = OpenAIProvider(
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=os.getenv("BASE_URL"),
        )
        model = OpenAIChatModel("qwen3.5:9b", provider=provider)

    def read_note(note_id: str) -> dict:
        """Read one note by its id, e.g. "1". Returns {"text": ...} or {"error": ...}"""
        try:
            return lookup(note_id)
        except Exception as e:
            # A crash would end the run; ModelRetry sends the error back to the
            # model as "try again" instead.
            raise ModelRetry(str(e)) from e

    agent = Agent[object, str](model, tools=[read_note])
    result = agent.run_sync(query)

    tools = [
        ToolCall(name=p.tool_name, args=p.args_as_dict())
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]

    return Result(answer=result.output, tool_calls=tools, framework="pydantic_ai")
