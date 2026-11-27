"""Shared fakes for the offline tests: no model, no Ollama, no network.

Both scripted models play the same part: call read_note(note_id="2") until a
tool result holding "call dentist" comes back, then answer with that result.
So they also cover recovery: a failed first call just makes them ask again.
"""

from __future__ import annotations

import agent_tools.tools as notes
import pytest


@pytest.fixture(autouse=True)
def fresh_notes(monkeypatch):
    monkeypatch.setattr(notes, "_NOTES", {"1": "buy milk", "2": "call dentist"})


@pytest.fixture
def lc_model():
    """A LangChain chat model for LangGraph. bind_tools() returns the model itself,
    so LangGraph's prebuilt agent (and a hand-built graph) can bind tools to it."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ScriptedChat(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            last = messages[-1]
            if isinstance(last, ToolMessage) and "call dentist" in str(last.content):
                msg = AIMessage(content=f"Note 2 says: {last.content}")
            else:
                call = {
                    "name": "read_note",
                    "args": {"note_id": "2"},
                    "id": f"call_{len(messages)}",
                }
                msg = AIMessage(content="", tool_calls=[call])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return ScriptedChat()


@pytest.fixture
def pai_model(monkeypatch):
    """A Pydantic AI FunctionModel. (TestModel would invent note_id="a".)
    Also blocks real model requests, so a runner that ignores `model` fails loudly
    instead of calling Ollama."""
    import pydantic_ai.models
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)

    def script(messages, info):
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart) and "call dentist" in str(part.content):
                return ModelResponse(parts=[TextPart(f"Note 2 says: {part.content}")])
        return ModelResponse(parts=[ToolCallPart("read_note", {"note_id": "2"})])

    return FunctionModel(script)
