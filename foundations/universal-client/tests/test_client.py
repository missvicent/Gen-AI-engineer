"""Offline tests: the SDK clients are swapped for fakes, so no API calls, no credits.

Each fake records the kwargs it was called with and returns a canned reply.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from src import ToolCall, call_tool, complete, extract, stream
from src import client as c

SPECS = [{"name": "calc", "description": "math", "parameters": {"type": "object"}}]


@pytest.fixture
def openai(monkeypatch):
    """Fake OpenAI client. Set `.reply` to the message it should return."""
    fake = NS(calls=[], reply=NS(content="hi", tool_calls=None))

    def create(**kw):
        fake.calls.append(kw)
        if kw.get("stream"):
            return iter([NS(choices=[]), NS(choices=[NS(delta=NS(content="hi"))])])
        return NS(choices=[NS(message=fake.reply)])

    client = NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr(c, "_openai_client", lambda p: client)
    return fake


@pytest.fixture
def anthropic(monkeypatch):
    """Fake Anthropic client. Set `.content` to the blocks it should return."""
    fake = NS(calls=[], content=[NS(type="text", text="hi")])

    def create(**kw):
        fake.calls.append(kw)
        return NS(content=fake.content)

    @contextmanager
    def stream_(**kw):
        fake.calls.append(kw)
        yield NS(text_stream=iter(["h", "i"]))  # a property in the real SDK

    client = NS(messages=NS(create=create, stream=stream_))
    monkeypatch.setattr(c, "_anthropic_client", lambda p: client)
    return fake


def test_gpt_gets_no_temperature(openai):
    complete("q", provider="gpt")
    call_tool("q", SPECS, provider="gpt")
    assert all("temperature" not in kw for kw in openai.calls)


def test_local_still_gets_temperature(openai):
    complete("q", provider="local", temperature=0.3)
    assert openai.calls[0]["temperature"] == 0.3


def test_complete_both_providers(openai, anthropic):
    assert complete("q", provider="gpt") == "hi"
    assert complete("q", provider="claude") == "hi"


def test_stream_openai_skips_empty_chunks(openai):
    assert list(stream("q", provider="gpt")) == ["hi"]


def test_call_tool_openai_keeps_every_call_and_parses_json(openai):
    fns = [NS(name="calc", arguments=f'{{"expression": "{e}"}}') for e in "12"]
    openai.reply = NS(content=None, tool_calls=[NS(function=f) for f in fns])
    assert call_tool("q", SPECS, provider="gpt") == [
        ToolCall("calc", {"expression": "1"}),
        ToolCall("calc", {"expression": "2"}),
    ]
    assert openai.calls[0]["tools"] == [{"type": "function", "function": SPECS[0]}]


def test_call_tool_claude_finds_tool_after_text(anthropic):
    anthropic.content = [
        NS(type="text", text="let me check"),
        NS(type="tool_use", name="calc", input={"expression": "1+1"}),
    ]
    assert call_tool("q", SPECS, provider="claude") == [
        ToolCall("calc", {"expression": "1+1"})
    ]
    sent = anthropic.calls[0]["tools"][0]
    assert sent == {
        "name": "calc",
        "description": "math",
        "input_schema": SPECS[0]["parameters"],
    }


def test_call_tool_text_reply(openai, anthropic):
    assert call_tool("q", SPECS, provider="gpt") == "hi"
    assert call_tool("q", SPECS, provider="claude") == "hi"


# Known bugs, still open in the tracker. strict=True: once you fix one,
# its test passes, pytest reports XPASS as a failure, and you delete the marker.


def test_stream_claude(anthropic):
    assert list(stream("q", provider="claude")) == ["h", "i"]


def test_extract_claude_sends_json_safe_schema(anthropic):
    import json

    anthropic.content = [NS(type="tool_use", name="extract", input={"name": "Ada"})]
    extract("q", {"name": {"type": "string"}}, provider="claude")
    json.dumps(anthropic.calls[0]["tools"])


def test_extract_openai_uses_json_schema(openai):
    openai.reply = NS(content='{"name": "Ada"}', tool_calls=None)
    extract("q", {"name": {"type": "string"}}, provider="gpt")
    assert openai.calls[0]["response_format"]["type"] == "json_schema"


def test_extract_sends_same_full_schema_to_both(openai, anthropic):
    props = {"name": {"type": "string"}}
    openai.reply = NS(content='{"name": "Ada"}', tool_calls=None)
    anthropic.content = [NS(type="tool_use", name="extract", input={"name": "Ada"})]
    extract("q", props, provider="gpt")
    extract("q", props, provider="claude")
    full = {"type": "object", "properties": props, "required": ["name"]}
    assert openai.calls[0]["response_format"]["json_schema"]["schema"] == full
    assert anthropic.calls[0]["tools"][0]["input_schema"] == full
