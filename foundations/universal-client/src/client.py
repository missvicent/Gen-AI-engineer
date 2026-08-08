"""TODO: build one client that talks to any provider behind a single call.

Goal: four capabilities across "gpt", "claude", "gemini", "local" (Ollama) and
"mlx", where adding a provider is a config change, not a new code path:

    complete(prompt)          -> str              one prompt, text back
    stream(prompt)            -> Iterator[str]    text deltas as they arrive
    call_tool(prompt, tools)  -> ToolCall | str   the model picks; YOU run it
    extract(prompt, schema)   -> dict             schema-valid JSON, checked

OpenAI-compatible endpoints (OpenAI, Google's Gemini endpoint, Ollama, MLX)
share ONE adapter — only base_url and model differ. Anthropic gets a thin
adapter with the same signatures. Build your SDK clients through
`_openai_client` / `_anthropic_client` so the offline tests can swap in fakes.

Fill in the TODOs. The tests in tests/test_client.py tell you the shape.
Reference: ../after/src/client.py (peek only when stuck).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Provider:
    base_url: str | None
    api_key: str
    model: str


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool request. Same shape whichever provider produced it."""

    name: str
    args: dict[str, Any]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# TODO 1: fill in the provider registry. OpenAI-compatible endpoints (OpenAI,
# Gemini's OpenAI endpoint, Ollama, MLX) share one adapter — only base_url and
# model differ. Keys: "gpt", "claude", "gemini", "local", "mlx".
PROVIDERS: dict[str, Provider] = {
    # "gpt": Provider(None, _env("OPENAI_API_KEY"), "gpt-5.5"),
    # "local": Provider("http://localhost:11434/v1", "ollama", "qwen3.5:9b"),
}


def _openai_client(p: Provider) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=p.base_url, api_key=p.api_key or "x")


def _anthropic_client(p: Provider) -> Any:
    from anthropic import Anthropic

    return Anthropic(api_key=p.api_key)


def complete(prompt: str, provider: str = "local", temperature: float = 0.2) -> str:
    """TODO 2: dispatch to the right adapter and return the text response.
    Anthropic replies with content blocks — concatenate the text ones."""
    raise NotImplementedError("build the universal client")


def stream(
    prompt: str, provider: str = "local", temperature: float = 0.2
) -> Iterator[str]:
    """TODO 3: yield text deltas. OpenAI wire: stream=True, skip chunks with no
    choices and None deltas. Anthropic: messages.stream(...) -> .text_stream."""
    raise NotImplementedError("''.join(stream(...)) should equal complete(...)")


def call_tool(
    prompt: str,
    tools: list[dict[str, Any]],
    provider: str = "local",
    temperature: float = 0.2,
) -> ToolCall | str:
    """TODO 4: offer provider-neutral tools ({name, description, parameters}),
    adapt them to each wire format, and normalize the reply: a ToolCall if the
    model picked a tool, its plain text if it didn't. Never run the tool here."""
    raise NotImplementedError("normalize both wire formats into ToolCall")


def extract(
    prompt: str,
    schema: dict[str, Any],
    provider: str = "local",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """TODO 5: structured output. OpenAI wire: response_format=json_schema.
    Anthropic: one forced tool whose input_schema IS the schema. Raise
    ValueError if the reply is missing any of schema["required"]."""
    raise NotImplementedError("schema in, checked dict out")
