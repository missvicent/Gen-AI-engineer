"""
This module implements a universal LLM client that supports multiple providers (OpenAI, Claude, Gemini, Ollama, MLX) 
via a unified interface. It normalizes outputs—especially tool calls—so downstream code works identically across providers. 
The client enforces schema validation for extraction, flagging incomplete/invalid model responses.
Core concepts include the 'prompt', optional 'temperature', models, tools, and schema-driven extraction.
Configuration determines provider, not code changes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


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



# Gemini's OpenAI endpoint, Ollama, MLX) share one adapter — only base_url and
# model differ. Keys: "gpt", "claude", "gemini", "local", "mlx".
PROVIDERS: dict[str, Provider] = {
    "gpt": Provider(None, _env("OPENAI_API_KEY"), "gpt-5.5"),
    "local": Provider(_env("BASE_URL"), _env("OLLAMA_API_KEY"), "qwen3.5:9b"),
    "claude": Provider(None, _env("ANTHROPIC_API_KEY"), "claude-sonnet-4-6"),
    "gemini": Provider(None, _env("GEMINI_API_KEY"), "gemini-2.5-flash"),
    "mlx": Provider(None, _env("MLX_API_KEY"), "mlx-01-flash"),  
}


def _openai_client(p: Provider) -> Any:

    return OpenAI(base_url=p.base_url, api_key=p.api_key or "x")


def _anthropic_client(p: Provider) -> Any:

    return Anthropic(api_key=p.api_key)


def complete(prompt: str, provider: str = "local", temperature: float = 0.2) -> str:
    p = PROVIDERS[provider]
    if provider == "claude":
        resp = _anthropic_client(p).messages.create(
            model=p.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[ {"role": "user", "content": prompt} ]
        )
        return "".join(block.text for block in resp.content if block.type == "text")
    else:
        resp = _openai_client(p).chat.completions.create(
            model=p.model,
            temperature=temperature,
            messages=[ {"role": "user", "content": prompt} ]
        )
        return resp.choices[0].message.content or ""


def stream(
    prompt: str, provider: str = "local", temperature: float = 0.2
) -> Iterator[str]:
    """skip chunks with no choices and None deltas"""
    p = PROVIDERS[provider]
    if provider == 'claude':
        with _anthropic_client(p).messages.stream(
            model=p.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            yield from stream.text_stream()
    else:
        chunks = _openai_client(p).chat.completions.create(
            model=p.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

def call_tool(
    prompt: str,
    tools: list[dict[str, Any]],
    p: str = "local",
    temperature: float = 0.2,
) -> ToolCall | str:
    """offer provider-neutral tools ({name, description, parameters}),
    adapt them to each wire format, and normalize the reply: a ToolCall if the
    model picked a tool, its plain text if it didn't."""
    p = PROVIDERS[p]
    if p == "claude":
        resp = _anthropic_client(p).messages.create(
            model=p.model,
            max_tokens=1024,
            temperature=temperature,
            tools=[
                {
                    "type": tool["type"], 
                    "name": tool["name"], 
                    "description": tool["description"], 
                    "parameters": tool["parameters"]
                } for tool in tools
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        for block in resp.content:
            if block.type == "tool_use":

                return ToolCall(name=block.name, args=dict(block.input))
            return "".join(block.text for block in resp.content if block.type == "text")
    else:
        resp = _openai_client(p).chat.completions.create(
            model=p.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            tools= [
                {
                    "type": "function",
                    "function": tool
                } for tool in tools
            ],
            tool_choice="auto"
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return ToolCall(name=call.function.name, args=dict(call.function.arguments))
        return msg.content or ""

def extract(
    prompt: str,
    schema: dict[str, Any],
    provider: str = "local",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """structured output. OpenAI wire: response_format=json_schema.
    Anthropic: one forced tool whose input_schema IS the schema. Raise
    ValueError if the reply is missing any of schema["required"]."""
    p = PROVIDERS[provider]
    if p == "claude":
        resp = _anthropic_client(p).messages.create(
            model=p.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {"name": "extract", "description": "Return the extraction.", 
                "input_schema": {
                    "type": "object",
                    "properties": schema,
                    "required": schema.keys(),
                }}
            ],
            tool_choice={"type": "tool", "name": "extract"},
        )
        data = next(dict(b.input) for b in resp.content if b.type == "tool_use")
    else:
        resp = _openai_client(p).chat.completions.create(
            model=p.model,
            temperature=temperature,
            response_format={
                "type": "json_object",
                "json_schema": {"name": "extraction", "schema": schema},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    missing = [key for key in schema.keys() if key not in data or data[key] is None]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    return data

def __main__():
    for chunk in stream("if the shih tzu the most popular dog in the world, how many shih tzus are there in the world?"):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    __main__()