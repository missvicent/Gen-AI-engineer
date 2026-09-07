"""Where the candidates are declared, and the one function that talks to a vendor.

Everything above this file is provider-agnostic. That is the whole point of the
adapter from lesson 01: the bench does not import `openai`, this module does.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from .core import Candidate, Reply, Usage

load_dotenv()

# Prices are $/MTok (input, output). Local models are free per token — the cost
# is your electricity and your patience, neither of which bills per request.
CANDIDATES: dict[str, Candidate] = {
    "local": Candidate("local", "qwen3.5:9b", price_in=0.0, price_out=0.0),
    "local-fast": Candidate(
        "local-fast", "qwen2.5:7b-instruct", price_in=0.0, price_out=0.0
    ),
    "local-big": Candidate("local-big", "qwen3-coder:30b", price_in=0.0, price_out=0.0),
    "gpt": Candidate("gpt", "gpt-5.2", price_in=1.75, price_out=14.00),
    "gpt-mini": Candidate("gpt-mini", "gpt-5.4-mini", price_in=0.25, price_out=2.00),
}

OLLAMA_BASE_URL = os.environ["BASE_URL"]


def resolve(names: list[str]) -> list[Candidate]:
    """Turn CLI names into candidates.

    Raises KeyError if any name is unknown.
    """
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise KeyError(
            f"Unknown candidate(s): {', '.join(unknown)} - ping from {list(CANDIDATES)}"
        )
    return [CANDIDATES[n] for n in names]


def live_runner(candidate: Candidate, prompt: str) -> Reply:
    """One call over an OpenAI-compatible endpoint. Ollama and OpenAI both are.

    `usage` comes straight off the response. if a vendor over omits it,
    the bench should use 0 for all costs rather than inventing them.
    """

    local = candidate.name.startswith("local")
    client = OpenAI(
        base_url=OLLAMA_BASE_URL if local else None,
        api_key=os.environ["OLLAMA_API_KEY"] if local else os.environ["OPENAI_API_KEY"],
    )
    response = client.chat.completions.create(
        model=candidate.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = response.usage
    cached = 0
    if usage is not None and usage.prompt_tokens_details is not None:
        cached = usage.prompt_tokens_details.cached_tokens or 0
    return Reply(
        text=response.choices[0].message.content or "",
        usage=Usage(
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
            cache_read_input_tokens=cached,
        ),
    )
