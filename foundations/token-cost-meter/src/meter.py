"""
Token metering utilities for large language models.

- `count_openai(text)`: Uses tiktoken ("o200k_base") to estimate token count for a
  given string before sending it to the model—a quick way to check input size.
- `cost(model, usage)`: Computes the dollar cost based on model pricing and usage
  details (number of input, cached, and output tokens). Cached input tokens are
  billed at a discounted (~10%) rate; output tokens cost more than input.

See tests in tests/test_meter.py for expected behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

PRICE: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "gpt-5.5": (5.00, 30.00),
}


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0


def count_openai(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def cost(model: str, usage: Usage) -> float:
    p_input, p_output = PRICE[model]
    cached = usage.cache_read_input_tokens
    fresh = usage.input_tokens - cached
    # cached input ~10% price
    dollars = (
        p_input * fresh + p_output * usage.output_tokens + p_input * cached * 0.1
    ) / 1_000_000
    return dollars


if __name__ == "__main__":
    n = count_openai("The taxi meter is always running.")
    print(f"Pre-flight token count: {n}")
    usage = Usage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=800,
    )
    print(f"Post-flight cost: ${cost('claude-sonnet-4-6', usage):.2f}")
