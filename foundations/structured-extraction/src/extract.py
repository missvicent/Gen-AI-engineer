"""TODO: extract a typed Invoice from messy text using structured outputs.

1) build_client(provider): Instructor-wrap an OpenAI-compatible client
   ('local' -> Ollama JSON mode; 'gpt' -> OpenAI) and return its
   `.chat.completions`.
2) extract_invoice(text, client): return an Invoice via response_model — no regex.
   Take the client as a parameter so this is testable without a live model.
3) extract_all / violation_rate: run a batch and report what refused to parse.
4) field_accuracy / compare_providers: whether the values are RIGHT, per field,
   for each provider. The violation rate alone cannot decide the shoot-out —
   constrained decoding takes every provider to roughly zero, so the difference
   you would tier a model on lives entirely in the accuracy columns.

The Invoice model is given. Note the constraints live in the type (`min_length`,
`ge`) rather than in the prompt — on a strict provider the decoder enforces them.

Reference: ../after/src/extract.py.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

LOCAL_MODEL = "qwen3.5:9b"


class Invoice(BaseModel):
    vendor: str = Field(min_length=1)
    date: str = Field(description="ISO date, e.g. 2026-07-16")
    total: float = Field(ge=0)


class Extractor(Protocol):
    """The one method Instructor-patched clients expose that we depend on."""

    def create(self, **kwargs: Any) -> Any: ...


def build_client(provider: str = "local") -> Extractor:
    if provider == "local":
        base = OpenAI(base_url=os.getenv("BASE_URL"), api_key="ollama")
        patched = instructor.from_openai(base, mode=instructor.Mode.JSON)
    else:
        patched = instructor.from_openai(OpenAI())
    return patched.chat.completions


def extract_invoice(
    text: str,
    client: Extractor | None = None,
    provider: str = "local",
    model: str = LOCAL_MODEL,
) -> Invoice:
    extractor = client or build_client(provider)
    return extractor.create(
        model=model,
        response_model=Invoice,
        messages=[{"role": "user", "content": text}],
    )


def extract_all(
    texts: list[str],
    client: Extractor | None = None,
    provider: str = "local",
    model: str = LOCAL_MODEL,
) -> tuple[list[Invoice], list[str]]:
    invoices: list[Invoice] = []
    violations: list[str] = []
    for text in texts:
        try:
            invoice = extract_invoice(text, client, provider, model)
            invoices.append(invoice)
        except Exception as e:
            violations.append(str(e))
    return invoices, violations


def violation_rate(texts: list[str], violations: list[str]) -> float:
    return len(violations) / len(texts) if texts else 0.0


def field_accuracy(
    got: Sequence[Invoice | None], expected: Sequence[Invoice]
) -> dict[str, float]:
    """share correct per field — the other half of the shoot-out.

    One entry per field of `Invoice`, not one average: they fail differently.
    `vendor` is copied out of the text, `total` has to be read off the right
    line, and `date` has to be reformatted, which is where a small model loses.
    An average hides the column you would have tiered on.

    Compare totals within a cent. A float that came back through JSON will not be
    bit-identical, and scoring `4231.5` against `4231.50` as a miss measures the
    parser rather than the model. Everything else compares case-insensitively
    after a strip. No expected rows means no evidence: return zeros, not a
    division by zero and not a cheerful 1.0.

    `got` is positional and carries `None` where the provider produced nothing —
    score those as misses. A shorter list instead of a hole would slide every
    later answer onto the wrong expected row and mark a model wrong about
    invoices it read correctly.
    """
    # This computes, for each field of the Invoice, what fraction of the expected
    # values were correctly produced by the model. It:
    # - Gets all field names from the Invoice class.
    # - If there are no expected rows, returns 0.0 for each field.
    # - Otherwise, for each field:
    #   - Counts the number of times the model's value (`have`) matches the
    #     expected value (`want`) — ignoring case/strip for strings, tolerance
    #     for total as in _same().
    #   - Divides the count by the number of expected rows for per-field accuracy.
    # - Returns a dict like: {'vendor': 0.9, 'date': 0.7, 'total': 0.8}
    fields = list(Invoice.model_fields)
    if not expected:
        return dict.fromkeys(fields, 0.0)
    scores = {}
    for field in fields:
        hits = sum(
            1
            for want, have in zip(expected, got, strict=False)
            if have is not None and _same(getattr(want, field), getattr(have, field))
        )
        scores[field] = hits / len(expected)
    return scores


def _same(want: Any, have: Any) -> bool:
    if isinstance(want, float) and isinstance(have, float):
        return abs(want - have) < 0.01
    return str(want).strip().casefold() == str(have).strip().casefold()


def compare_providers(
    texts: list[str],
    expected: list[Invoice],
    clients: dict[str, Extractor],
    model: str = "qwen3.5:9b",
) -> dict[str, dict[str, float]]:
    """Compare the performance of multiple providers on a list of texts.
    Take the clients as a dict rather than provider names, so the same procedure
    runs against fakes offline and against `build_client("gpt")` /
    `build_client("local")` for the real shoot-out. A comparison you can only run
    with two API keys is one you cannot test, and an untested comparison is how a
    tiering decision ends up resting on a number nobody re-derived.

    Extract one text at a time and append `None` for the ones that raise, so the
    answers stay lined up with `expected`; `extract_all`'s shorter list is the
    right shape for a rate and the wrong one for an accuracy.
    """
    rows = {}
    for name, client in clients.items():
        got: list[Invoice | None] = []
        violations: list[str] = []
        for text in texts:
            try:
                got.append(extract_invoice(text, client, model=model))
            except (ValidationError, ValueError) as e:
                got.append(None)
                violations.append(f"{text[:40]}... {type(e).__name__}")
        rows[name] = {
            "violation_rate": violation_rate(texts, violations),
            **field_accuracy(got, expected),
        }
    return rows


if __name__ == "__main__":
    inv = extract_invoice("Acme Corp billed us $4,231.50 on July 3rd 2026.")
    print(inv.model_dump())
