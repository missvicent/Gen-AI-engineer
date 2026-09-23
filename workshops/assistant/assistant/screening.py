"""Trust boundaries — everything that crosses one gets re-screened.

Two channels carry untrusted content into the assistant besides the user's own
message: documents coming back from retrieval, and content coming back from
read-only tools (including tools discovered over MCP). Both are classic
indirect-injection paths, so both are run through the same L1 guardrails
screen as user input before they become evidence.

"The same screen" is enforced by passing it in rather than by everyone importing
the same function: when `ASSISTANT_GUARD_MODEL` adds a model-in-the-loop second
opinion (`guard.py`), it has to apply to every channel or it applies to none —
and a filter that only covers the channel the user types into is the channel an
attacker will not use.
"""
from __future__ import annotations

from dataclasses import replace

from assistant import guardrails
from assistant.guardrails import Screen
from assistant.rag import Chunk
from assistant.tools import Tool


def harden_registry(
    registry: dict[str, Tool], screen: Screen = guardrails.screen
) -> dict[str, Tool]:
    """TODO 1: re-screen the OUTPUT of every read-only tool with the injected
    `screen` — a fetched email/page is exactly where an indirect injection
    arrives, and MCP-discovered tools take this path too. Leave the irreversible
    (requires_approval) tools alone; the agent's approval gate covers those.
    Return a NEW registry; do not mutate the originals — and build the replacement
    with `tools.rewrap`, not `Tool(...)`: a wrapper that quietly drops
    `required_args` un-selects the tool from the planner without failing anything.

    Use the `screen` ARGUMENT, not `guardrails.screen` directly. The default
    makes the plain case read the same; the parameter is what lets the guard
    model cover this channel too."""
    raise NotImplementedError


def screen_contexts(docs: list[str], screen: Screen = guardrails.screen) -> list[str]:
    """Retrieved documents are the classic indirect-injection channel — a poisoned
    page lands in the corpus, retrieval hands it straight to the composer. Every
    document is re-screened before it becomes evidence: an injection-bearing doc
    is DROPPED from the context entirely, PII in a clean one is redacted."""
    kept = []
    for doc in docs:
        ok, cleaned = screen(str(doc))
        if ok:
            kept.append(cleaned)
    return kept


def screen_chunks(chunks: list[Chunk], screen: Screen = guardrails.screen) -> list[Chunk]:
    """The same gate, over retrieved chunks, keeping their provenance.

    A redaction rewrites the text and nothing else: the source, version and
    offsets still describe where the text came from, so a citation on a redacted
    chunk points at the real document rather than at an anonymous fragment. The
    offsets now describe the ORIGINAL span, which is the honest reading — this is
    that part of that document, with something removed."""
    kept = []
    for chunk in chunks:
        ok, cleaned = screen(chunk.text)
        if ok:
            kept.append(replace(chunk, text=cleaned))
    return kept
