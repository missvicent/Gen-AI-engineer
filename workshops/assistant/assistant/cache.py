"""TODO: Workshop 8 layer — caching an *agent's* answers, which is not like caching RAG.

A RAG answer is a pure function of a question and a corpus, so caching it is
arithmetic. An agent run is not pure: it sends messages, books calendar entries and
pauses for humans. Replay a cached answer for a request whose original run fired
`send_telegram` and you have skipped a message the user expected to be sent.

So the real content of this module is the **refusal rules**, not the dictionary:

  - a run that fired an irreversible tool is never cached;
  - a run paused for approval is never cached (it isn't an answer yet);
  - a run that failed or hit the step cap is never cached;
  - and the key covers everything that changes the answer, memory included.

Caching is rung 2 of the Phase-8 ladder — behaviour-preserving *only if* you get
these rules right. Get them wrong and it becomes the most damaging rung, because the
failure is invisible: the assistant answers instantly and confidently with something
that is no longer true.

Reference: ../../after/src/assistant/cache.py.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Clock = Callable[[], float]


def answer_key(goal: str, context: str = "", model: str = "any") -> str:
    """TODO 1: hash everything that can change the answer.

    `context` matters: two identical questions against different memory or retrieved
    documents are different questions. Leaving it out is the bug that makes a cache
    look brilliant in testing and wrong in production. Normalize case and whitespace
    on the goal so trivial differences still hit.
    """
    raise NotImplementedError


def is_cacheable(result: Any, gated_tools_fired: list[str] | None = None) -> bool:
    """TODO 2: the whole safety argument of this module, in one predicate.

    Refuse a paused run, a containment breach, a run that fired gated tools, an
    empty answer, and a `"stopped:"` result. Be deliberately conservative: a cache
    that misses too often costs money, which you can see on a graph. A cache that
    returns an answer it should not have costs trust, which you cannot.
    """
    raise NotImplementedError


@dataclass
class Entry:
    answer: str
    stored_at: float


@dataclass
class AnswerCache:
    """Exact-match answers with a TTL, and a hard rule about what may enter.

    The TTL is a staleness statement, not a performance setting. Pick it from how
    fast the underlying truth moves: an inbox summary goes stale in minutes, a
    policy answer in months.
    """

    ttl_s: float = 300.0
    clock: Clock = time.monotonic
    entries: dict[str, Entry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    refused: int = 0

    def get(self, key: str) -> str | None:
        """TODO 3: return a live answer or None, counting hits and misses and
        dropping an expired entry rather than re-checking it forever."""
        raise NotImplementedError

    def offer(self, key: str, result: Any, gated_tools_fired: list[str] | None = None) -> bool:
        """TODO 4: store only if `is_cacheable` allows it; return whether it landed.

        `offer` rather than `put` on purpose — the caller proposes, the policy
        decides. A `put` that silently accepts a side-effecting run is exactly the
        API mistake this layer exists to prevent. Count refusals.
        """
        raise NotImplementedError

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass(frozen=True)
class CachedAnswer:
    text: str
    cached: bool
    stored: bool = False


def cached_run(
    goal: str,
    run: Callable[[str], Any],
    cache: AnswerCache,
    context: str = "",
    gated_tools_fired: Callable[[Any], list[str]] | None = None,
) -> CachedAnswer:
    """TODO 5: look first, run second, offer third — in that order.

    `gated_tools_fired` is injected so the safety check can read the **trace** rather
    than trusting the agent's own report. Wiring the observability layer into the
    caching decision is not a coincidence: you cannot safely cache what you cannot
    see.
    """
    raise NotImplementedError
