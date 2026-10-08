"""Workshop 8 layer — caching an *agent's* answers, which is not like caching RAG.

A RAG answer is a pure function of a question and a corpus, so caching it is
arithmetic. An agent run is not pure: it sends messages, books calendar entries and
pauses for humans. Replay a cached answer for a request whose original run fired
`send_telegram` and you have either skipped a message the user expected, or — if you
cache the *request* rather than the answer — sent it twice.

So this module's real content is the refusal rules, not the dictionary:

  - a run that fired an **irreversible** tool is never cached;
  - a run **paused for approval** is never cached (it isn't an answer yet);
  - a run that **failed** is never cached;
  - and the key covers everything that changes the answer, including the day, for
    anything time-sensitive.

Caching is rung 2 of the Phase-8 ladder — behaviour-preserving *only if* you get
these rules right. Get them wrong and it is the most damaging rung, because the
failure is invisible: the assistant answers instantly and confidently with
something that is no longer true.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Clock = Callable[[], float]


def answer_key(goal: str, context: str = "", model: str = "any") -> str:
    """Hash everything that can change the answer.

    Note `context`: two identical questions against different memory or retrieved
    documents are different questions. Leaving it out is the bug that makes a cache
    look brilliant in testing and wrong in production.
    """
    parts = "\u0000".join([goal.strip().lower(), context, model])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def is_cacheable(result: Any, gated_tools_fired: list[str] | None = None) -> bool:
    """The whole safety argument of this module, in one predicate.

    Deliberately conservative: anything ambiguous is a no. A cache that misses too
    often costs money, which you can see on a graph. A cache that returns an answer
    it should not have costs trust, which you cannot.
    """
    if getattr(result, "pending", None) is not None:
        return False  # paused for a human — there is no answer yet
    if getattr(result, "fired_irreversible_tool_without_approval", False):
        return False
    if gated_tools_fired:
        return False  # the run had side effects; replaying it would skip them
    text = getattr(result, "text", "") or ""
    if not text or text.startswith("stopped:"):
        return False  # a run that hit the step cap is a failure, not an answer
    return True


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
        entry = self.entries.get(key)
        if entry is None or self.clock() - entry.stored_at > self.ttl_s:
            if entry is not None:
                del self.entries[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry.answer

    def offer(self, key: str, result: Any, gated_tools_fired: list[str] | None = None) -> bool:
        """Store the answer only if the rules allow it. Returns whether it landed.

        `offer` rather than `put` on purpose: the caller proposes, the policy
        decides. A `put` that silently accepts a side-effecting run is exactly the
        API mistake this layer exists to prevent.
        """
        if not is_cacheable(result, gated_tools_fired):
            self.refused += 1
            return False
        self.entries[key] = Entry(answer=result.text, stored_at=self.clock())
        return True

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
    """Look first, run second, offer third — the whole rung, in that order.

    `gated_tools_fired` is injected so the safety check can read the *trace* rather
    than trusting the agent's own report. Reusing the observability layer as the
    input to a caching decision is not a coincidence: you cannot safely cache what
    you cannot see.
    """
    key = answer_key(goal, context)
    hit = cache.get(key)
    if hit is not None:
        return CachedAnswer(text=hit, cached=True)
    result = run(goal)
    fired = gated_tools_fired(result) if gated_tools_fired else None
    stored = cache.offer(key, result, fired)
    return CachedAnswer(text=getattr(result, "text", "") or "", cached=False, stored=stored)
