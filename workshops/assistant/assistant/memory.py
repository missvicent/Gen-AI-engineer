"""Memory layer — the assistant stops waking up as a stranger.

Two capabilities, and the second is the one people skip:

  * **remember**, with a source and an expiry on every claim
  * **forget**, so a fact the user corrects is deleted rather than outranked

Build the `forget` path in the same commit as the `write` path. "It remembered" is the
most convincing thing this assistant can do, and "it remembered something wrong and
would not let go" is the fastest way to lose that trust.

Everything stays dependency-free and deterministic so this layer runs in the same fast
tier as `evals.py`. Recall *quality* is not asserted here — it belongs in the eval suite
as golden-set rows, where a regression shows up as a number.

Reference: ../../../after/src/assistant/memory.py
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

Kind = str  # "working" | "episodic" | "semantic" | "procedural"
KINDS: tuple[Kind, ...] = ("working", "episodic", "semantic", "procedural")

DAY_SECONDS = 86_400
_WORD = re.compile(r"[a-z0-9']+")

#: Facts about a person go stale quietly, so semantic writes get a default lifetime.
DEFAULT_TTL_DAYS: dict[Kind, int | None] = {
    "working": 1,
    "episodic": 180,
    "semantic": 365,
    "procedural": None,  # a procedure is good until the tool changes
}


@dataclass(frozen=True)
class Memory:
    id: str
    kind: Kind
    text: str
    source: str
    written_at: float
    expires_at: float | None = None
    score: float = 0.0

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and self.expires_at < now


@dataclass(frozen=True)
class Context:
    """What goes into the prompt, plus the receipt for how the budget was spent."""

    lines: list[Memory]
    tokens: int
    budget: int
    parked: list[Memory]

    def render(self) -> str:
        return "\n".join(f"[{line.source}] {line.text}" for line in self.lines)

    def receipt(self) -> str:
        return (
            f"tokens {self.tokens}/{self.budget}   lines {len(self.lines)}   "
            f"parked {len(self.parked)}"
        )


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def count_tokens(text: str) -> int:
    """Deterministic stand-in. Swap for the Phase-1 meter when real money is involved."""
    return len(words(text))


def overlap(query: str, text: str) -> float:
    left, right = set(words(query)), set(words(text))
    if not left or not right:
        return 0.0
    return len(left & right) / (len(left | right) ** 0.5)


def classify(turn: str) -> Kind | None:
    """Which kind of memory (if any) this turn deserves.

    TODO: procedural = how to do the job ("always retry ... on a 429").
    TODO: episodic = something that happened ("the send step failed last week").
    TODO: semantic = a durable fact about the user ("I work in UTC-5").
    TODO: **return None for everything else** — this is the most important branch.
          Most turns are not worth remembering, and every stored sentence is a
          permanent tax on recall precision. Beware of matching a bare "my ": "what's
          on my calendar tomorrow?" is not a fact about the user.
    """
    raise NotImplementedError("classify() is yours to implement")


class AssistantMemory:
    """Four namespaces, one dict, and a `forget` that deletes."""

    def __init__(self, user: str = "me") -> None:
        self.user = user
        self._rows: dict[str, Memory] = {}
        self._next = 0

    # ------------------------------------------------------------- write path
    def write(
        self,
        kind: Kind,
        text: str,
        *,
        source: str,
        ttl_days: int | None | Literal["default"] = "default",
        now: float | None = None,
    ) -> str:
        """Store one claim and return its id.

        TODO: reject an unknown kind, an empty text, and a blank source. A memory with
              no provenance cannot be audited, so it does not get to exist.
        TODO: `now=None` means the real clock (`time.time()`; you add the import).
        TODO: `ttl_days="default"` means "whatever this kind should live for"
              (`DEFAULT_TTL_DAYS`); `None` means "until superseded". Conflating those
              two is how facts quietly become immortal.
        """
        raise NotImplementedError("write() is yours to implement")

    def remember(self, turn: str, *, source: str, now: float | None = None) -> str | None:
        """Store a turn only if it is worth storing. Returns the id, or None.

        TODO: `classify` first; `None` means write nothing and return None.
        """
        raise NotImplementedError("remember() is yours to implement")

    def correct(
        self, memory_id: str, text: str, *, source: str, now: float | None = None
    ) -> str:
        """Replace a fact: the stale row is deleted, not merely outranked.

        TODO: forget the old row, write the new one with the same kind, return its id.
              An unknown id is a `KeyError` — silently writing a new fact would leave
              the wrong one in place.
        """
        raise NotImplementedError("correct() is yours to implement")

    # ------------------------------------------------------------ recall path
    def recall(self, kind: Kind, query: str, k: int = 3, now: float | None = None) -> list[Memory]:
        """Top-k live memories of one kind, scored by `overlap`.

        TODO: skip expired rows, score the rest, drop zero-score rows, return k.
        """
        raise NotImplementedError("recall() is yours to implement")

    def all(self, kind: Kind | None = None) -> list[Memory]:
        """Everything stored, expired rows included — the audit view."""
        raise NotImplementedError("all() is yours to implement")

    def forget(self, memory_id: str) -> None:
        """Delete one row. Forgetting an unknown id is a no-op, not an error."""
        raise NotImplementedError("forget() is yours to implement")

    def forget_all(self, kind: Kind | None = None) -> int:
        """Drop a whole kind and return how many rows went."""
        raise NotImplementedError("forget_all() is yours to implement")


# ----------------------------------------------------------------- assembly
def assemble(
    task: str,
    pinned: Sequence[Memory],
    candidates: Iterable[Memory],
    budget_tokens: int,
    count: Callable[[str], int] = count_tokens,
) -> Context:
    """Fill a bounded window: pins first, then by relevance, parking what will not fit.

    TODO: refuse a non-positive budget.
    TODO: task + pinned guardrails go in first; if they alone exceed the budget, raise —
          an assistant that silently drops its own leash is worse than one that stops.
    TODO: a candidate that does not fit is parked WHOLE. Never truncate: half a fact
          reads as a whole fact to the model.
    """
    raise NotImplementedError("assemble() is yours to implement")


def context_for(
    memory: AssistantMemory,
    task: str,
    *,
    guardrails: Sequence[str] = (),
    budget_tokens: int = 120,
    now: float | None = None,
) -> Context:
    """Everything worth knowing for this task, inside the budget, guardrails pinned.

    TODO: turn each guardrail string into a pinned `Memory` with source "policy".
    TODO: recall from semantic, procedural and episodic, rank by score, then `assemble`.
    """
    raise NotImplementedError("context_for() is yours to implement")


def audit(memory: AssistantMemory, now: float | None = None) -> str:
    """Rows per kind and how many have gone stale — expiry hides facts, not problems.

    TODO: one line per kind: count, and how many are expired at `now`.
    """
    raise NotImplementedError("audit() is yours to implement")
