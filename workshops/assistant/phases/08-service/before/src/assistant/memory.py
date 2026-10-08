"""Memory layer — the assistant stops waking up as a stranger.

Two capabilities, and the second is the one people skip:

  * **remember**, with a source and an expiry on every claim
  * **forget**, so a fact the user corrects is deleted rather than outranked

Everything is dependency-free and deterministic on purpose: this layer has to be
testable in the same fast tier as `evals.py`, and recall quality is measured *there*
(as golden-set rows) rather than asserted here. `phase5-memory/01-memory-types` is the
same design over Qdrant, and `04-memory-frameworks` swaps in Mem0 or LangMem behind the
identical method signatures.

Context assembly lives here too, because in a real assistant the two are one decision:
what you remembered is only useful if it survives the trip into a bounded window.
"""

from __future__ import annotations

import re
import time
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

    Returning `None` is the most important branch: most of what a user says is not
    worth remembering, and every stored sentence is a permanent tax on recall.
    """
    lowered = turn.lower()
    if re.search(r"\b(retry|always|never|use the|prefer to call|workaround)\b", lowered):
        return "procedural"
    if re.search(r"\b(failed|errored|broke|timed out|last (week|time)|yesterday)\b", lowered):
        return "episodic"
    subject = r"(name|timezone|manager|team|boss|email|address|role)"
    if re.search(rf"\b(i am|i'm|i work|i live|i prefer|call me|my {subject})\b", lowered):
        return "semantic"
    return None


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
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind!r}")
        if not text.strip():
            raise ValueError("refusing to remember an empty string")
        if not source.strip():
            raise ValueError("every memory needs a source — provenance is not optional")
        moment = time.time() if now is None else now
        # "default" means "whatever this kind of memory should live for"; None means
        # "until superseded". Conflating the two is how facts quietly become immortal.
        days = DEFAULT_TTL_DAYS[kind] if ttl_days == "default" else ttl_days
        self._next += 1
        memory_id = f"m{self._next}"
        self._rows[memory_id] = Memory(
            id=memory_id,
            kind=kind,
            text=text.strip(),
            source=source,
            written_at=moment,
            expires_at=None if days is None else moment + days * DAY_SECONDS,
        )
        return memory_id

    def remember(self, turn: str, *, source: str, now: float | None = None) -> str | None:
        """Store a turn only if it is worth storing. Returns the id, or None."""
        kind = classify(turn)
        if kind is None:
            return None
        return self.write(kind, turn, source=source, now=now)

    def correct(self, memory_id: str, text: str, *, source: str, now: float | None = None) -> str:
        """Replace a fact: the stale row is deleted, not merely outranked."""
        stale = self._rows.get(memory_id)
        if stale is None:
            raise KeyError(memory_id)
        self.forget(memory_id)
        return self.write(stale.kind, text, source=source, now=now)

    # ------------------------------------------------------------ recall path
    def recall(self, kind: Kind, query: str, k: int = 3, now: float | None = None) -> list[Memory]:
        moment = time.time() if now is None else now
        alive = [row for row in self._rows.values() if row.kind == kind and not row.expired(moment)]
        scored = [
            Memory(
                row.id,
                row.kind,
                row.text,
                row.source,
                row.written_at,
                row.expires_at,
                overlap(query, row.text),
            )
            for row in alive
        ]
        ranked = sorted(scored, key=lambda row: (-row.score, row.id))
        return [row for row in ranked if row.score > 0][:k]

    def all(self, kind: Kind | None = None) -> list[Memory]:
        rows = [row for row in self._rows.values() if kind is None or row.kind == kind]
        return sorted(rows, key=lambda row: row.id)

    def forget(self, memory_id: str) -> None:
        self._rows.pop(memory_id, None)

    def forget_all(self, kind: Kind | None = None) -> int:
        doomed = [row.id for row in self.all(kind)]
        for memory_id in doomed:
            self.forget(memory_id)
        return len(doomed)


# ----------------------------------------------------------------- assembly
def assemble(
    task: str,
    pinned: Sequence[Memory],
    candidates: Iterable[Memory],
    budget_tokens: int,
    count: Callable[[str], int] = count_tokens,
) -> Context:
    """Fill a bounded window: pins first, then by relevance, parking what will not fit."""
    if budget_tokens <= 0:
        raise ValueError("a context budget must be positive")
    kept = list(pinned)
    used = count(task) + sum(count(line.text) for line in kept)
    if used > budget_tokens:
        raise ValueError(f"task + pinned guardrails need {used} tokens, budget is {budget_tokens}")
    parked: list[Memory] = []
    pinned_ids = {line.id for line in kept}
    for line in candidates:
        if line.id in pinned_ids:
            continue
        cost = count(line.text)
        if used + cost > budget_tokens:
            parked.append(line)  # parked WHOLE: half a fact reads as a whole one
            continue
        kept.append(line)
        used += cost
    return Context(lines=kept, tokens=used, budget=budget_tokens, parked=parked)


def context_for(
    memory: AssistantMemory,
    task: str,
    *,
    guardrails: Sequence[str] = (),
    budget_tokens: int = 120,
    now: float | None = None,
) -> Context:
    """Everything worth knowing for this task, inside the budget, guardrails pinned."""
    pins = [
        Memory(f"pin{index}", "procedural", text, "policy", 0.0)
        for index, text in enumerate(guardrails)
    ]
    recalled: list[Memory] = []
    for kind in ("semantic", "procedural", "episodic"):
        recalled.extend(memory.recall(kind, task, k=3, now=now))
    recalled.sort(key=lambda row: -row.score)
    return assemble(task, pins, recalled, budget_tokens)


def audit(memory: AssistantMemory, now: float | None = None) -> str:
    """Rows per kind and how many have gone stale — expiry hides facts, not problems."""
    moment = time.time() if now is None else now
    lines = [f"user={memory.user}"]
    for kind in KINDS:
        rows = memory.all(kind)
        stale = sum(row.expired(moment) for row in rows)
        lines.append(f"  {kind:<11} {len(rows):>3} rows   expired={stale}")
    return "\n".join(lines)
