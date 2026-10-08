"""What a request consumed, and what that costs.

One meter, used twice: `core.py` puts these numbers on the compose span, and
`report.py` totals them for the CI cost gate. Two meters would be two answers to
"what did last week cost", and the interesting week is always the one where they
disagree.

The price table is `crew.PRICE` — the same tiers the Phase-5 crew routes across —
rather than a second copy here. `local` is (0.0, 0.0) and that is the honest
number for a self-hosted model: no per-token invoice exists. It is not a licence
to stop counting. Tokens are counted regardless, so pointing the composer at a
paid API and setting `ASSISTANT_PRICE_TIER` turns the same measurement into a
bill the merge gate can block on, with no new instrumentation.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from assistant.crew import PRICE
from assistant.memory import count_tokens

#: How a Usage was arrived at. Reported wherever the number is, because the two
#: are not interchangeable: `counted` is what the provider will invoice, and
#: `estimated` is a word count standing in for a tokenizer nobody ran.
Source = Literal["counted", "estimated"]


@dataclass(frozen=True)
class Usage:
    tokens_in: int
    tokens_out: int
    source: Source = "estimated"

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    def cost(self, tier: str) -> float:
        """USD for this exchange at that tier's prices. An unknown tier costs
        nothing, which is the safe direction for a REPORT — a made-up price is
        worse than a missing one — and the reason `settings.price_tier` is
        documented next to the tiers that exist."""
        price_in, price_out = PRICE.get(tier, (0.0, 0.0))
        return round((self.tokens_in * price_in + self.tokens_out * price_out) / 1e6, 6)


#: What the provider said the last exchange cost, if it said anything.
#:
#: A ContextVar rather than a return value because the composer is a `Callable
#: [..., str]` — one seam, swapped between the offline composer, Ollama and a
#: hosted API — and widening that signature to carry token counts would force
#: every composer to have an opinion about billing. The adapter that talks to a
#: provider drops the provider's own numbers here; `measure` picks them up if
#: they are there. Context-local, so concurrent requests cannot read each
#: other's.
_REPORTED: ContextVar[Usage | None] = ContextVar("reported_usage", default=None)


def report(tokens_in: int, tokens_out: int) -> None:
    """Called by an adapter with the counts the provider returned."""
    _REPORTED.set(Usage(tokens_in=tokens_in, tokens_out=tokens_out, source="counted"))


def take_reported() -> Usage | None:
    """Read and clear. Clearing matters: a stale count is worse than no count,
    because the next exchange would be billed for the previous one."""
    reported = _REPORTED.get()
    _REPORTED.set(None)
    return reported


def adopt(reported: Usage | None) -> None:
    """Take on counts a producer read out of another thread's context.

    Context-local is the right default and a dead end at a thread boundary: a
    worker starts with an empty context and nothing it writes comes back. The
    batch path solves that generically, by copying its context back on success
    (`resilience.resilient`). A streaming worker cannot — it is still running
    when the first chunk is consumed, and copying a context mid-flight would
    adopt whatever half-finished state happened to be in it.

    So the streaming producer hands its counts over explicitly, on the same queue
    that carries the chunks, and the consuming thread adopts them here. `None`
    does nothing, which is what an unreported exchange means: the estimate stands.
    """
    if reported is not None:
        _REPORTED.set(reported)


#: The result of the most recent `measure`, for a caller downstream of the one
#: that metered. `report.py` scores answers that `core.py` already metered; it
#: used to re-measure them from a prompt it rebuilt itself, which was a second
#: opinion nobody wanted and — once the provider's real counts arrived — a
#: worse one.
_LAST: ContextVar[Usage | None] = ContextVar("last_usage", default=None)


def take_last() -> Usage | None:
    """Read and clear, for the same reason as `take_reported`: an exchange that
    never reached a model (an abstention) must not inherit the previous one."""
    last = _LAST.get()
    _LAST.set(None)
    return last


def measure(prompt: str, completion: str) -> Usage:
    """Tokens in and out for one model exchange — the provider's numbers if it
    reported any, otherwise the estimate.

    Ollama returns `prompt_eval_count` and `eval_count` on every completion and
    the adapter forwards them, so the deployed tier is billed on real tokens
    rather than on words. When nothing was reported — the offline composer, a
    provider that stays quiet — `count_tokens` stands in, and the Usage says so
    rather than letting an approximation be quoted as a measurement.

    `count_tokens` is the same approximation the context budget uses (memory.py).
    Sharing it means the budget and the bill are denominated in the same unit; a
    real tokenizer swaps in there and both follow."""
    reported = take_reported()
    used = reported or Usage(tokens_in=count_tokens(prompt), tokens_out=count_tokens(completion))
    _LAST.set(used)
    return used
