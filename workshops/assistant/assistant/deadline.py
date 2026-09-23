"""TODO: when a request stops being worth doing.

Two ways that happens, and they deserve one answer rather than two:

**The budget ran out.** A request the caller will wait 30 seconds for should not
spend 90 seconds retrying. Without a shared budget, each layer enforces its own
timeout and they compose by *addition* — three retries of a 10-second call inside
a 60-second composer inside a 30-second request is a 30-second promise that takes
ninety. The arithmetic is never done by anyone, which is why it is always wrong.

**Nobody is listening.** The client disconnected, closed the tab, or hit its own
timeout ten seconds ago. Everything downstream — the model tokens, the retries,
the vector search — is now being paid for to produce an answer no one will read.
Under load this is the failure that compounds: the queue fills with work for
callers who left, which makes everyone slower, which makes more callers leave.

Both are the same question ("is this still worth doing?"), so they are one object
here. `Budget` carries an absolute monotonic deadline plus a `cancelled`
predicate, lives in a ContextVar, and is consulted at the seams: between pipeline
stages, before each retry, and before each streamed chunk.

Checked at seams, deliberately — not enforced by killing a thread. Python cannot
safely interrupt arbitrary code, and pretending otherwise produces a system that
leaks half-finished work. What it CAN do is refuse to start the next thing, which
turns an unbounded request into a bounded one and is most of the value.

Reference: ../../after/src/assistant/deadline.py.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


class Expired(Exception):
    """The request ran out of time, or the caller stopped waiting.

    Its own type because the handling is different from a failure: there is
    nobody to apologise to, and retrying is the exact wrong move."""

    def __init__(self, reason: str = "deadline exceeded") -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    """What is left of one request's time, and whether anyone still wants it."""

    #: Absolute, on the monotonic clock — not a duration. A duration would have to
    #: be decremented by every layer that used some, which is how a budget quietly
    #: becomes per-layer instead of per-request. None means unbounded.
    deadline: float | None = None
    #: Consulted, never awaited. The HTTP layer flips a flag from its own task when
    #: the client disconnects; everything below just reads a boolean, which keeps
    #: the pipeline synchronous and testable.
    cancelled: Callable[[], bool] = field(default=lambda: False)
    clock: Callable[[], float] = time.monotonic

    def remaining(self) -> float | None:
        """TODO 1: seconds left, or None when unbounded.

        None rather than a very large float, so a caller can tell "no limit" from
        "lots of time left" — those want different code paths. Never negative:
        somebody doing arithmetic on this should not have to check the sign.
        """
        raise NotImplementedError

    def expired(self) -> str | None:
        """TODO 2: the reason this request is over, or None.

        Report cancellation BEFORE the deadline. They are not the same incident:
        a deadline means the work was too slow and someone should look, a
        disconnect means the work was pointless and nobody should be paged.
        """
        raise NotImplementedError

    def check(self) -> None:
        """TODO 3: raise `Expired(reason)` if the request is over.

        Called at seams, so the exception unwinds from a place where nothing is
        half-done.
        """
        raise NotImplementedError


_BUDGET: ContextVar[Budget | None] = ContextVar("assistant_budget", default=None)


@contextmanager
def budget(
    seconds: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[Budget]:
    """TODO 4: install a budget for the duration of one request.

    Build a `Budget` whose `deadline` is `clock() + seconds` (or None when
    `seconds` is None), set it on the ContextVar, yield it, and RESET the token
    in a `finally`. The reset is the part that matters: worker threads are
    reused, and a budget that outlives its request expires the next one
    instantly — which looks exactly like an outage.

    A ContextVar rather than a parameter for the same reason the request id is
    one: the budget is relevant to six layers and interesting to none of them,
    and threading it through every signature makes every signature about
    timeouts.
    """
    raise NotImplementedError


def current() -> Budget:
    """The active budget, or a fresh unbounded one. Fresh rather than a shared
    module-level instance: a default everyone holds a reference to is a default
    somebody eventually mutates, and then every request in the process has a
    deadline nobody set."""
    return _BUDGET.get() or Budget()


def remaining() -> float | None:
    return current().remaining()


def check() -> None:
    current().check()


def expired() -> str | None:
    return current().expired()


def cancelled() -> bool:
    """Whether the caller has left, specifically — not whether the request is over.

    `expired()` collapses the two conditions into one string, which is the right
    shape for a seam that only has to stop. A blocking wait needs them apart:
    `resilience`'s wait owes the clock a `TimeoutError` with a particular message and
    owes a departed client a prompt `Expired`, and it cannot honour both contracts
    from one boolean."""
    return current().cancelled()


def capped(timeout: float | None) -> float | None:
    """TODO 5: a per-call timeout, shrunk to fit what the request has left.

    This is the whole reason the budget is shared. A 60-second composer timeout
    inside a request with 4 seconds left is a 4-second composer timeout, and the
    layer that owns the composer never had to know the request existed.

    Handle all four combinations: no budget (return `timeout` untouched), no
    policy timeout (return what is left), and both (the tighter one wins).
    """
    raise NotImplementedError
