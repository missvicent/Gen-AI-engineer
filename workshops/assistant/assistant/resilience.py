"""Reliability primitives — stdlib only, injectable clocks, no magic.

Three small pieces the service composes around its network adapters:

- resilient(fn, policy)  : retries with exponential backoff + full jitter, and a
                           per-call timeout enforced from a worker thread
- TokenBucket            : the classic rate limiter — capacity is the burst, the
                           refill rate is the sustained requests-per-second
- ConcurrencyCap         : an admission counter — over the cap means REJECT, not
                           queue, because a queued request still holds the caller

Everything takes an injectable clock/sleep so the tests run in microseconds and
never actually wait.

## TODO: what is worth retrying

`retry on Exception` is the default everyone writes and it is wrong in two
directions at once. It retries a `TypeError` three times — the bug is
deterministic, so all it buys is three times the latency before the same 500 —
and it retries a 400 from an API that will reject the same payload forever. Worse,
each pointless attempt holds a connection and a worker while the real incident is
somewhere else.

So a failure has to be CLASSIFIED before it is retried, and a call that is not
idempotent has to be able to say so. Reference:
../../after/src/assistant/resilience.py.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

#: TODO 1: the failures where the same call, later, could plausibly succeed.
#: Keep it honest and short — the network, the clock, and the server saying
#: "later". `ConnectionError` and the socket errors are `OSError` subclasses, so
#: this is shorter than it looks; include `TimeoutError`, because our own timeout
#: raises it. Everything NOT in here is a bug or a refusal, and both should
#: surface on the first attempt, where the stack trace still points at the cause.
TRANSIENT: tuple[type[BaseException], ...] = ()


class Permanent(Exception):
    """A refusal that will be a refusal next time too.

    An adapter that can read a status code should raise this for a 4xx rather
    than letting a generic error be retried three times against a server that
    has already given its final answer."""


def is_transient(exc: BaseException) -> bool:
    """TODO 2: the default classifier.

    True only for `TRANSIENT` failures. `Permanent` must win over the type check,
    so an adapter can mark a `ConnectionError`-shaped failure as final without
    inventing a new exception hierarchy — and `deadline.Expired` (import it) is
    never transient, because the whole point of a deadline is that waiting
    longer is not an option.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class Policy:
    """How hard to try before giving up. attempts counts the FIRST call too."""

    attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0
    timeout: float | None = 10.0
    #: Which failures deserve another attempt. Injectable because "transient" is
    #: protocol-specific — an HTTP adapter knows 429 and 503 are worth retrying
    #: and 422 is not, and that knowledge belongs to the adapter, not here.
    retry_if: Callable[[BaseException], bool] = is_transient


DEFAULT_POLICY = Policy()

#: TODO 3: the policy for calls that are NOT idempotent — one attempt, still
#: bounded by the timeout. Retrying is safe when the call is idempotent;
#: `send_telegram` is not, because a timeout tells you nothing about whether the
#: server acted. Making that a NAMED policy rather than a comment is the
#: difference between a rule and a hope.
ONCE = Policy()


def resilient(
    fn: Callable[..., Any],
    policy: Policy = DEFAULT_POLICY,
    *,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> Callable[..., Any]:
    """Wrap a callable in retries + backoff + timeout. The LAST failure always
    propagates — resilience means trying again, not pretending it worked.

    Backoff is exponential with FULL jitter (delay drawn uniformly from
    [0, base * 2^attempt]) — the variant that empties a thundering herd fastest.
    The timeout runs the call on a worker thread; a call that overruns is
    abandoned there (Python cannot kill it) and reported as TimeoutError here.

    TODO 4: three things should stop this loop, and only one of them is "it
    worked". Make the other two real:

      - the failure is not transient (`policy.retry_if`), so it surfaces on the
        attempt that produced it rather than on the third identical retry;
      - the REQUEST's budget is gone. `deadline.check()` before each attempt, so
        no call is started that we already know we cannot wait for; and
        `deadline.capped(policy.timeout)` instead of `policy.timeout`, so a call
        made with four seconds of request left gets four seconds. Skip the
        backoff sleep when it would run past the deadline — sleeping through the
        rest of the budget in order to attempt a call that cannot finish is the
        most expensive possible way to fail.

    Then the harder half of the same idea, and the one a green test suite hid for
    two rounds: `future.result(timeout=...)` answers exactly one question, *has the
    clock run out*, and a thread parked in it can answer no other. It cannot notice
    that the caller hung up, because noticing requires looking, and this looks once.
    A buffered `/ask` whose client disappeared kept a model generating for the full
    budget and threw the answer away.

    So wait in slices instead — `POLL_SECONDS` at a time up to `limit` — and between
    slices ask `deadline.cancelled()`. If it is true, raise
    `deadline.Expired("caller disconnected")` and abandon the worker exactly as the
    timeout path does. Two details are load bearing:

      - `cancelled()`, not `expired()`. Clock exhaustion must still raise
        `TimeoutError` with the message it has now, because "too slow" is an alert
        and "they left" is not — `api.expired` turns that into 504 versus 499;
      - subtract the SLICE you waited, not `POLL_SECONDS`. Otherwise a 0.05s timeout
        is charged a 0.25s tick and the tight guard-model bound stops being a bound.

    A seam here only pays off if the work itself stops: `adapters.joined` closes the
    provider's stream, and `fallbacks.not_a_fallback` keeps the `Expired` from being
    laundered into a fake model outage. All three, or none.
    """

    def timed(*args: Any, **kwargs: Any) -> Any:
        if policy.timeout is None:
            return fn(*args, **kwargs)
        # NOT a `with` block: the context manager's __exit__ is shutdown(wait=True),
        # which would block right here until the abandoned call finished — turning
        # the timeout into a lie. (Caught live: a stalled model generation held
        # /ask hostage for half an hour past its 60s budget.) shutdown(wait=False)
        # lets the worker thread die with the overrun call still on it.
        pool = ThreadPoolExecutor(max_workers=1)
        # The worker gets a COPY of this request's context, and what it writes
        # there is copied back on success. A thread otherwise starts with an
        # empty context in both directions, and both directions had already gone
        # wrong: the request budget was invisible to anything the call ran, and
        # the token counts Ollama reports — set by the adapter into a ContextVar
        # — were written into a context that died with the thread. The release
        # page said "tokens estimated by word split" while the provider had been
        # returning exact counts all along, which is the failure mode this whole
        # module is supposed to prevent: a fallback that reads like a result.
        ctx = copy_context()
        try:
            future = pool.submit(ctx.run, fn, *args, **kwargs)
            try:
                result = future.result(timeout=policy.timeout)
            except FutureTimeout:
                raise TimeoutError(
                    f"{getattr(fn, '__name__', 'call')} exceeded {policy.timeout:.3g}s"
                ) from None
            # Only on success, and only after the result is in hand: an
            # abandoned call is still running on that thread, still writing to
            # that context, and adopting its half-finished state would be worse
            # than losing it.
            for var, value in ctx.items():
                var.set(value)
            return result
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(policy.attempts):
            try:
                return timed(*args, **kwargs)
            except Exception:
                if attempt == policy.attempts - 1:
                    raise
                ceiling = min(policy.max_delay, policy.base_delay * (2**attempt))
                sleep(ceiling * rng())
        raise AssertionError("unreachable")  # pragma: no cover

    return wrapped


@dataclass
class TokenBucket:
    """rate tokens/second refill, burst capacity. take() never blocks — a caller
    that finds the bucket empty gets False and should answer 429."""

    rate: float
    burst: float
    clock: Callable[[], float] = time.monotonic
    _tokens: float = field(init=False)
    _last: float = field(init=False)
    _lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self) -> None:
        self._tokens = self.burst
        self._last = self.clock()

    def take(self, n: float = 1.0) -> bool:
        with self._lock:
            now = self.clock()
            self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


@dataclass
class ConcurrencyCap:
    """Admission control: at most `limit` requests in flight. enter() is
    non-blocking — beyond the cap the service should shed load (503), because a
    request parked in a queue still ties up its caller's budget."""

    limit: int
    _active: int = field(init=False, default=0)
    _lock: Lock = field(init=False, default_factory=Lock)

    @property
    def active(self) -> int:
        return self._active

    def enter(self) -> bool:
        with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
