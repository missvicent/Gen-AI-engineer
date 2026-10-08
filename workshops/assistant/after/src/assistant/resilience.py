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

## What is worth retrying

`retry on Exception` is the default everyone writes and it is wrong in two
directions at once. It retries a `TypeError` three times — the bug is
deterministic, so all it buys is three times the latency before the same 500 —
and it retries a 400 from an API that will reject the same payload forever. Worse,
each pointless attempt holds a connection and a worker while the real incident is
somewhere else.

So a failure is classified before it is retried. `TRANSIENT` is the honest list:
the network, the clock, and the server saying "later". Everything else is a bug or
a refusal, and both should surface on the first attempt, where the stack trace
still points at the cause.

## What is worth retrying ONCE

Retrying is safe when the call is idempotent. `send_telegram` is not: a retry
after a timeout may deliver a second message, because a timeout tells you nothing
about whether the server acted. `ONCE` exists for that case, and side-effecting
adapters use it — the timeout still applies, the retries do not. Making that a
named policy rather than a comment is the difference between a rule and a hope.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from assistant import deadline

#: Failures where the same call, later, could plausibly succeed. `ConnectionError`
#: and the socket errors are `OSError` subclasses, so this is shorter than it
#: looks; `TimeoutError` is listed because our own timeout raises it.
TRANSIENT: tuple[type[BaseException], ...] = (OSError, TimeoutError)


class Permanent(Exception):
    """A refusal that will be a refusal next time too.

    An adapter that can read a status code should raise this for a 4xx rather
    than letting a generic error be retried three times against a server that
    has already given its final answer."""


def is_transient(exc: BaseException) -> bool:
    """The default classifier. `Permanent` wins over the type check, so an
    adapter can mark a `ConnectionError`-shaped failure as final without having
    to invent a new exception hierarchy."""
    if isinstance(exc, deadline.Expired | Permanent):
        return False
    return isinstance(exc, TRANSIENT)


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

#: For calls that are NOT idempotent. One attempt, still bounded by the timeout:
#: a send that may or may not have happened must not be turned into a send that
#: definitely happened twice.
ONCE = Policy(attempts=1)


#: How often a blocked wait looks up from the clock to ask whether the caller is
#: still there. Short enough that a 499 is prompt, long enough that a request
#: waiting a minute for a model wakes about 240 times and not 60,000.
POLL_SECONDS = 0.25


def waited(future: Future, limit: float, name: str) -> Any:
    """Wait for a worker, watching two clocks and one door.

    `future.result(timeout=limit)` was one line and answered one question — has the
    time run out. It could not answer the other one, because a thread parked in
    `result()` reads nothing: **has the caller left.** A buffered `/ask` whose client
    hangs up flips `Budget.cancelled` immediately (the watcher task is running, and a
    sync route does not block the loop), and this wait carried right on to the end of
    a 60-second budget generating an answer for a closed socket.

    So the wait happens in slices, and between slices cancellation gets a look. The
    exception is `deadline.Expired`, not `TimeoutError`, and the difference is load
    bearing in two places: `fallbacks` re-raises `Expired` instead of composing an
    offline answer (a hangup is not a model outage and must not be recorded as one),
    and the API maps it to 499.

    Clock exhaustion still raises `TimeoutError` with the message it always had.
    That is why the loop asks `Budget.cancelled()` specifically rather than
    `deadline.expired()` — the two conditions look identical from inside a `Budget`
    and mean opposite things: one is our failure to be fast enough, the other is not
    our failure at all.

    The abandoned worker is abandoned the same way the timeout path already
    abandons it: Python cannot kill a thread, so the caller stops waiting and
    `pool.shutdown(wait=False)` lets it die on its own. What makes that cheap rather
    than merely tidy is the seam at the other end — `adapters.joined` checks the
    deadline between stream parts and closes the provider's iterator, so the work
    itself stops instead of running on unread.
    """
    left = limit
    while left > 0:
        # The slice is subtracted, not `POLL_SECONDS`: a 0.05s timeout must still
        # give up after 0.05s, and a poll interval that ate a whole slice out of a
        # budget smaller than itself would turn every tight timeout into a lie in the
        # other direction.
        span = min(POLL_SECONDS, left)
        try:
            return future.result(timeout=span)
        except FutureTimeout:
            left -= span
            if deadline.cancelled():
                raise deadline.Expired("caller disconnected") from None
    raise TimeoutError(f"{name} exceeded {limit:.3g}s")


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

    Three things stop a retry loop, and only one of them is "it worked": the
    failure is not transient (`policy.retry_if`), the attempts are used up, or
    the REQUEST's budget is gone (deadline.py). That last one is the difference
    between a service whose timeouts add up and one whose timeouts hold: a call
    with 4 seconds of request budget left gets a 4-second timeout no matter what
    its own policy says."""

    def timed(*args: Any, **kwargs: Any) -> Any:
        # The tighter of the two clocks. A per-call timeout that outlives the
        # request it serves is not a timeout, it is a suggestion.
        limit = deadline.capped(policy.timeout)
        if limit is None:
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
            result = waited(future, limit, getattr(fn, "__name__", "call"))
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
            # Checked BEFORE the attempt: starting a call we already know we
            # cannot wait for burns a connection to produce a result nobody
            # reads. On the first pass this also refuses work for a caller who
            # gave up while an earlier stage was running.
            deadline.check()
            try:
                return timed(*args, **kwargs)
            except Exception as exc:
                last = attempt == policy.attempts - 1
                # Order matters: a permanent failure must surface on attempt one,
                # with the stack trace still pointing at the cause rather than at
                # the third identical retry of it.
                if last or not policy.retry_if(exc):
                    raise
                ceiling = min(policy.max_delay, policy.base_delay * (2**attempt))
                delay = ceiling * rng()
                left = deadline.remaining()
                if left is not None and delay >= left:
                    # Sleeping past the deadline to attempt a call that cannot
                    # finish is the most expensive possible way to fail.
                    raise
                sleep(delay)
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
