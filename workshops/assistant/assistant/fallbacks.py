"""Degraded operation — real adapters fail soft, never hard.

A capstone that 500s because Qdrant restarted has not learned the operate
lesson. Every network adapter is wrapped twice: `resilient` retries transient
failures, and if the call still fails the OFFLINE default answers instead —
with the degradation reported on /health so an operator can see it. Retry
budgets differ by adapter: a vector search should fail fast (the standby is
right there); a model completion is worth waiting for.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextvars import copy_context
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from assistant import auth, deadline, usage
from assistant.adapters import close_stream
from assistant.agent import Step
from assistant.composers import (
    Composer,
    StreamComposer,
    StreamTruncated,
    offline_compose,
    word_stream,
)
from assistant.resilience import Policy, resilient

RAG_POLICY = Policy(attempts=3, base_delay=0.2, timeout=10.0)
COMPOSE_POLICY = Policy(attempts=2, base_delay=0.2, timeout=60.0)


def not_a_fallback(exc: BaseException) -> None:
    """Re-raise the failures that are not degradation. Called first in every
    `except Exception` in this module.

    `deadline.Expired` means the request is over — the clock ran out, or the client
    hung up. Neither says anything about the dependency, and both used to arrive here
    and be treated as a dependency failure. That produced the worst possible output
    of a resilience layer: a caller who had already left got a full offline answer
    composed for them, and /health reported that the *brain tier was degraded*
    because of it. An operator would then see a model outage that never happened, and
    the release lane's `require_no_fallback_during` would fail the run over a client
    that pressed stop.

    It became reachable when `resilience.waited` gained the cancellation poll: before
    that, a hangup could not interrupt a call in progress, so this was a latent bug
    with no path to it. Adding the seam upstream is what made this line necessary,
    which is the usual shape — a fix one layer down turns a dead branch live.
    """
    if isinstance(exc, deadline.Expired):
        raise exc


class FallbackRag:
    """Primary RAG with retries, offline BM25 as the warm standby. Ingest is
    MIRRORED into the standby up front — a fallback with an empty corpus at
    query time would be an abstention generator, not a fallback."""

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        report: Callable[[str, str], None],
        policy: Policy = RAG_POLICY,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self._report = report
        self._add = resilient(primary.add, policy)
        self._search = resilient(primary.search, policy)

    def add(self, docs: list, tenant: str = auth.ANONYMOUS) -> int:
        mirrored = self.fallback.add(docs, tenant=tenant)
        try:
            return self._add(docs, tenant=tenant)
        except Exception as exc:
            not_a_fallback(exc)
            self._report("rag", f"ingest fell back to in-memory: {exc}")
            # the standby's count, not len(docs): both stores chunk the same way,
            # so this is the real number of retrievable units either way
            return mirrored

    def search(self, query: str, k: int = 3, tenant: str = auth.ANONYMOUS) -> list:
        try:
            return self._search(query, k, tenant=tenant)
        except Exception as exc:
            not_a_fallback(exc)
            self._report("rag", f"search fell back to in-memory: {exc}")
            return self.fallback.search(query, k, tenant=tenant)

    def delete(self, source: str, tenant: str = auth.ANONYMOUS) -> int:
        """Delete from BOTH stores, always. A deletion that only lands on the
        primary is a deletion that un-deletes itself the next time the primary is
        unreachable and the standby answers — which is the worst possible moment
        for a withdrawn document to come back."""
        removed = self.fallback.delete(source, tenant=tenant)
        try:
            return self.primary.delete(source, tenant=tenant)
        except Exception as exc:
            not_a_fallback(exc)
            self._report("rag", f"delete fell back to in-memory: {exc}")
            return removed

    def get(self, chunk_id: str, tenant: str = auth.ANONYMOUS):
        try:
            return self.primary.get(chunk_id, tenant=tenant)
        except Exception as exc:
            not_a_fallback(exc)
            self._report("rag", f"citation lookup fell back to in-memory: {exc}")
            return self.fallback.get(chunk_id, tenant=tenant)


def fallback_composer(
    primary: Composer,
    report: Callable[[str, str], None],
    policy: Policy = COMPOSE_POLICY,
) -> Composer:
    """Model-tier composition with retries; the offline stitcher answers when the
    model stays down. Degraded prose beats a 500 — the evidence is already here."""
    guarded = resilient(primary, policy)

    def compose(
        goal: str,
        contexts: list[str],
        state: list[tuple[Step, Any]],
        memories: list[str] | None = None,
    ) -> str:
        try:
            return guarded(goal, contexts, state, memories)
        except Exception as exc:
            not_a_fallback(exc)
            report("brain", f"composition fell back to offline: {exc}")
            return offline_compose(goal, contexts, state, memories)

    return compose


def fallback_stream(
    primary: StreamComposer,
    report: Callable[[str, str], None],
    timeout: float | None = COMPOSE_POLICY.timeout,
) -> StreamComposer:
    """Streaming with a fallback, bounded in TIME as well as guarded on errors.

    The batch composer gets its timeout from `resilient`; a generator cannot,
    because the slowness happens between yields — so the primary is drained on a
    worker thread and each chunk must arrive within `timeout`. A model that dies
    or stalls BEFORE the first chunk falls back to the offline answer; one that
    stalls MID-stream raises `StreamTruncated`, because those chunks are already
    on the wire and cannot be taken back. Either way the degradation is
    reported, never absorbed. (Caught live: a thinking model spent minutes
    reasoning before its first token, and the unbounded stream held /ask/stream
    hostage the whole time.)

    Raising rather than returning is the point of the second case. `return`
    ends the generator normally, and every layer above reads a normal end as
    "that was the whole answer" — so a model that died mid-sentence gets
    delivered as a complete, cited, confident fragment.

    The worker also has to know when to stop, and it has two ways of finding out —
    the request's budget, which it can only see because its context is copied in,
    and this generator ending, which the budget cannot see at all. Both matter, and
    neither is enough alone: the budget catches a caller who left, the ending catches
    a response that finished for any other reason. Closing the source is what makes
    either one actually stop a provider rather than merely stop reading it."""

    def stream(
        goal: str,
        contexts: list[str],
        state: list[tuple[Step, Any]],
        memories: list[str] | None = None,
    ) -> Iterator[str]:
        frames: Queue = Queue()
        degraded = word_stream(offline_compose)
        #: Set when this generator is done with the producer for ANY reason —
        #: disconnect, a truncation this raised, an output gate that ended the
        #: response, a consumer that simply stopped iterating. The budget covers the
        #: first of those and nothing else, and "nobody is reading the queue" is
        #: reason enough to stop filling it.
        done = Event()

        def drain() -> None:
            # Held in a name so the `finally` can close it. `for chunk in primary(...)`
            # discards the iterator, and an iterator you cannot reach is an HTTP
            # response you cannot tear down.
            source = primary(goal, contexts, state, memories or [])
            try:
                for chunk in source:
                    # Between frames, which is the only place this thread is ever
                    # idle. `adapters.joined` and the stream helpers check the same
                    # budget one layer down; this check is what covers a composer
                    # that checks nothing, and there is always one of those.
                    deadline.check()
                    if done.is_set():
                        return
                    frames.put(("chunk", chunk))
                # The provider's own token counts ride the terminal frame.
                #
                # The adapter reports them into a ContextVar, and this thread's
                # context dies with it — so the request thread metered a word-split
                # estimate while exact counts sat unread. Only on a clean end: a
                # stream that failed has no totals to hand over, and the estimate
                # over what actually escaped is the honest number for it.
                frames.put(("end", usage.take_reported()))
            except Exception as exc:  # noqa: BLE001 — the seam exists to catch anything
                frames.put(("error", exc))
            finally:
                close_stream(source)

        # `copy_context()`, and this is the fix rather than a detail. A new thread
        # starts with an EMPTY context, so `deadline.current()` in here returned a
        # fresh unbounded `Budget` whose `cancelled` is `lambda: False` — the request's
        # budget was not absent, it was invisible. Every cancellation seam below this
        # point was therefore asleep on the only path that ships: the round-8 audit
        # watched a provider run from 4 chunks to 41 after the socket closed, still
        # generating, while the request itself had already returned.
        #
        # The copy shares the `Budget` OBJECT, so the flag the disconnect watcher flips
        # is the flag this thread reads. Writes stay in the copy, which is exactly what
        # the `("end", ...)` handover above and `usage.adopt` are built around — a
        # streaming producer cannot copy its context back, because it is still running
        # when the first chunk is consumed. Same seam, same reason, as
        # `resilience.timed`: it has submitted `ctx.run` since the day the token counts
        # went missing down this identical hole.
        #
        # Abandoned on timeout, like `resilient`.
        Thread(target=copy_context().run, args=(drain,), daemon=True).start()
        yielded = False
        try:
            while True:
                try:
                    # Shrunk to what the request has left, the way `resilient` bounds
                    # the batch composer. A 60-second per-chunk budget inside a request
                    # with four seconds left is a four-second budget; waiting the full
                    # 60 means the caller is long gone before this notices, and the
                    # per-chunk timeout was never meant to outlive the request it
                    # serves.
                    kind, payload = frames.get(timeout=deadline.capped(timeout))
                except Empty:
                    report("brain", f"stream produced no chunk within {timeout}s")
                    if yielded:
                        raise StreamTruncated(f"no chunk within {timeout}s") from None
                    yield from degraded(goal, contexts, state, memories or [])
                    return
                if kind == "chunk":
                    yielded = True
                    yield payload
                elif kind == "error":
                    # Same rule as the batch path, and reachable for the same reason:
                    # the adapters now raise `Expired` from inside a stream. The
                    # streaming gate in `output_gate` already ends the response as
                    # `truncated` when it sees one, which is the correct handling — what
                    # must not happen is a brain-degradation report on the way past.
                    not_a_fallback(payload)
                    report("brain", f"stream fell back to offline: {payload}")
                    if yielded:
                        raise StreamTruncated(str(payload)) from payload
                    yield from degraded(goal, contexts, state, memories or [])
                    return
                else:
                    usage.adopt(payload)
                    return
        finally:
            # Whatever ended this generator — a return above, a `StreamTruncated`, or
            # the `GeneratorExit` a closed connection delivers — the producer is done
            # being useful. This is what covers the endings a budget cannot see, and it
            # does not wait for a poll interval to do it.
            done.set()

    return stream
