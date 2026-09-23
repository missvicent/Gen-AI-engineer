"""Memory, partitioned by verified identity.

Both memory stores already take a `user` and scope every read and write to it —
`SqliteMemory` filters on a `user` column, `AssistantMemory` keeps its own dict.
What was missing was anyone passing one. The service built a single store with
the default `user="me"` and distinguished callers only by writing
`source="user:alice"`, a label that `recall` does not look at. Alice's
preferences were one query away from Bob's answer.

That bug is worth sitting with before fixing it, because nothing about it looks
wrong: every caller gets sensible answers, the tests pass, and the label in the
database says exactly who said what. It only fails when two people's words
happen to overlap — which is to say, in production and not in the test.

The fix composes rather than modifies: hand this a factory, and it hands each
subject its own store. Isolation then comes from the store's existing scoping
instead of from a filter someone has to remember to apply at every call site —
and the two backends need no changes to gain it.

Reference: ../../after/src/assistant/tenancy.py.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from assistant.memory import Kind, Memory


class TenantMemory:
    def __init__(self, make: Callable[[str], Any]) -> None:
        self._make = make
        self._stores: dict[str, Any] = {}
        self._lock = threading.Lock()

    def store_for(self, subject: str) -> Any:
        """This subject's memory, created on first sight.

        TODO 1: build the store with `self._make(subject)` the first time a
        subject is seen and reuse it after — under `self._lock`, because FastAPI
        serves requests on a threadpool and two simultaneous first requests from
        the same subject must not end up with two stores.

        Backends differ in where the partition lives and it must not matter
        here: a SqliteMemory writes the subject into a column of the shared
        file, an AssistantMemory gets a private dict. Both answer only for
        their own.
        """
        raise NotImplementedError

    def remember(self, turn: str, *, source: str, subject: str) -> str | None:
        return self.store_for(subject).remember(turn, source=source)

    def recall(self, kind: Kind, query: str, subject: str, k: int = 3) -> list[Memory]:
        return self.store_for(subject).recall(kind, query, k=k)

    def all(self, kind: Kind | None = None) -> list[Memory]:
        """Every tenant's rows — the audit view, not a recall path.

        TODO 2: aggregate across the stores built so far. Note the asymmetry and
        keep it deliberate: this one crosses tenants because an operator asking
        "what has this service remembered" needs the whole answer, while
        `recall` must never cross one.
        """
        raise NotImplementedError
