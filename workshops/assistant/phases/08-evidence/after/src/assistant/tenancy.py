"""Memory, partitioned by verified identity.

Both memory stores already take a `user` and scope every read and write to it —
`SqliteMemory` filters on a `user` column, `AssistantMemory` keeps its own dict.
What was missing was anyone passing one. The service built a single store with
the default `user="me"` and distinguished callers only by writing
`source="user:alice"`, a label that `recall` does not look at. Alice's
preferences were one query away from Bob's answer.

The fix composes rather than modifies: hand this a factory, and it hands each
subject its own store. Isolation then comes from the store's existing scoping
instead of from a filter someone has to remember to apply at every call site —
and the two backends need no changes to gain it.

`all()` deliberately crosses tenants, because it is the operator's view: "what
has this service remembered". Recall never does.
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

        Backends differ in where the partition lives and it does not matter here:
        a SqliteMemory writes the subject into a column of the shared file, an
        AssistantMemory gets a private dict. Both answer only for their own.
        """
        with self._lock:
            if subject not in self._stores:
                self._stores[subject] = self._make(subject)
            return self._stores[subject]

    def remember(self, turn: str, *, source: str, subject: str) -> str | None:
        return self.store_for(subject).remember(turn, source=source)

    def recall(self, kind: Kind, query: str, subject: str, k: int = 3) -> list[Memory]:
        return self.store_for(subject).recall(kind, query, k=k)

    def all(self, kind: Kind | None = None) -> list[Memory]:
        """Every tenant's rows — the audit view, not a recall path."""
        rows: list[Memory] = []
        with self._lock:
            stores = list(self._stores.values())
        for store in stores:
            rows.extend(store.all(kind))
        return rows
