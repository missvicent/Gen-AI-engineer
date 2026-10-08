"""Idempotency keys for every side effect — a SQLite dedupe table.

A client that retries a timed-out request is a well-behaved client: a timeout
tells it nothing about whether the server acted, so asking again is the only
correct move. It is also the move that grants a second approval, ingests a batch
twice, or deletes a source that had been re-added in between. The server has to
be the one that makes the retry safe, because the client genuinely cannot.

The mechanism is the standard one: the client sends an `Idempotency-Key`, the
server records it, and a key it has seen is acknowledged without re-applying the
effect. Two details separate a real implementation from a decorative one.

**The stored answer is the ORIGINAL answer.** Returning `{"replayed": true}` and
nothing else technically avoids the double effect and still breaks the client,
which asked a question and got a receipt. `run()` stores the first result and
replays it verbatim, so a retry is indistinguishable from the call it repeats —
which is the entire promise of the word "idempotent".

**A failure is not recorded.** If the operation raises, the key is released, so
the retry the client is about to send actually runs. Recording the key up front
would turn one transient failure into a permanent one: every retry cheerfully
acknowledged, the effect never applied, and nothing in the logs saying so.

Keys are namespaced by subject and operation before they get here (`api.py`).
They are chosen by clients, and one tenant's `"retry-1"` must not swallow
another's.

The table lives in the same SQLite file as memory when ASSISTANT_DB is set, so
replay protection survives a restart; without it, an in-process `:memory:` table
covers the single-process demo.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from typing import Any


class IdempotencyStore:
    def __init__(self, path: str = ":memory:") -> None:
        # one long-lived connection: ':memory:' would otherwise vanish per call
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS idempotency_keys ("
            "  key TEXT PRIMARY KEY,"
            "  result TEXT,"
            "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add `result` to a table written by a build that only stored the key.

        `CREATE TABLE IF NOT EXISTS` does nothing at all when the table exists,
        including when it exists with the wrong shape. On a fresh database the
        statement above is the whole schema and this looks like dead code; on a
        volume that has been through an upgrade, it is the difference between a
        working deploy and one where every mutating route raises `no such column:
        result` — at write time, on the retry path, long after the health check
        went green. That is the whole failure mode: the schema drift is invisible
        to startup and to every read, and surfaces first on the code path a
        client only reaches when something else already went wrong.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(idempotency_keys)")}
        if "result" not in existing:
            self._conn.execute("ALTER TABLE idempotency_keys ADD COLUMN result TEXT")

    def seen(self, key: str) -> bool:
        """Record the key; report whether it was already there. INSERT OR IGNORE
        makes the check-and-set a single atomic statement."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys (key) VALUES (?)", (key,)
            )
            self._conn.commit()
        return cur.rowcount == 0

    def recall(self, key: str) -> Any | None:
        """The stored answer for a key, or None if there isn't one yet."""
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM idempotency_keys WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    def release(self, key: str) -> None:
        """Forget a claimed key so the client's retry actually runs. Used when
        the operation failed — an unrecorded effect must stay retryable."""
        with self._lock:
            self._conn.execute("DELETE FROM idempotency_keys WHERE key = ?", (key,))
            self._conn.commit()

    def store(self, key: str, result: Any) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE idempotency_keys SET result = ? WHERE key = ?",
                (json.dumps(result, default=str), key),
            )
            self._conn.commit()

    def run(self, key: str | None, operation: Callable[[], Any]) -> tuple[Any, bool]:
        """Run `operation` at most once per key. Returns `(result, replayed)`.

        No key means no protection, which is the caller's choice and their risk —
        the same shape as every idempotency header in the wild. It is not silently
        upgraded to a server-generated key, because a key the client does not know
        cannot be used to retry.

        The claim (`seen`) is a single atomic INSERT, so two concurrent retries of
        the same key race for one row: one runs the operation, the other replays.
        The loser can observe `None` briefly — the winner has claimed the key but
        not yet stored its answer — and gets the honest `{"replayed": true}`
        rather than a fabricated result. Rare, and the alternative is holding a
        write lock for the duration of a network call.
        """
        if key is None:
            return operation(), False
        if self.seen(key):
            stored = self.recall(key)
            return (stored if stored is not None else {"replayed": True}), True
        try:
            result = operation()
        except Exception:
            self.release(key)
            raise
        self.store(key, result)
        return result, False
