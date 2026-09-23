"""TODO: idempotency keys for EVERY side effect — a SQLite dedupe table.

A client that retries a timed-out request is a well-behaved client: a timeout
tells it nothing about whether the server acted, so asking again is the only
correct move. It is also the move that grants a second approval, ingests a batch
twice, or deletes a source that had been re-added in between. The server has to
be the one that makes the retry safe, because the client genuinely cannot.

The mechanism is the standard one: the client sends an `Idempotency-Key`, the
server records it, and a key it has seen is acknowledged without re-applying the
effect. Two details separate a real implementation from a decorative one, and
this scaffold has neither yet.

**The stored answer must be the ORIGINAL answer.** Returning `{"replayed": true}`
and nothing else technically avoids the double effect and still breaks the
client, which asked a question and got a receipt.

**A failure must not be recorded.** If the operation raises, the key has to be
released, or one transient failure becomes a permanent one: every retry
cheerfully acknowledged, the effect never applied, and nothing in the logs
saying so.

Keys are namespaced by subject and operation before they get here (`api.py`).
They are chosen by clients, and one tenant's `"retry-1"` must not swallow
another's.

Reference: ../../after/src/assistant/idempotency.py.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from typing import Any


class IdempotencyStore:
    def __init__(self, path: str = ":memory:") -> None:
        # one long-lived connection: ':memory:' would otherwise vanish per call
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        # TODO 1: this table records that a key was seen and nothing else, which
        # is why a replay can only answer with a receipt. Add a `result` column
        # so the original answer can be stored and replayed verbatim.
        #
        # Then read the statement below again. `CREATE TABLE IF NOT EXISTS` does
        # nothing when the table exists, INCLUDING when it exists with the old
        # two-column shape — which is exactly what is sitting in the Docker
        # volume of anyone who ran this stack before your change. Adding the
        # column to the CREATE is not the migration; it only helps a database
        # that does not exist yet. Compare the schema to the live table
        # (`PRAGMA table_info`) and `ALTER TABLE ... ADD COLUMN` what is missing.
        #
        # Skip it and the failure is instructive: startup succeeds, /health is
        # green, every read works, and the first client to time out and retry
        # gets `sqlite3.OperationalError: no such column: result` — a 500 on the
        # one code path that only runs when something has already gone wrong.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS idempotency_keys ("
            "  key TEXT PRIMARY KEY,"
            "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        self._conn.commit()

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
        """TODO 2: the stored answer for a key, or None if there isn't one yet."""
        raise NotImplementedError

    def release(self, key: str) -> None:
        """TODO 3: forget a claimed key so the client's retry actually runs.

        Used when the operation failed — an unrecorded effect must stay
        retryable.
        """
        raise NotImplementedError

    def store(self, key: str, result: Any) -> None:
        """TODO 4: save the answer against the key (JSON, `default=str`)."""
        raise NotImplementedError

    def run(self, key: str | None, operation: Callable[[], Any]) -> tuple[Any, bool]:
        """TODO 5: run `operation` at most once per key. Return `(result, replayed)`.

        No key means no protection, which is the caller's choice and their risk —
        the same shape as every idempotency header in the wild. Do NOT silently
        upgrade it to a server-generated key: a key the client does not know
        cannot be used to retry.

        Otherwise: claim the key with `seen`. If it was already claimed, replay
        `recall` (falling back to `{"replayed": True}` when the winner has
        claimed the key but not yet stored its answer — rare, and honest). If the
        claim is yours, run the operation, `store` the result, and `release` the
        key on any exception before re-raising.
        """
        raise NotImplementedError
