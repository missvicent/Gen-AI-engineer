"""SQLite-backed memory — the same interface as AssistantMemory, but it survives a
restart. The in-process dict store proves the *logic* (namespacing, TTL, forgetting);
this proves the property a deployed assistant actually needs: it remembers you across
process boundaries, and a `forget` stays forgotten after a reboot.

The interface is identical on purpose (write/remember/correct/recall/all/forget/
forget_all), so `service.py` swaps one for the other with no other change — the whole
argument for programming against a shape instead of a class.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Literal

from assistant.memory import (
    DAY_SECONDS,
    DEFAULT_TTL_DAYS,
    KINDS,
    Kind,
    Memory,
    classify,
    overlap,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT UNIQUE,
    user        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL,
    written_at  REAL NOT NULL,
    expires_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_user_kind ON memories(user, kind);
"""


class SqliteMemory:
    """Durable four-namespace memory. `db_path=":memory:"` gives an ephemeral store
    that still exercises the SQL path; a real file path makes it persistent."""

    def __init__(self, db_path: str, user: str = "me") -> None:
        self.user = user
        # check_same_thread=False: FastAPI serves requests on a threadpool, and the
        # store is guarded by SQLite's own locking, not shared Python state.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

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
        days = DEFAULT_TTL_DAYS[kind] if ttl_days == "default" else ttl_days
        expires_at = None if days is None else moment + days * DAY_SECONDS
        # AUTOINCREMENT never reuses a seq, so ids stay unique even after a forget.
        cur = self.conn.execute(
            "INSERT INTO memories(user, kind, text, source, written_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (self.user, kind, text.strip(), source, moment, expires_at),
        )
        memory_id = f"m{cur.lastrowid}"
        self.conn.execute("UPDATE memories SET id = ? WHERE seq = ?", (memory_id, cur.lastrowid))
        self.conn.commit()
        return memory_id

    def remember(self, turn: str, *, source: str, now: float | None = None) -> str | None:
        kind = classify(turn)
        if kind is None:
            return None
        return self.write(kind, turn, source=source, now=now)

    def correct(
        self, memory_id: str, text: str, *, source: str, now: float | None = None
    ) -> str:
        row = self.conn.execute(
            "SELECT kind FROM memories WHERE id = ? AND user = ?", (memory_id, self.user)
        ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        self.forget(memory_id)
        return self.write(row["kind"], text, source=source, now=now)

    # ------------------------------------------------------------ recall path
    def recall(
        self, kind: Kind, query: str, k: int = 3, now: float | None = None
    ) -> list[Memory]:
        moment = time.time() if now is None else now
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user = ? AND kind = ?"
            " AND (expires_at IS NULL OR expires_at >= ?)",
            (self.user, kind, moment),
        ).fetchall()
        scored = [self._to_memory(row, overlap(query, row["text"])) for row in rows]
        ranked = sorted(scored, key=lambda row: (-row.score, row.id))
        return [row for row in ranked if row.score > 0][:k]

    def all(self, kind: Kind | None = None) -> list[Memory]:
        if kind is None:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE user = ?", (self.user,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE user = ? AND kind = ?", (self.user, kind)
            ).fetchall()
        return sorted((self._to_memory(row) for row in rows), key=lambda row: row.id)

    def forget(self, memory_id: str) -> None:
        self.conn.execute(
            "DELETE FROM memories WHERE id = ? AND user = ?", (memory_id, self.user)
        )
        self.conn.commit()

    def forget_all(self, kind: Kind | None = None) -> int:
        doomed = [row.id for row in self.all(kind)]
        for memory_id in doomed:
            self.forget(memory_id)
        return len(doomed)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _to_memory(row: sqlite3.Row, score: float = 0.0) -> Memory:
        return Memory(
            id=row["id"],
            kind=row["kind"],
            text=row["text"],
            source=row["source"],
            written_at=row["written_at"],
            expires_at=row["expires_at"],
            score=score,
        )
