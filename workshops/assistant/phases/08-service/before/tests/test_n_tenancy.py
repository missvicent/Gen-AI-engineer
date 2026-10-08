"""Memory belongs to a person, and recall has to enforce that.

The shipped defect was quiet: one store for everyone, `user="me"`, with the real
caller written into a `source` label that `recall` never reads. Nothing looked
broken — alice got her own answers back — right up until bob asked a question
whose words happened to match one of her memories.

These tests are written against the leak rather than the fix, so they would fail
again if someone reintroduced a single shared store.
"""

import sqlite3

import pytest

from assistant.memory import AssistantMemory
from assistant.sqlite_memory import SqliteMemory
from assistant.tenancy import TenantMemory

ALICE_FACT = "I prefer to be called Lu and I work in the Lima timezone"

BOB_QUESTION = "what timezone should we use for the Lima call"


def in_memory() -> TenantMemory:
    return TenantMemory(lambda subject: AssistantMemory(user=subject))


def on_disk(tmp_path) -> TenantMemory:
    db = str(tmp_path / "memory.db")
    return TenantMemory(lambda subject: SqliteMemory(db, user=subject))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_one_persons_memory_never_surfaces_in_anothers_recall(backend, tmp_path):
    store = in_memory() if backend == "memory" else on_disk(tmp_path)
    store.remember(ALICE_FACT, source="user:alice", subject="alice")

    assert [row.text for row in store.recall("semantic", BOB_QUESTION, subject="alice")]
    assert store.recall("semantic", BOB_QUESTION, subject="bob") == []


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_each_subject_recalls_their_own_answer_to_the_same_question(backend, tmp_path):
    store = in_memory() if backend == "memory" else on_disk(tmp_path)
    store.remember("I prefer the window seat", source="user:alice", subject="alice")
    store.remember("I prefer the aisle seat", source="user:bob", subject="bob")

    def recalled(subject: str) -> str:
        return store.recall("semantic", "which seat do I prefer", subject=subject)[0].text

    assert "window" in recalled("alice")
    assert "aisle" in recalled("bob")


def test_the_partition_is_in_the_database_not_only_in_the_process(tmp_path):
    """A restart must not merge tenants. The rows carry the subject, so a fresh
    process reading the same file still answers only for the caller."""
    db = str(tmp_path / "memory.db")
    first = TenantMemory(lambda subject: SqliteMemory(db, user=subject))
    first.remember(ALICE_FACT, source="user:alice", subject="alice")

    rows = sqlite3.connect(db).execute("SELECT DISTINCT user FROM memories").fetchall()
    assert rows == [("alice",)]

    reborn = TenantMemory(lambda subject: SqliteMemory(db, user=subject))
    assert reborn.recall("semantic", BOB_QUESTION, subject="bob") == []
    assert reborn.recall("semantic", BOB_QUESTION, subject="alice")


def test_all_crosses_tenants_because_it_is_the_operator_view():
    store = in_memory()
    store.remember("I prefer the window seat", source="user:alice", subject="alice")
    store.remember("I prefer the aisle seat", source="user:bob", subject="bob")
    assert len(store.all("semantic")) == 2


def test_a_turn_not_worth_remembering_is_not_remembered():
    store = in_memory()
    assert store.remember("what time is it", source="user:alice", subject="alice") is None
    assert store.all() == []
