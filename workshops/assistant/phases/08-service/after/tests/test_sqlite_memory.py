"""SqliteMemory in the fast tier: SQLite is stdlib, so the property that matters —
memory survives a restart — is testable offline with no service at all."""

from assistant.sqlite_memory import SqliteMemory


def test_it_survives_a_restart(tmp_path):
    db = str(tmp_path / "mem.db")
    first = SqliteMemory(db)
    mid = first.write("semantic", "I am based in Lima", source="onboarding")

    # a brand-new instance on the same file == a fresh process reading the store
    reopened = SqliteMemory(db)
    rows = reopened.all("semantic")
    assert [row.text for row in rows] == ["I am based in Lima"]
    assert rows[0].id == mid


def test_a_forget_stays_forgotten_after_a_restart(tmp_path):
    db = str(tmp_path / "mem.db")
    store = SqliteMemory(db)
    mid = store.write("semantic", "old fact", source="chat")
    store.forget(mid)
    assert SqliteMemory(db).all("semantic") == []


def test_provenance_is_not_optional():
    store = SqliteMemory(":memory:")
    try:
        store.write("semantic", "a claim with no source", source="  ")
    except ValueError:
        return
    raise AssertionError("a memory without a source should be refused")


def test_expiry_hides_a_row_from_recall_but_not_from_the_audit():
    store = SqliteMemory(":memory:")
    store.write("working", "ephemeral note", source="run", ttl_days=1, now=0.0)
    later = 2 * 86_400  # two days on
    assert store.recall("working", "ephemeral", now=later) == []
    assert len(store.all("working")) == 1  # staleness is auditable, not invisible


def test_correct_replaces_rather_than_outranks(tmp_path):
    db = str(tmp_path / "mem.db")
    store = SqliteMemory(db)
    mid = store.write("semantic", "timezone is Bogota", source="chat")
    store.correct(mid, "timezone is Lima", source="chat")
    texts = [row.text for row in SqliteMemory(db).all("semantic")]
    assert texts == ["timezone is Lima"]  # the stale row is gone, not merely lower
