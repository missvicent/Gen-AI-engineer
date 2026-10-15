"""What a retrieval layer has to do besides find things.

Finding things is the part that demos well. The parts that decide whether the
system survives a year in production are the ones tested here: does re-ingesting
a document update it or duplicate it, can you delete something, does a citation
still resolve to the text it named, and does the poisoned page get dropped
without taking the provenance of the clean ones with it.
"""

import pytest

from assistant.rag import Chunk, RagStore, as_chunks, chunk_document, source_for

REFUNDS = "approved refunds are processed within five business days"


def test_a_chunk_id_is_derived_from_where_it_came_from_not_from_when():
    """The single decision that makes re-ingest an update. Two runs over the same
    slice of the same source must agree on identity, or the corpus grows a
    revision every time somebody re-runs the loader."""
    first = chunk_document(REFUNDS, "refunds.md", tenant="alice")[0]
    again = chunk_document(REFUNDS, "refunds.md", tenant="alice")[0]
    assert first.id == again.id

    # tenant is part of identity: alice's copy and bob's copy are different points
    assert chunk_document(REFUNDS, "refunds.md", tenant="bob")[0].id != first.id
    # so is position, or every chunk of a page would overwrite the one before it
    long_doc = chunk_document("x" * 2000, "long.md")
    assert long_doc[1].id != long_doc[0].id


def test_an_edit_keeps_the_id_and_changes_the_version():
    """Content is deliberately NOT in the id: an edited paragraph must replace
    the old one rather than sit beside it. The version stamp is what tells you
    the text moved, and it is what a stale citation is caught by."""
    before = chunk_document(REFUNDS, "refunds.md")[0]
    after = chunk_document("approved refunds are processed within ten business days", "refunds.md")[
        0
    ]
    assert before.id == after.id, "an edit must UPDATE the point, not add one"
    assert before.version != after.version, "an edit must be visible in the stamp"


def test_an_unnamed_document_still_deduplicates():
    """The lazy path — `add(["..."])`, no source — has to be safe too, because it
    is the path a retry takes. A content-derived name means an accidental
    double-POST lands on the same point."""
    assert source_for(REFUNDS) == source_for(REFUNDS)
    assert source_for(REFUNDS) != source_for("something else entirely")


def test_a_long_document_becomes_several_chunks_that_know_their_offsets():
    body = ("refund policy. " * 200).strip()
    chunks = chunk_document(body, "policy.md")
    assert len(chunks) > 1, "a long document must be split or it cannot be ranked"
    for chunk in chunks:
        # the offsets are a claim about the ORIGINAL text, and they are checkable
        assert body[chunk.start : chunk.end] == chunk.text
    assert chunks[1].start < chunks[0].end, "chunks must overlap or boundaries lose sentences"


def test_as_chunks_accepts_and_checks_strings():
    """
    A plain string is a whole document: it gets a derived
    source and version and it is rejected when it is blank
    or bigger thn MAX_DOC_CHARS.
    """
    chunks = as_chunks(["hello", "world"])
    assert [c.text for c in chunks] == ["hello", "world"]
    assert all(c.source.startswith("doc-") for c in chunks)
    assert all(c.version for c in chunks)

    with pytest.raises(ValueError, match="missing 'text'"):
        as_chunks([" "])
    with pytest.raises(ValueError, match="max"):
        as_chunks(["x" * 200_001])


def test_as_chunks_accepts_and_checks_dicts():
    """
    A dict with 'text' and 'source' keys is a named document.
    It is rejected when it is missing or blank 'text' or 'source'.
    """
    chunks = as_chunks([{"text": "REFUNDS", "source": "faq.md"}])
    assert [c.text for c in chunks] == ["REFUNDS"]
    assert [c.source for c in chunks] == ["faq.md"]

    with pytest.raises(ValueError, match="must be a string"):
        as_chunks([{"text": 123, "source": "faq.md"}])
    with pytest.raises(ValueError, match="missing or blank 'source'"):
        as_chunks([{"text": "REFUNDS", "source": " "}])
    with pytest.raises(ValueError, match="max"):
        as_chunks([{"text": "x" * 200_001, "source": "faq.md"}])
    with pytest.raises(ValueError, match="missing 'text'"):
        as_chunks([{"text": " ", "source": "faq.md"}])


def test_as_chunks_accepts_and_checks_chunks():
    """
    A Chunk is a slice someone else already cut. It brings its own source and version,
    and is rejected when it is missing or blank 'text' or 'source' or 'version'.
    """
    good = Chunk(text="REFUNDS", source="faq.md", version="1.0.0")
    chunks = as_chunks([good], tenant="alice")
    assert [c.text for c in chunks] == ["REFUNDS"]
    assert [c.tenant for c in chunks] == ["alice"]

    blank_text = Chunk(text="", source="faq.md", version="1.0.0")
    no_source = Chunk(text="REFUNDS", version="1.0.0")
    no_version = Chunk(text="REFUNDS", source="faq.md")
    too_long = Chunk(text="x" * 481, source="faq.md", version="1.0.0")

    with pytest.raises(ValueError, match="missing 'text'"):
        as_chunks([blank_text])
    with pytest.raises(ValueError, match="missing 'source'"):
        as_chunks([no_source])
    with pytest.raises(ValueError, match="missing 'version'"):
        as_chunks([no_version])
    with pytest.raises(ValueError, match="max"):
        as_chunks([too_long])


def test_as_chunks_rejects_a_bare_container():
    """
    `add` takes a list of documents. A bare str would be looped over letter by
    letter and a bare dict key by key, each becoming a junk document.
    """
    with pytest.raises(TypeError, match="list of documents"):
        as_chunks("hello")
    with pytest.raises(TypeError, match="list of documents"):
        as_chunks({"text": "hi", "source": "faq.md"})
    with pytest.raises(TypeError, match="unknown type"):
        as_chunks([123])


def test_reingesting_the_same_source_updates_instead_of_duplicating():
    """
    When a source is re-ingested, the existing chunks should be updated
    instead of duplicated.
    """
    doc = [{"text": "REFUNDS. This is a refund policy." * 100, "source": "faq.md"}]
    store = RagStore(doc)
    n = len(store.chunks)
    stats = store.add(doc)
    assert n > 1, "a long document must be split or it cannot be ranked"
    assert len(store.chunks) == n, "re-ingesting the same source should not duplicate chunks"
    assert stats["inserted"] == 0, "re-ingesting the same source should not insert new chunks"
    assert stats["unchanged"] == n, (
        "re-ingesting the same source should not change the existing chunks"
    )


def test_reingesting_the_same_source_updates_the_chunks():
    """When a source is re-ingested, the existing chunks should be updated."""
    doc = [{"text": "REFUNDS. This is a refund policy." * 100, "source": "faq.md"}]
    doc2 = [
        {
            "text": "REFUNDS. this is a return policy." * 100,
            "source": "faq.md",
        }
    ]
    store = RagStore(doc)
    n = len(store.chunks)
    ids_before = {c for c in store.chunks}
    versions_before = {c.version for c in store.chunks.values()}
    stats = store.add(doc2)
    assert stats["inserted"] == 0
    assert stats["updated"] == n
    assert {c for c in store.chunks} == ids_before  # could it be set(store.chunks) too
    assert {c.version for c in store.chunks.values()} != versions_before


def test_a_retrieved_chunk_cites_its_source_version_and_offsets():
    """A retrieved chunk should cite its source, version, and offsets."""
    doc = [{"text": "REFUNDS. This is a refund policy." * 100, "source": "faq.md"}]
    store = RagStore(doc)
    chunk = store.retrieve("refund", k=1)[0]
    c = chunk.cite("1")
    start, end = c["offsets"]
    assert c["source"] == "faq.md"
    assert c["version"]
    assert doc[0]["text"][start:end] == chunk.text


# Ten near-identical policy notes outrank the one chunk that names the id: under
# RRF it lands 6th, below any k=3 cut. A reranker has to rescue it.
NOTES = [f"refund policy note {i}" for i in range(10)]
CREDIT = "refund issued by credit note INV-88231 after review of the full payment history"
QUERY = "refund note inv-88231"


def all_words(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Fake reranker: chunks containing every query word go first. Checks the
    wiring without downloading a model."""
    words = set(query.lower().split())
    return sorted(chunks, key=lambda c: not words <= set(c.text.lower().split()))


def test_an_identity_reranker_changes_nothing():
    """The wiring alone must not change the answer: same ids, same length."""
    plain = RagStore(NOTES + [CREDIT]).retrieve(QUERY, k=3)
    wired = RagStore(NOTES + [CREDIT], reranker=lambda q, cs: cs).retrieve(QUERY, k=3)
    assert [c.id for c in wired] == [c.id for c in plain]
    assert {c.scored_by for c in wired} == {"rrf+rerank"}


def test_the_reranker_can_rescue_a_chunk_from_below_the_cut():
    """The reranker sees a pool wider than k and the result is cut back to k.
    Without the wide pool it could only shuffle the top 3; without the cut the
    caller would get the whole pool."""
    assert CREDIT not in RagStore(NOTES + [CREDIT]).search(QUERY, k=3)

    store = RagStore(NOTES + [CREDIT], reranker=all_words)
    hits = store.retrieve(QUERY, k=3)
    assert len(hits) == 3
    assert hits[0].text == CREDIT


def test_search_follows_the_same_path_as_retrieve():
    store = RagStore(NOTES + [CREDIT], reranker=all_words)
    assert store.search(QUERY, k=3) == [c.text for c in store.retrieve(QUERY, k=3)]


def test_a_reranker_does_not_break_abstention_or_the_k_guard():
    store = RagStore(NOTES + [CREDIT], reranker=lambda q, cs: cs)
    assert store.retrieve("zebra quantum", k=3) == []
    assert store.retrieve(QUERY, k=0) == []
    with pytest.raises(ValueError, match="k must be"):
        store.retrieve(QUERY, k=-1)
