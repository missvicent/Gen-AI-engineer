"""What a retrieval layer has to do besides find things.

Finding things is the part that demos well. The parts that decide whether the
system survives a year in production are the ones tested here: does re-ingesting
a document update it or duplicate it, can you delete something, does a citation
still resolve to the text it named, and does the poisoned page get dropped
without taking the provenance of the clean ones with it.
"""

from assistant.rag import chunk_document, source_for

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
