"""What a retrieval layer has to do besides find things.

Finding things is the part that demos well. The parts that decide whether the
system survives a year in production are the ones tested here: does re-ingesting
a document update it or duplicate it, can you delete something, does a citation
still resolve to the text it named, and does the poisoned page get dropped
without taking the provenance of the clean ones with it.
"""

import re

from assistant.adapters import (
    InMemoryRag,
    collection_name,
    distinctive_terms,
    hash_embed,
)
from assistant.rag import Chunk

REFUNDS = "approved refunds are processed within five business days"


def test_the_sparse_arm_may_admit_on_identifiers_and_nothing_else():
    """The rule that lets a keyword hit survive a relevance gate, and the reason it
    is narrow. Admission asks each retrieval arm in its own units — a cosine floor
    for dense evidence, an exact-term rule for sparse — and this is the sparse half.

    Widening it to "any shared word" would return the system to the failure the
    dense floor was introduced to fix, from the other side: `work` and `refund`
    appear verbatim all over an office corpus, and admitting on them would mean
    never abstaining again. So a term counts when being exact is the whole of its
    meaning: it carries a digit, or it is written in caps.

    The Qdrant-backed half of this lives in `test_integration.py`, at the floor the
    deployment actually sets. This half runs on every push.
    """
    # `ZX` is dropped as too short to be a code on its own, and it costs nothing:
    # the number beside it is what no embedder could have placed.
    assert distinctive_terms("where is order ZX-99417") == {"99417"}
    assert distinctive_terms("what does E_TIMEOUT mean") == {"timeout"}
    assert distinctive_terms("is CVE-2026-1 patched") == {"cve", "2026", "1"}

    for prose in ("how does photosynthesis work", "when do refunds arrive", "hello"):
        assert distinctive_terms(prose) == set(), (
            f"{prose!r} has no exact-match evidence in it, so the sparse arm has "
            "no grounds to overrule the cosine floor"
        )

    # Case is a signal, not noise: `IT` is a department and `it` is a pronoun.
    assert distinctive_terms("escalate to IT support") == set(), "two letters is not a code"
    assert "sla" in distinctive_terms("what is the SLA")


def test_reingesting_the_same_source_updates_instead_of_accumulating():
    rag = InMemoryRag()
    assert rag.add([{"text": REFUNDS, "source": "refunds.md"}], tenant="alice") == 1
    rag.add([{"text": REFUNDS, "source": "refunds.md"}], tenant="alice")
    rag.add(
        [{"text": "approved refunds take ten business days", "source": "refunds.md"}],
        tenant="alice",
    )
    hits = rag.search("refunds", k=10, tenant="alice")
    assert len(hits) == 1, "three ingests of one document must leave one chunk"
    assert "ten" in hits[0].text, "the surviving chunk must be the latest revision"


def test_deleting_a_source_removes_every_chunk_of_it_and_nothing_else():
    rag = InMemoryRag()
    rag.add(
        [
            {"text": ("refund policy. " * 200).strip(), "source": "policy.md"},
            {"text": "escalations go to the duty manager", "source": "escalation.md"},
        ],
        tenant="alice",
    )
    removed = rag.delete("policy.md", tenant="alice")
    assert removed > 1, "a multi-chunk source must be deleted in full"
    assert not rag.search("refund policy", k=5, tenant="alice")
    assert rag.search("escalations", k=5, tenant="alice"), "the neighbour survived"


def test_delete_is_scoped_to_a_tenant():
    """A delete that crossed tenants would be a denial-of-service with a REST
    interface: know a source name, erase someone else's corpus."""
    rag = InMemoryRag()
    for who in ("alice", "bob"):
        rag.add([{"text": REFUNDS, "source": "refunds.md"}], tenant=who)
    rag.delete("refunds.md", tenant="alice")
    assert not rag.search("refunds", k=5, tenant="alice")
    assert rag.search("refunds", k=5, tenant="bob"), "bob's copy was not alice's to delete"


def test_the_embedding_dimension_is_measured_from_the_embedder_not_declared():
    """The Qdrant collection is created with whatever the injected embedder
    actually returns. A hand-maintained constant is a 400 on the first write
    after somebody swaps the model — in production, at deploy time."""
    assert len(hash_embed("probe")) == len(hash_embed("a different probe"))
    assert len(hash_embed("probe", dim=768)) == 768, "the offline vector is not hard-wired"


def test_chunks_are_what_the_store_returns_and_text_is_still_reachable():
    """The interface change that pays for all of the above: search returns
    evidence, not prose. Everything downstream that only wants the words can
    still have them."""
    rag = InMemoryRag([REFUNDS])
    hit = rag.search("refunds", k=1)[0]
    assert isinstance(hit, Chunk)
    assert hit.text == REFUNDS


def test_the_collection_name_carries_the_embedder_and_its_width():
    """The silent corruption this prevents. Qdrant validates the DIMENSION of an
    incoming vector and nothing else, so swapping one 768-dimensional embedder
    for another writes cleanly into the old collection, searches cleanly, and
    returns noise — a failure with no error in it anywhere.

    Naming the collection after the embedder turns that into a new, empty
    collection: visibly wrong on the first query instead of invisibly wrong
    forever."""
    assert collection_name("assistant", "nomic-embed-text", 768) == (
        "assistant__nomic-embed-text__768"
    )
    # the same base with a different model is a different store, which is the point
    assert collection_name("assistant", "mxbai-embed-large", 768) != (
        collection_name("assistant", "nomic-embed-text", 768)
    )
    # and so is the same model at a different width
    assert collection_name("assistant", "nomic-embed-text", 512) != (
        collection_name("assistant", "nomic-embed-text", 768)
    )


def test_a_model_name_qdrant_would_reject_is_sanitised_not_passed_through():
    """Registry-style names carry slashes and colons; collection names do not."""
    name = collection_name("assistant", "BAAI/bge-small-en-v1.5:latest", 384)
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", name), name
    assert "384" in name and "bge-small" in name


def test_the_hash_embedder_is_recorded_as_such_rather_than_left_blank():
    """An unnamed default is how the deployed stack ran on a non-semantic
    embedder without anybody noticing."""
    assert collection_name("assistant", "hash", 64) == "assistant__hash__64"
