from assistant.rag import RagStore

DOCS = [
    "hybrid search fuses keyword and vector retrieval",
    "invoice INV-88231 was paid on July third",
    "the weather is sunny today",
]


def test_semantic_search():
    s = RagStore(DOCS)
    hits = s.search("keyword and vector", k=1)
    assert "hybrid" in hits[0]


def test_exact_id_via_keyword_arm():
    s = RagStore(DOCS)
    hits = s.search("INV-88231", k=2)
    assert any("INV-88231" in h for h in hits)
