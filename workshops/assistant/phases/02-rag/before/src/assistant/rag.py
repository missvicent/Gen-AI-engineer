"""Workshop 2 layer — a small hybrid-retrieval RAG core the assistant can query.

Self-contained (offline) BM25 + bag-of-words dense, fused with RRF — the same
shape as phase2/02-hybrid-rerank, now a reusable component. Swap the store for
Qdrant + real embeddings in production; the interface stays put.

What the store holds is a **chunk**, not a string, and that distinction is the
whole difference between a demo and something you can operate. A chunk knows
where it came from (`source`), which revision of that source it is (`version`),
which slice of it it is (`ordinal`, `start`, `end`) and what its identity is
(`id`). Each of those exists because of a question a string cannot answer:

- *"Where did this claim come from?"* — a citation that says "rag" tells a user
  nothing they can check. One that names a source and a character range does.
- *"The policy page changed; is the answer stale?"* — the version stamp says so.
- *"Delete everything from that document."* — you cannot delete by prose.
- *"We re-ingested the corpus and now every answer cites four copies."* — an id
  derived from `(tenant, source, ordinal)` makes a re-ingest an UPDATE. An
  auto-incrementing counter makes it a duplicate, every single time.

A chunk carries text and not a vector: how it is embedded is the store's
business (`adapters.py`), and the offline store does not embed at all.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from assistant.provenance import digest

#: Chunk ids are UUIDv5 — deterministic, and one of the two id types Qdrant
#: accepts, so the same value works in memory and in the real store.
CHUNK_NAMESPACE = uuid.UUID("6f1a4a2e-6f2a-5c7b-9b23-2f4c1a9d8e10")

#: Characters, not tokens. Tokens are the right unit and require a tokenizer;
#: this is the workshop tier, and the offsets have to mean something a human can
#: check with a text editor.
CHUNK_CHARS = 480
CHUNK_OVERLAP = 60

#: With a reranker, fetch this many times k. The reranker can only reorder what
#: it is given, so the pool has to reach below the cut for it to rescue anything.
RERANK_POOL = 4

#: Rejected before chunking, at the trust boundary. One unbounded upload becomes
#: thousands of chunks and a full reindex that stalls every other caller.
MAX_DOC_CHARS = 200_000

DEFAULT_SOURCE_PREFIX = "doc"


@dataclass(frozen=True)
class Chunk:
    """One retrievable slice of one source document."""

    text: str
    source: str = "inline"
    version: str = ""
    ordinal: int = 0
    start: int = 0
    end: int = 0
    tenant: str = "local"
    #: What the retriever ranked and KEPT this chunk by, and in which units. Two
    #: fields rather than one because a bare float here would be uninterpretable:
    #: 0.66 is a strong cosine and a meaningless reciprocal rank. `scored_by` is
    #: "cosine" when a dense similarity decided it, "rrf" when a fusion did, and
    #: "sparse" when the keyword arm did — an IDF-weighted score, which is not on
    #: the same scale as a cosine and must never be compared to one.
    #:
    #: Carried through composition and into the citation because the relevance
    #: DECISION is part of the provenance. The alternative — a chunk that arrives
    #: as bare text — is what let a lexical filter downstream overrule a semantic
    #: retriever, with no number anywhere in the response to contradict it.
    score: float = 0.0
    scored_by: str = ""

    @property
    def id(self) -> str:
        """stable across re-ingests of the same slice of the same source

        Note what is NOT in here: the text. An id that changed with the content
        would turn every edit into a new point and leave the old one behind,
        which is how a corpus ends up quietly holding three revisions of the
        same paragraph and citing whichever one ranked highest."""
        return str(uuid.uuid5(CHUNK_NAMESPACE, f"{self.tenant}|{self.source}|{self.ordinal}"))

    @property
    def doc_key(self) -> tuple[str, str]:
        """The key used to identify the document in the store."""
        return (self.source, self.tenant)

    def cite(self, label: str) -> dict:
        """The chunk as a citation a caller can act on: where it came from,
        which revision, and the exact span — plus the id, so the same evidence
        can be fetched again later.

        The retrieval score rides along when there is one, named by its units. A
        reader who wants to know why a citation is in the list gets an answer, and
        so does an operator tuning `ASSISTANT_MIN_SCORE` against real traffic."""
        cited = {
            "id": label,
            "chunk_id": self.id,
            "source": self.source,
            "version": self.version,
            "offsets": [self.start, self.end],
            "snippet": self.text[:240],
        }
        # Only when the store actually scored it: publishing `"score": 0.0` for a
        # chunk nobody ranked invites exactly the misreading the units field is
        # here to prevent.
        return (
            cited | {"score": round(self.score, 4), "scored_by": self.scored_by}
            if self.scored_by
            else cited
        )


def source_for(text: str) -> str:
    """A name for a document that arrived without one, derived from its content.

    Two anonymous ingests of the same text therefore land on the same source and
    the same ids — an accidental double-POST updates one point instead of
    creating a second copy that will be retrieved alongside the first forever."""
    return f"{DEFAULT_SOURCE_PREFIX}-{digest(text)}"


def chunk_document(
    text: str,
    source: str | None = None,
    *,
    tenant: str = "local",
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split one document into overlapping chunks that remember where they were.

    The overlap is what stops a sentence that straddles a boundary from being
    unfindable in both halves. The offsets are what let a citation point back
    into the original document rather than at a copy of a fragment of it.
    """
    body = text.strip()
    if not body:
        return []
    name = source or source_for(body)
    version = digest(body)
    step = max(1, size - overlap)
    spans = [(i, min(len(body), i + size)) for i in range(0, len(body), step)]
    # prevent a duplicated final chunk that only repeats the previous chunk's overlap
    spans = [(s, e) for s, e in spans if s == 0 or e - s > overlap]
    return [
        Chunk(
            text=body[start:end],
            source=name,
            version=version,
            ordinal=i,
            start=start,
            end=end,
            tenant=tenant,
        )
        for i, (start, end) in enumerate(spans)
    ]


def as_chunks(docs: Iterable[str | dict | Chunk], tenant: str = "local") -> list[Chunk]:
    """Accept what callers actually have: prose, a `{"text", "source"}` record,
    or chunks somebody already cut. Strings still work — this is the same
    `add(["..."])` the earlier workshops call — they simply arrive with a
    content-derived source instead of an anonymous integer id."""
    if isinstance(docs, (str, dict)):
        raise TypeError(f"pass a list of documents, not a {type(docs).__name__}")
    out: list[Chunk] = []
    for i, doc in enumerate(docs):
        if isinstance(doc, Chunk):
            _check_chunk(i, doc)
            out.append(replace(doc, tenant=tenant))
        elif isinstance(doc, dict):
            _check_doc(i, doc)
            _check_text(i, doc["text"])
            out.extend(chunk_document(doc["text"], doc["source"], tenant=tenant))
        elif isinstance(doc, str):
            _check_text(i, doc)
            out.extend(chunk_document(doc, tenant=tenant))
        else:
            raise TypeError(f"Document {i} is of unknown type: {type(doc).__name__}")
    return out


def _check_text(i: int, text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"doc {i}: missing 'text'")
    if len(text) > MAX_DOC_CHARS:
        raise ValueError(f"doc {i}: {len(text)} chars, max {MAX_DOC_CHARS}")


def _check_doc(i: int, doc: dict) -> None:
    text, source = doc.get("text"), doc.get("source")
    if not isinstance(text, str):
        raise ValueError(f"doc {i}: 'text' must be a string")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"doc {i}: missing or blank 'source'")


def _check_chunk(i: int, doc: Chunk) -> None:
    text, source, version = doc.text, doc.source, doc.version
    if not text.strip():
        raise ValueError(f"chunk {i}: missing 'text'")
    if source == "inline" or not source.strip():
        raise ValueError(f"chunk {i}: missing 'source'")
    if version == "":
        raise ValueError(f"chunk {i}: missing 'version'")
    if len(text) > CHUNK_CHARS:
        raise ValueError(f"chunk {i}: {len(text)} chars, max {CHUNK_CHARS}")


def texts(chunks: Sequence[Chunk]) -> list[str]:
    return [c.text for c in chunks]


Reranker = Callable[[str, list[Chunk]], list[Chunk]]


class RagStore:
    def __init__(
        self,
        docs: Iterable[str | dict | Chunk],
        tenant: str = "local",
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks: dict[str, Chunk] = {}
        self._reindex()
        self.reranker = reranker
        self.add(docs, tenant)

    def add(self, docs: Iterable[str | dict | Chunk], tenant: str = "local") -> dict[str, int]:
        old = self.chunks
        all_new = as_chunks(docs, tenant=tenant)
        unchanged = {c.doc_key for c in all_new if (o := old.get(c.id)) and o.version == c.version}
        new = [c for c in all_new if c.doc_key not in unchanged]
        replacing = {c.doc_key for c in new}
        if new:
            self.chunks = {cid: c for cid, c in old.items() if c.doc_key not in replacing}
            for c in new:
                self.chunks[c.id] = c
            self._reindex()
        updated = sum(1 for c in new if c.id in old)
        return {
            "inserted": len(new) - updated,
            "updated": updated,
            "removed": len(old.keys() - self.chunks.keys()),
            "unchanged": len(all_new) - len(new),
        }

    def _reindex(self) -> None:
        self.order = list(self.chunks.values())
        self.docs = texts(self.order)
        self.toks = [d.lower().split() for d in self.docs]
        self.df: Counter[str] = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.n = len(self.docs)
        self.avg = sum(len(t) for t in self.toks) / max(1, self.n)

    def _bm25(self, q: list[str]) -> list[int]:
        out = []
        for i, toks in enumerate(self.toks):
            tf = Counter(toks)
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = tf[term] + 1.5 * (0.25 + 0.75 * len(toks) / self.avg)
                s += idf * (tf[term] * 2.5) / denom
            out.append((i, s))
        return [i for i, s in sorted(out, key=lambda x: -x[1]) if s > 0]

    def _dense(self, q: list[str]) -> list[int]:
        qc = Counter(q)
        out = []
        for i, toks in enumerate(self.toks):
            d = Counter(toks)
            dot = sum(qc[t] * d[t] for t in qc)
            norm = math.sqrt(sum(v * v for v in qc.values())) * math.sqrt(
                sum(v * v for v in d.values())
            )
            out.append((i, dot / norm if norm else 0.0))
        return [i for i, s in sorted(out, key=lambda x: -x[1]) if s > 0]

    def _rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """fuse the two rankings with RRF and return the top k as
        `(index, fused_score)` pairs, highest first.
        Reciprocal rank fusion scores a document by `1 / (60 + rank)` in each
        list it appears in, summed.
        """
        q = query.lower().split()
        rankings = [
            self._bm25(q),
            self._dense(q),
        ]
        score: dict[int, float] = {}
        for r in rankings:
            for rank, doc_id in enumerate(r):
                score[doc_id] = score.get(doc_id, 0.0) + 1 / (60 + rank)
        ordered = sorted(score, key=lambda i: score[i], reverse=True)[:k]
        return [(i, score[i]) for i in ordered]

    def search(self, query: str, k: int = 3) -> list[str]:
        """The text of the top k. Kept as strings because this is the Workshop-2
        interface and half the course calls it; `retrieve` is the same ranking
        with the provenance attached.

        This store abstains on its own: both arms drop anything scoring zero, and
        a question sharing no vocabulary with the corpus scores zero everywhere.
        That is why it needs no threshold and a vector store does — and why every
        offline test of "the assistant abstains" passed while the deployed stack
        could not abstain at all."""
        return texts(self.retrieve(query, k))

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        # The fused rank, not a similarity, and labelled as such. Comparing this
        # number to a cosine threshold is a category error the units prevent.
        # With a reranker the order is the reranker's but the score stays RRF,
        # hence "rrf+rerank": never re-sort these by score.

        if k < 0:
            raise ValueError("k must be >= 0")

        if self.reranker is None:
            return self._candidates(query, k, "rrf")
        pool = self._candidates(query, RERANK_POOL * k, "rrf+rerank")
        return self.reranker(query, pool)[:k]

    def _candidates(self, query: str, n: int, scored_by: str) -> list[Chunk]:
        return [
            replace(self.order[i], score=score, scored_by=scored_by)
            for i, score in self._rank(query, n)
        ]
