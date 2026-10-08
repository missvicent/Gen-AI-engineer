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
from collections.abc import Iterable, Sequence
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
        """TODO 1: an id that is stable across re-ingests of the same slice of
        the same source, and distinct across tenants, sources and positions.

        `uuid.uuid5(CHUNK_NAMESPACE, key)` turns any string key into a
        deterministic UUID. The whole question is what goes in the key.

        Think hard about what must NOT be in it. If the text is part of the id,
        an edited paragraph becomes a NEW point and the old one stays behind —
        which is how a corpus quietly ends up holding three revisions of the
        same paragraph and citing whichever one happened to rank highest.
        `test_an_edit_keeps_the_id_and_changes_the_version` is that argument,
        written as an assertion."""
        raise NotImplementedError

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
    """TODO 2: split one document into overlapping chunks that remember where
    they were.

    Three things have to come out right, and each has a test:

    - The offsets are a CLAIM about the original text: `body[start:end]` must
      equal the chunk's own `text`, or every citation you write is a lie with a
      number in it.
    - Consecutive windows must OVERLAP, or a sentence that straddles a boundary
      is unfindable in both halves — the classic silent recall hole.
    - The version stamp is derived from the document body (`digest`), so an edit
      is visible without anyone remembering to bump anything.

    Empty (or whitespace-only) input produces no chunks: an empty point is a
    result that will be returned to somebody eventually."""
    raise NotImplementedError


def as_chunks(docs: Iterable[str | dict | Chunk], tenant: str = "local") -> list[Chunk]:
    """Accept what callers actually have: prose, a `{"text", "source"}` record,
    or chunks somebody already cut. Strings still work — this is the same
    `add(["..."])` the earlier workshops call — they simply arrive with a
    content-derived source instead of an anonymous integer id."""
    out: list[Chunk] = []
    for doc in docs:
        if isinstance(doc, Chunk):
            out.append(replace(doc, tenant=tenant))
        elif isinstance(doc, dict):
            out.extend(chunk_document(str(doc["text"]), doc.get("source"), tenant=tenant))
        else:
            out.extend(chunk_document(str(doc), tenant=tenant))
    return out


def texts(chunks: Sequence[Chunk]) -> list[str]:
    return [c.text for c in chunks]


class RagStore:
    def __init__(self, docs: Sequence[str | Chunk]) -> None:
        self.chunks = [d if isinstance(d, Chunk) else Chunk(text=d) for d in docs]
        self.docs = [c.text for c in self.chunks]
        self.toks = [d.lower().split() for d in self.docs]
        self.df: Counter[str] = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.n = max(1, len(self.docs))
        self.avg = sum(len(t) for t in self.toks) / self.n

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
        """TODO 3: fuse the two rankings with RRF and return the top k as
        `(index, fused_score)` pairs, highest first.

        Reciprocal rank fusion scores a document by `1 / (60 + rank)` in each
        list it appears in, summed. It needs no score calibration between the
        two arms, which is the reason to prefer it over adding raw BM25 and
        cosine numbers that live on different scales.

        Return the score along with the index, because the caller publishes it in
        the citation and a number whose units nobody recorded is worse than no
        number — see `Chunk.scored_by`."""
        raise NotImplementedError

    def search(self, query: str, k: int = 3) -> list[str]:
        """The text of the top k. Kept as strings because this is the Workshop-2
        interface and half the course calls it; `retrieve` is the same ranking
        with the provenance attached.

        This store abstains on its own: both arms drop anything scoring zero, and
        a question sharing no vocabulary with the corpus scores zero everywhere.
        That is why it needs no threshold and a vector store does — and why every
        offline test of "the assistant abstains" passed while the deployed stack
        could not abstain at all."""
        return [self.docs[i] for i, _ in self._rank(query, k)]

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        # The fused rank, not a similarity, and labelled as such. Comparing this
        # number to a cosine threshold is a category error the units prevent.
        return [
            replace(self.chunks[i], score=score, scored_by="rrf")
            for i, score in self._rank(query, k)
        ]
