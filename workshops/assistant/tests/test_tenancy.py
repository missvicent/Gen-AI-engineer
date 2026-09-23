"""Memory belongs to a person, and recall has to enforce that.

The shipped defect was quiet: one store for everyone, `user="me"`, with the real
caller written into a `source` label that `recall` never reads. Nothing looked
broken — alice got her own answers back — right up until bob asked a question
whose words happened to match one of her memories.

These tests are written against the leak rather than the fix, so they would fail
again if someone reintroduced a single shared store.
"""
import sqlite3
from dataclasses import replace

import pytest

from assistant.composers import ABSTAIN
from assistant.core import Assistant
from assistant.memory import AssistantMemory
from assistant.rag import Chunk
from assistant.service import build_assistant, relevance_gap
from assistant.settings import Settings
from assistant.sqlite_memory import SqliteMemory
from assistant.tenancy import TenantMemory

ALICE_FACT = "I prefer to be called Lu and I work in the Lima timezone"
BOB_QUESTION = "what timezone should we use for the Lima call"


def in_memory() -> TenantMemory:
    return TenantMemory(lambda subject: AssistantMemory(user=subject))


def on_disk(tmp_path) -> TenantMemory:
    db = str(tmp_path / "memory.db")
    return TenantMemory(lambda subject: SqliteMemory(db, user=subject))


# --- the leak ------------------------------------------------------------------
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


# --- recall through the whole pipeline -----------------------------------------
def test_the_assistant_answers_from_a_fact_the_same_caller_stated_earlier():
    """The assertion is on the ANSWER, and that is the whole point of it.

    This test used to check `answer["memories"]` and passed while the assistant
    replied "I don't know" with `Lima` sitting in the response metadata beside
    it. Recall worked; nothing consumed it. A memory the caller cannot see in
    the answer is a memory the system does not have."""
    assistant = build_assistant(Settings())
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("which timezone should we schedule in", subject="alice")
    assert "Lima" in answer["answer"]
    assert answer["grounding"] == "memory"


def test_the_assistant_does_not_answer_from_another_callers_fact():
    assistant = build_assistant(Settings())
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("which timezone should we schedule in", subject="bob")
    assert answer["memories"] == []
    assert "Lima" not in answer["answer"]
    assert answer["grounding"] == "none"


def test_a_recalled_memory_is_never_passed_off_as_a_citation():
    """Personalisation must not become fabricated grounding: a memory can colour
    an answer, but only a retrieved document earns a citation.

    Load-bearing now that memory can answer on its own — the previous version of
    this test was satisfied by an assistant that abstained."""
    assistant = build_assistant(Settings())
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("which timezone should we schedule in", subject="alice")
    assert answer["memories"]
    assert answer["citations"] == []
    # attributed to the person, not to a document that does not exist
    assert "told me" in answer["answer"].lower()


def test_a_memory_answer_says_so_rather_than_stating_it_as_fact():
    """"You told me earlier" and "Lima is your timezone" are different claims
    with different failure modes. The first is checkable by the person reading
    it; the second is the assistant vouching for something it never verified."""
    assistant = build_assistant(Settings())
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("which timezone should we schedule in", subject="alice")
    assert answer["answer"].startswith("You told me earlier:")


def test_an_unrelated_memory_does_not_license_an_answer():
    """The negative half, and the one that keeps this from being a regression.

    Recall is greedy — it returns what it has about the caller. If any memory at
    all were enough to answer, knowing alice's timezone would turn every
    unanswerable question into an answer about Lima. That is worse than the
    abstention it replaced, because it is confident."""
    assistant = build_assistant(Settings())
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("what is the refund policy", subject="alice")
    assert answer["answer"] == ABSTAIN
    assert answer["grounding"] == "none"


def test_a_document_answer_is_still_grounded_in_documents():
    """Memory does not outrank evidence: with a real document in front of it the
    assistant answers from the document and cites it."""
    assistant = build_assistant(Settings())
    assistant.ask(ALICE_FACT, subject="alice")
    assistant.ingest([{"text": "refunds are processed within five business days",
                       "source": "refunds.md"}], subject="alice")
    answer = assistant.ask("how long do refunds take", subject="alice")
    assert "five business days" in answer["answer"]
    assert answer["grounding"] == "documents"
    assert answer["citations"]


#: What `nomic-embed-text` actually scores, measured against the deployed corpus
#: and used here so the fake ranks the way the real store ranks. The pairs below
#: are the two that matter: a question answered in entirely different words scores
#: 0.66, and the nearest document to an off-topic question scores 0.51. Everything
#: about relevance in this system lives in that gap.
MEASURED = {
    ("which timezone should we schedule in", "refunds"): 0.51,
    ("which timezone should we schedule in", "expenses"): 0.43,
    ("how long do refunds take", "refunds"): 0.79,
    ("how long do refunds take", "expenses"): 0.52,
    ("how quickly do i get money back for a work trip", "expenses"): 0.66,
    ("how quickly do i get money back for a work trip", "refunds"): 0.64,
}

#: The floor from `docker-compose.yml`, midway between the two sides of that gap.
FLOOR = 0.58


class NearestNeighbours:
    """A vector store: it ranks everything, and abstains only if told how.

    Not a strawman — it is what a vector index does. BM25 returns nothing when no
    term matches, so every test above ran with an empty `contexts` list on an
    off-topic question and never exercised the branch that decides between
    document and memory grounding. Qdrant has no such thing as no match: ask it
    about timezones and it hands back the nearest rows in the corpus, which are
    about refunds.

    `floor` is the whole point of the fake. Set, it is `ASSISTANT_MIN_SCORE` and
    the store can say "nothing here". Unset, it cannot, and the deployed stack
    failed exactly there — `grounding: "documents"`, three irrelevant citations,
    and an answer of "I don't know" with the caller's own timezone sitting in the
    `memories` field of the same response, while this suite was green.

    Scores come from `MEASURED` rather than from a keyword rule, because a fake
    that decides relevance by shared words cannot show the difference between
    lexical and semantic retrieval — which is the difference this file exists to
    hold on to.
    """

    def __init__(self, docs: dict[str, str], floor: float = FLOOR) -> None:
        self.docs = docs
        self.floor = floor

    def add(self, docs: list) -> None:  # pragma: no cover - not what this proves
        pass

    def search(self, query: str, k: int = 3, tenant: str = "local") -> list[Chunk]:
        ranked = sorted(
            ((MEASURED.get((query, name), 0.3), name) for name in self.docs),
            reverse=True,
        )
        return [
            Chunk(text=self.docs[name], source=f"{name}.md", score=score,
                  scored_by="cosine")
            for score, name in ranked[:k]
            if score >= self.floor
        ]


CORPUS = {
    "refunds": "approved refunds are processed within five business days",
    "expenses": "staff are reimbursed for travel expenses within ten business days",
}


def with_neighbours(floor: float = FLOOR) -> Assistant:
    assistant = build_assistant(Settings())
    assistant.rag = NearestNeighbours(CORPUS, floor=floor)
    return assistant


def test_a_document_that_shares_no_words_with_the_question_still_answers_it():
    """The promise the whole of Phase 2 makes, asserted on the answer.

    "Money back for a work trip" and "reimbursed for travel expenses" have no word
    longer than three characters in common, which is exactly why this is the test
    worth having: it passes only if relevance is decided by the embedder's score
    and not by string overlap. A lexical filter anywhere on this path — and there
    was one, in the composer — turns this answer into an abstention while the
    document sits in the response under `contexts`.
    """
    assistant = with_neighbours()
    answer = assistant.ask("how quickly do i get money back for a work trip", subject="alice")
    assert "reimbursed" in answer["answer"]
    assert answer["grounding"] == "documents"
    assert answer["citations"][0]["source"] == "expenses.md"
    # the score that kept it travels with the citation, in named units
    assert answer["citations"][0]["scored_by"] == "cosine"
    assert answer["citations"][0]["score"] >= FLOOR


def test_irrelevant_neighbours_do_not_outrank_the_memory_that_answers():
    """The regression the vector store found and the fake now keeps.

    Retrieval returning something is not retrieval answering something. Nothing in
    this corpus is within the floor of a timezone question, so the evidence class
    is whatever is — here, what the caller said about themselves."""
    assistant = with_neighbours()
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("which timezone should we schedule in", subject="alice")
    assert "Lima" in answer["answer"]
    assert answer["grounding"] == "memory"
    # and the refund snippets that happened to be nearest are not offered as the
    # support for a statement about a timezone
    assert answer["citations"] == []


def test_relevant_neighbours_still_win_over_a_memory():
    """The other half: abstaining retrieval must not turn every document answer
    into a memory one. Ask these same neighbours something they DO answer and the
    document is the ground, cited."""
    assistant = with_neighbours()
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("how long do refunds take", subject="alice")
    assert "five business days" in answer["answer"]
    assert answer["grounding"] == "documents"
    assert answer["citations"]


def test_neighbours_that_answer_nothing_and_no_memory_still_abstain():
    assistant = with_neighbours()
    answer = assistant.ask("which timezone should we schedule in", subject="alice")
    assert answer["answer"] == ABSTAIN
    assert answer["grounding"] == "none"


def test_a_store_with_no_floor_grounds_an_answer_in_its_nearest_neighbours():
    """The misconfiguration, kept as an executable description of itself.

    With no floor the store cannot abstain, so a timezone question is answered
    from a refund policy and the response says `grounding: "documents"` with a
    citation to prove it. Nothing downstream can repair this — the composer is
    holding text that retrieval has already vouched for. That is why the fix is a
    floor rather than a second filter, and why `build_assistant` reports a vector
    store without one as degraded instead of waiting to be asked."""
    assistant = with_neighbours(floor=0.0)
    assistant.ask(ALICE_FACT, subject="alice")
    answer = assistant.ask("which timezone should we schedule in", subject="alice")
    assert answer["grounding"] == "documents"
    assert answer["answer"].startswith("approved refunds")
    assert answer["citations"][0]["source"] == "refunds.md"

    vector_store = Settings(qdrant_url="http://qdrant:6333")
    assert "cannot abstain" in (relevance_gap(vector_store) or "")
    assert relevance_gap(replace(vector_store, min_score=FLOOR)) is None
    # and BM25 needs no floor, so asking for one is not a rule about all stores
    assert relevance_gap(Settings()) is None

    assistant.settings = vector_store
    assert assistant.tier()["threshold"] == "none"
    assistant.settings = replace(vector_store, min_score=FLOOR)
    assert assistant.tier()["threshold"] == FLOOR
