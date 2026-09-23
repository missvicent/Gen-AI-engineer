"""Answer composition — how evidence becomes prose, in every tier.

Tool planning stays deterministic everywhere (that is policy, and policy should
not depend on model mood); what the model tier changes is COMPOSITION. The
offline composer stitches retrieved context together; the Ollama composers write
grounded prose from the same evidence, batch or streamed. Both consume the same
`grounded_prompt`, and both cite the same `citations_for` ids, so swapping tiers
never changes the shape of an answer.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from assistant import guardrails
from assistant.agent import Step
from assistant.rag import Chunk

ABSTAIN = "I don't know — that isn't in what I can see."

#: What the MODEL is asked to say when the evidence falls short. The reader never
#: sees it; `without_refusal` and `stop_at_refusal` turn it into `ABSTAIN`.
#:
#: The two used to be one string, and the prompt said "reply exactly: <sentence>".
#: Measured against `qwen3.5:9b` on the deployed stack, that produced an answer
#: with the refusal stapled to the end of it 6 to 8 times in 10 — on questions
#: with nothing ambiguous about them:
#:
#:     Travel expenses are reimbursed within ten business days of filing [c1].
#:     I don't know — that isn't in what I can see.
#:
#: A model that does that has not misjudged the evidence; it found the evidence,
#: answered from it, cited it, and then filled in what it took to be a required
#: field. One reply said as much in a parenthetical: "(Note: The second sentence
#: is required by the prompt's constraint format)".
#:
#: Four rewordings were measured over 8-10 trials each. None removed it, and the
#: most emphatic ("Answer or refuse — never both in one reply") made it worse by
#: turning contradictions into outright refusals of evidence the model had in
#: front of it. So the instruction is no longer asked to carry the contract. A
#: sentinel is a word an answer cannot accidentally contain, which makes the same
#: mistake detectable instead of merely discouraged, and `stop_at_refusal` cuts
#: the reply there. The model still appends it; nobody downstream can tell.
NO_ANSWER = "NO_ANSWER"

#: Word-bounded so an answer that happens to begin a longer word with it — the
#: string "NO_ANSWERS ARE FINAL" quoted out of a document, say — is not truncated
#: into a refusal.
_SENTINEL = re.compile(rf"\b{re.escape(NO_ANSWER)}\b")


def _pending(text: str) -> int:
    """How many trailing characters could still turn into the sentinel.

    A streaming filter has to hold these back: `NO_ANSW` is not the sentinel yet
    and may never be, but forwarding it and then discovering the rest arrives in
    the next frame means the sentinel already reached the client in pieces.

    The scan starts at the sentinel's full length rather than one below it, so a
    buffer ending in a complete `NO_ANSWER` is held too — until the next
    character says whether the word ended there or ran on into `NO_ANSWERS`.
    """
    for size in range(min(len(text), len(NO_ANSWER)), 0, -1):
        if NO_ANSWER.startswith(text[-size:]):
            return size
    return 0


def without_refusal(reply: str) -> str:
    """A complete model reply, resolved to exactly one of an answer or `ABSTAIN`.

    Everything from the sentinel onward is discarded rather than searched for
    meaning: when the model keeps talking past it, what follows is commentary
    about the output format, not about the question. A reply with nothing before
    the sentinel — or nothing in it at all — is the refusal.
    """
    match = _SENTINEL.search(reply)
    answer = (reply[: match.start()] if match else reply).strip()
    return answer or ABSTAIN


def stop_at_refusal(chunks: Iterator[str]) -> Iterator[str]:
    """`without_refusal` for a stream, where the reply arrives in pieces.

    Chunk boundaries fall wherever the provider puts them, so the sentinel can be
    split across any number of frames. Each frame is therefore forwarded only up
    to the point where the tail could still become the sentinel; the held-back
    remainder is reconsidered when the next frame lands, and released at the end
    of the stream if the sentinel never materialised. Nothing is held forever —
    that failure would look like a working filter and quietly clip the last few
    characters off every answer.
    """
    buffer = said = ""
    refused = False
    for chunk in chunks:
        buffer += chunk
        hold = _pending(buffer)
        settled, buffer = buffer[: len(buffer) - hold], buffer[len(buffer) - hold :]
        match = _SENTINEL.search(settled)
        ready = settled[: match.start()] if match else settled
        if ready:
            said += ready
            yield ready
        if match:
            refused = True
            break
    if not refused and buffer:
        match = _SENTINEL.search(buffer)
        if ready := (buffer[: match.start()] if match else buffer):
            said += ready
            yield ready
    if not said.strip():
        yield ABSTAIN

#: (goal, retrieved contexts, tool state, recalled memories) -> answer.
#: Memories are what the assistant already knew about THIS caller; contexts are
#: what it just looked up. They are separate arguments because they are separate
#: kinds of evidence: a citation points at a document, never at a memory.
Composer = Callable[[str, list[str], list[tuple[Step, Any]], list[str]], str]
StreamComposer = Callable[
    [str, list[str], list[tuple[Step, Any]], list[str]], Iterator[str]
]


class StreamTruncated(Exception):
    """Raised by a `StreamComposer` that stopped partway through an answer.

    Part of the streaming contract rather than an error to swallow: chunks are
    already on the wire, so the stream cannot restart, but the fragment left
    behind is not the answer either. A model that dies mid-sentence tends to
    leave something grammatical, and grammatical reads as complete — which is
    how a truncation gets filed as a bad answer instead of a failed request.
    `output_gate` turns this into an explicit `truncated` terminal event.
    """


def citations_for(contexts: list) -> list[dict]:
    """Structured citations for the response: [c1] is contexts[0] and so on. The
    grounded prompt labels evidence with the same ids, so a model-tier answer can
    reference the exact snippet it used.

    A `Chunk` cites itself properly — source document, revision, character span
    and the stable id that can fetch it again — because "source: rag" is not a
    citation, it is the name of the machine that found it. Plain strings still
    work and degrade to exactly that, which is the honest rendering of evidence
    that arrived without provenance."""
    return [
        ctx.cite(f"c{i + 1}")
        if isinstance(ctx, Chunk)
        else {"id": f"c{i + 1}", "source": "rag", "snippet": str(ctx)[:240]}
        for i, ctx in enumerate(contexts)
    ]


def grounded_prompt(
    goal: str,
    contexts: list[str],
    state: list[tuple[Step, Any]],
    memories: list[str] | None = None,
) -> str:
    """One prompt for both the batch and the streaming Ollama composers. Evidence
    is SPOTLIGHTED — wrapped as data the model must not treat as instructions —
    because screened is not the same as trusted.

    Recalled memories are spotlighted too, and for the same reason: the caller
    wrote them, which makes them useful, not trustworthy. They sit in their own
    block so the model can personalise an answer without mistaking a remembered
    preference for a cited source."""
    evidence = [f"[c{i + 1}] {guardrails.spotlight(ctx)}" for i, ctx in enumerate(contexts)]
    evidence += [f"[tool] {guardrails.spotlight(str(out))}" for _, out in state]
    lines = "\n".join(f"- {item}" for item in evidence)
    known = ""
    if memories:
        recalled = "\n".join(f"- {guardrails.spotlight(m)}" for m in memories)
        known = f"\n\nKnown about the person asking:\n{recalled}"
    # The refusal is a sentinel rather than the sentence the reader gets; see
    # NO_ANSWER for the measurements. Deliberately no clause here about wording
    # or paraphrase: "a passage answers the question when it states the fact
    # asked for, however different its wording" was measured too, and it raised
    # false refusals from 0 to 2 in 10 by giving the model a judgment to make
    # where it previously just answered.
    return (
        "Answer the question using ONLY the evidence below, in one or two "
        "sentences, citing the [c#] ids you used. If no passage states the fact "
        f"asked for, reply with this one word alone: {NO_ANSWER}\n\n"
        f"Evidence:\n{lines}{known}\n\nQuestion: {goal}"
    )


def memory_prompt(goal: str, memories: list[str]) -> str:
    """The prompt for an answer whose only evidence is what the caller said.

    Deliberately not `grounded_prompt` with the documents left out. That prompt
    demands `[c#]` citations, and a model asked to cite when there is nothing to
    cite invents an id — which is how a remembered preference gets served
    wearing a document's authority. This one has no citation slot to fill and
    asks for the attribution that is actually true: they told us.
    """
    recalled = "\n".join(f"- {guardrails.spotlight(m)}" for m in memories)
    return (
        "Answer the question using ONLY what this person told you earlier, "
        "below, in one sentence. Say that they told you. Do NOT cite sources or "
        "use [c#] markers — there are no documents here. If what they told you "
        f"does not answer the question, reply with this one word alone: {NO_ANSWER}\n\n"
        f"What they told you earlier:\n{recalled}\n\nQuestion: {goal}"
    )


def _shares_a_content_word(goal: str, context: str) -> bool:
    """Lexical overlap on words long enough to mean something. A goal with no
    content words at all cannot be judged, so its contexts get the benefit of
    the doubt rather than a reflexive abstention.

    Used for MEMORIES only, and that limit is the whole lesson of this module's
    history. It is the right test for a memory because memory recall is itself
    lexical (`memory.overlap`) — the filter speaks the same language as the store
    it is filtering. It was briefly applied to retrieved documents too, and that
    was wrong for exactly the same reason: it spoke a different language from the
    retriever, and the retriever was the one that had been made semantic.
    """
    words = {w for w in goal.lower().split() if len(w) > 3}
    if not words:
        return True
    lowered = context.lower()
    return any(w in lowered for w in words)


def relevant_memories(goal: str, memories: list[str] | None) -> list[str]:
    """The memories that could plausibly answer THIS question.

    The filter is what keeps memory from becoming a licence to answer anything.
    Recall is cheap and greedy — it returns what it has about the caller — so
    without a relevance check, knowing someone's timezone would turn every
    unanswerable question about the refund policy into an answer about Lima.
    """
    return [m for m in memories or [] if _shares_a_content_word(goal, m)]


def grounding_of(
    contexts: list[str],
    state: list[tuple[Step, Any]],
    memories: list[str] | None,
    goal: str = "",
) -> str:
    """Which class of evidence an answer stands on: documents, tools, memory, or
    nothing.

    Worth reporting because the three are not interchangeable to a reader. A
    document-grounded answer is checkable against a citation; a memory-grounded
    one is only as good as what the caller said about themselves, and has no
    citation by design (ADR-0008). Collapsing them leaves the caller unable to
    tell "the handbook says" from "you said".

    A context that came back IS evidence, and this function does not second-guess
    that. It used to: a vector store returns the nearest neighbours of any
    question, so `if contexts:` was true of every request and "documents" became
    the answer to questions nobody had asked. The fix attempted here was a lexical
    relevance filter, and it broke the more important promise — a document that
    answers the question in different words is precisely what semantic retrieval
    is for, and a word-overlap test at this layer threw those away after the
    embedder had gone to the trouble of finding them.

    Relevance belongs to the retriever, which is the only component holding the
    scores to judge it with (`adapters.QdrantStore.search`, `ASSISTANT_MIN_SCORE`).
    A store that cannot abstain is a broken store, not something for the composer
    to paper over — and `/health` reports the floor so an operator can see whether
    one is set.
    """
    if contexts:
        return "documents"
    if state:
        return "tools"
    if relevant_memories(goal, memories):
        return "memory"
    return "none"


def offline_compose(
    goal: str,
    contexts: list[str],
    state: list[tuple[Step, Any]],
    memories: list[str] | None = None,
) -> str:
    """TODO 3: the fast-tier composer — build the final answer from retrieved
    context plus any tool outputs in `state`; return ABSTAIN when there is nothing
    to ground an answer in.

    "Nothing to ground in" means an EMPTY `contexts` list, and resist the urge to
    make it mean more than that. Retrieval decides relevance, because retrieval is
    what holds the scores: BM25 abstains by scoring zero, and the vector store
    abstains at `ASSISTANT_MIN_SCORE`. A second opinion here would have only the
    text to work from, and the only judgement available from text alone is "do
    these share a word" — which throws away every document that answers the
    question in different words. That is what the embedder was for.

    Three classes of evidence, in order. A grounded context answers, with tool
    output in parentheses and memories appended in brackets. Tool output alone
    answers. And memory alone answers TOO, when `relevant_memories` finds one
    that bears on the question — attributed ("You told me earlier: ..."), never
    stated as fact, and never cited, because a memory is not a document. That
    last branch is the bug this lesson exists for: recall worked, nothing
    consumed it, and the assistant replied "I don't know" with the answer
    sitting in its own response payload.

    An IRRELEVANT memory is still nothing to ground in. Recall is greedy; if any
    memory at all licensed an answer, knowing a caller's timezone would turn
    every unanswerable question into a confident answer about Lima."""
    raise NotImplementedError


def model_composer(generate: Callable[[str], str]) -> Composer:
    """Real-tier composer: grounded generation against a model. Abstention on
    empty evidence stays DETERMINISTIC — a model given nothing to ground on should
    not be asked to improvise a refusal.

    Takes the completion callable rather than a host and a tag, so the prompt
    construction and the refusal contract below are written once and hold for
    every provider `providers.py` can build. A second backend must not be a
    second copy of this function; that is how two brains end up disagreeing about
    what counts as a refusal.
    """

    def compose(
        goal: str,
        contexts: list[str],
        state: list[tuple[Step, Any]],
        memories: list[str] | None = None,
    ) -> str:
        recalled = relevant_memories(goal, memories)
        if not contexts and not state and not recalled:
            return ABSTAIN

        # Memory-only questions get their own prompt, because the grounded one
        # asks for [c#] citations and there is nothing here to number. Sending
        # them to the grounded prompt is how "which timezone should we schedule
        # in" got answered with "I don't know [c1][c2]" while the caller's own
        # timezone sat in the same response: the instructions permitted the model
        # to use only the documents, and the documents were about refunds.
        prompt = (
            memory_prompt(goal, recalled)
            if not contexts and not state
            else grounded_prompt(goal, contexts, state, memories)
        )
        return without_refusal(generate(prompt))

    return compose


def word_stream(compose: Composer) -> StreamComposer:
    """Adapt any batch composer into a streaming one by yielding word-sized chunks.
    The fast tier streams THROUGH the same SSE path the model tier uses, so the
    endpoint's behavior is testable offline."""

    def stream(
        goal: str,
        contexts: list[str],
        state: list[tuple[Step, Any]],
        memories: list[str] | None = None,
    ) -> Iterator[str]:
        words = compose(goal, contexts, state, memories or []).split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word

    return stream


def model_stream_composer(
    stream_tokens: Callable[[str], Iterator[str]],
) -> StreamComposer:
    """Real-tier streaming: tokens leave the model and reach the client as they
    are produced, not after the full completion lands.

    Takes the streaming callable for the same reason the batch composer takes the
    completion one.
    """

    def stream(
        goal: str,
        contexts: list[str],
        state: list[tuple[Step, Any]],
        memories: list[str] | None = None,
    ) -> Iterator[str]:
        recalled = relevant_memories(goal, memories)
        if not contexts and not state and not recalled:
            yield ABSTAIN
            return

        # Same choice as the batch composer, on the same terms. Streaming and
        # batch answering the same question differently is a bug a client sees
        # and nobody can explain.
        prompt = (
            memory_prompt(goal, recalled)
            if not contexts and not state
            else grounded_prompt(goal, contexts, state, memories)
        )
        yield from stop_at_refusal(stream_tokens(prompt))

    return stream
