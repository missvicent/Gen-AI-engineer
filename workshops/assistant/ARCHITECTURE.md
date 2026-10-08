# Architecture — the composed assistant

Two diagrams: what the pieces are (and which tier each one swaps at), and what one
request actually goes through (and where the trust boundaries sit). Decisions with
alternatives and consequences live in [`adr/`](adr/); threats in
[`THREAT-MODEL.md`](THREAT-MODEL.md); incidents in [`RUNBOOK.md`](RUNBOOK.md).

## Components and tiers

Every port has two adapters: an offline one (default — zero keys, deterministic,
what the fast tests drive) and a real one (one env var each — what the composed
stack runs). The service code cannot tell which is plugged in; that indifference
is enforced by the tests, not just claimed.

```mermaid
flowchart LR
    subgraph edge["HTTP edge (api.py)"]
        MW["rate limit + concurrency cap<br/>(resilience.py)"]
        BUD["request budget<br/>(deadline.py)<br/>deadline + caller-left watcher"]
        AUTH["JWT gate — optional<br/>(auth.py)<br/>HS256 secret | RS256 via JWKS"]
        EP["/health /ingest /ask<br/>/ask/stream /approve<br/>/evidence /corpus /outbox"]
    end

    subgraph core["pipeline (core.py) — wired by the composition root (service.py)"]
        GR["guardrails.py<br/>expand + squash → scan · spotlight · output_ok<br/>+ guard.py model, when ASSISTANT_GUARD_MODEL is set"]
        PL["planner.py<br/>selects from the registry"]
        AG["agent.py<br/>loop · HITL pause"]
        OBS["observe.py<br/>root per request + span per stage, run and tool"]
    end

    subgraph ports["ports → offline | real adapter"]
        RAG["rag — chunks with derived ids<br/>InMemoryRag BM25 | Qdrant + embeddings"]
        MEM["memory — one store per subject<br/>(tenancy.py)<br/>in-process | SQLite"]
        BRAIN["composer<br/>offline stitcher | Ollama"]
        TOOLS["tools<br/>stubs | Telegram/RSS + MCP"]
        SPANS["spans<br/>in-memory | + OTLP collector"]
    end

    subgraph state["one SQLite file"]
        AUD["audit_log.py<br/>every row bound to request + trace + approval + args"]
        IDEM["idempotency.py<br/>one effect per key, original answer replayed"]
        OUT["outbox.py<br/>pending → sent | failed"]
        SMEM["sqlite_memory.py"]
    end

    MW --> BUD --> AUTH --> EP --> GR --> PL --> AG
    AG --- OBS
    BUD -. budget read at every seam .-> AG
    PL -.reads.-> TOOLS
    AG --> RAG & MEM & BRAIN & TOOLS
    AG --> OUT
    OBS --> SPANS
    EP --> AUD
    EP --> IDEM
    MEM --- SMEM
```

## One request, end to end

The trust rule that shapes the whole flow: **content is never trusted because of
where it was stored, only because of where it came from** — so user input,
retrieved documents and tool output are each screened at their own boundary, and
approval is trusted state (a `/approve` grant record bound to caller, arguments
and expiry), never text. Documents are screened once more on the way *in*
(`/ingest`), so a poisoned page is never written rather than merely never read,
and PII is redacted before it reaches disk (ADR-0010).

```mermaid
sequenceDiagram
    participant U as caller
    participant S as api.py → core.py
    participant G as guardrails
    participant A as agent loop
    participant R as rag (tenant-scoped)
    participant M as memory (subject-scoped)
    participant P as planner
    participant T as tools (hardened)
    participant L as audit log

    U->>S: POST /ask (+ Bearer JWT when auth is on)
    S->>S: rate limit · concurrency cap · verify exp/aud/iss/sub/scope -> subject
    S->>G: screen(question)
    alt injection
        G-->>S: blocked
        S->>L: policy.blocked
        S-->>U: refusal
    else clean
        S->>M: remember, then recall(subject) — this caller's store only
        M-->>S: what we already knew about them
        S->>R: search(question, tenant=subject)
        R-->>S: chunks (text + source + version + offsets)
        S->>G: screen each chunk (drop poisoned, redact PII, keep provenance)
        S->>A: run under a traced span, consume = the caller's grants
        A->>P: which tool, if any? (the GOAL and the registry — never the docs)
        P-->>A: a fully-specified call, or "answer now"
        A->>T: tool call (gated → claim a grant for THESE args, or pause)
        T->>G: screen tool output before it becomes evidence
        A-->>S: evidence + audit trail
        S->>L: tool.ran / tool.pending
        S->>S: compose (offline stitcher | Ollama, evidence + memories spotlighted)
        S->>G: output_ok(answer)
        S-->>U: answer + citations [{id, chunk_id, source, version, offsets, snippet}]
    end
```

The same pipeline serves `/ask/stream` as SSE, and the output gate is the same
gate: `output_gate.py` holds text back so a chunk reaching the client has already
been screened, rather than apologizing for one that was not. What it holds is
governed by two bounds, and it releases up to whichever is tighter: a **span**
bound, `HOLDBACK_CHARS`, derived from the patterns themselves (each declares its
longest possible match, and the gate takes the maximum, because a fixed 256 was a
guess an unbounded email pattern outgrew); and a **token** bound, the start of the
trailing run of characters that no pattern's charset can cross. The second exists
because the first is a statement about what can *match* and the leak was about what
gets *released* — a 1000-character email address matches on its last 64 characters
and the 600 in front of it belong to no match at all. Held prose costs nothing, since
prose is full of delimiters; an unbroken adversarial token buffers in full, which is
the correct behaviour for text the gate cannot yet classify. Both bounds are proved
from the compiled patterns in the test suite rather than trusted, and the property is
stated over candidate runs rather than match spans, because the first version of that
test asserted the weaker property and passed against the leaking gate (ADR-0006). A
stream that dies mid-answer ends with `truncated: true` rather than being
presented as complete. A replayed `/approve` is deduplicated by `Idempotency-Key`, and a
grant is claimed atomically by the run that spends it, so "approved once" means
"this caller, this call, fires once" (ADR-0003).

Two of the arrows above carry more weight than their size suggests. The planner
reads the **registry**, not a list of tool names, so a tool discovered from the
MCP server at boot is selectable without a code change — behind an approval,
because a tool that arrived without review is not a tool to run on a planner's
say-so. The server's `readOnlyHint` travels with it and can only make the gate
stricter; the single thing that opens one is
`ASSISTANT_MCP_READONLY_ALLOWLIST`, which is local and operator-owned. It reads the
**goal**, not the retrieved documents, so a poisoned corpus has no path to a
tool call at all (ADR-0007). Memory recall is scoped to the caller's own store
rather than filtered after the fact, which is why one person's remembered
preference cannot surface in another's answer (ADR-0008).

Recalled memory is a **third class of evidence**, not decoration on the other
two. When it is the only evidence and it bears on the question — same
content-word filter documents get — it answers, attributed ("you told me
earlier") and uncited, because a memory is not a document. The composers used to
abstain there while the recalled fact sat in the response payload beside the
refusal. `grounding` on the response names the class the answer stands on
(`documents`, `tools`, `memory`, `none`), so a reader can tell "the handbook
says" from "you said" without inspecting the citations to infer it.

Every arrow in that diagram is also a span. One root per request
(`assistant.request` over HTTP, `assistant.pipeline` when core is driven
directly) with a child per stage — auth, screen, memory, retrieval, the agent
run, compose, the output gate — so a slow request resolves to a stage rather than
to a shrug. The root names the system that answered (`service.name` on a
Resource, the model, and a prompt version *hashed from the prompt's own source*),
the retrieval span records what came back *and* what survived screening, and the
compose span carries tokens and cost from the same meter the CI cost gate reads
(ADR-0011). One request id ties the span, the response body and the
`x-request-id` header together.

What retrieval hands back is a **chunk**, not a string: text plus the source it
came from, a version hash of that source, and the character span it occupies.
That is what makes the citation on an answer checkable — `GET /evidence/{id}`
resolves it back to the exact text — and what makes `DELETE /corpus/{source}`
possible at all, since you cannot delete by prose. Chunk ids are derived from
`(tenant, source, ordinal)` rather than from a counter, so re-running the loader
updates the corpus instead of duplicating it, and an edit replaces a paragraph
instead of shelving the old one beside the new one. The embedder is injected and
its dimension measured, so `ASSISTANT_EMBED_MODEL` swaps the offline hash vector
for real semantic recall without a config change anywhere else (ADR-0012) — and
the collection is named after the embedder and its width, because Qdrant checks
dimensions and not meaning, so two 768-wide models write into each other's index
without an error anywhere.

The Qdrant search itself runs both arms of phase 2: a dense prefetch and a sparse
one, fused by Qdrant's own RRF rather than in Python, because an order number
carries no meaning for an embedder to place and a synonym carries no token for a
keyword index to match. `ASSISTANT_RERANK_MODEL` adds a cross-encoder over the
fused candidates, off by default and degrading to plain retrieval when the optional
dependency is missing. `/health` names each tier (`tier.embed`, `tier.retrieval`,
`tier.rerank`), because a stack running on the hash vector is indistinguishable from
a working one by every other probe.

**Relevance is a retrieval decision, and it is only made once.** Vector search never
returns nothing: ask about something absent from the corpus and you get its three
least-unrelated documents, at low scores, in the same shape a good answer arrives in.
`ASSISTANT_MIN_SCORE` puts a floor under the dense similarity, and it is what lets
the assistant abstain at all — `0.58` in the deployed stack, measured against that
corpus (a synonym-only true positive at 0.66, the best off-topic neighbour at 0.51),
and a store configured without it now reports `relevance` in `degraded` with
`tier.threshold: "none"` instead of quietly grounding on noise.

That floor is the **only** relevance test on the document path. There was briefly a
second one — the composer re-checked each context for words shared with the question
before letting it answer — and it broke the thing the embedder was there for: a
question phrased entirely in synonyms retrieved the right page and then had it thrown
away for using different vocabulary. A filter can only use what its layer can see,
and all a composer can see is text. So `contexts` reaching a composer now means "this
is the evidence"; empty means retrieval found nothing above the floor. `Chunk.score`
and `Chunk.scored_by` travel into the citation, so an answer shows the number that
admitted its evidence (ADR-0012, and ADR-0008 for why memory — recalled lexically —
keeps a lexical filter of its own).

Everything in that sequence runs inside **one budget**. A request carries a
deadline and a "the caller left" flag together (`deadline.py`), both installed at
the HTTP edge because that is the only layer that knows either fact, and both read
through a ContextVar at the seams: between pipeline stages, before each retry, and
before each streamed frame. That is what stops per-layer timeouts composing by
addition — a call made with four seconds of request budget left gets four seconds,
whatever its own policy says — and what stops a service generating tokens for a
caller who closed the tab. Running out of time answers 504, because somebody
should look; a caller who left answers 499, because nobody is there to read it
(ADR-0013).

The same ADR governs what happens when a request is *retried*. Every mutating
route — `/ingest`, `DELETE /corpus/{source}`, `/approve` — takes an
`Idempotency-Key` and replays the **original answer**, not a receipt, with the key
released if the operation failed so a transient error does not become a permanent
one. And every irreversible tool call writes its intent to an outbox *before* it
runs: `pending` committed first, `sent` or `failed` after. A crash mid-send then
leaves a question `GET /outbox` can answer, rather than a message that went out
and nothing that remembers it.

Every arrow into `guardrails` above is the *same object*, injected into the
pipeline by the composition root rather than imported at each call site. That is
what makes `ASSISTANT_GUARD_MODEL` a one-line decision: turning the guard model
on hardens the question, the retrieved documents, the tool output and the
ingested documents together, instead of covering the channel someone remembered
(ADR-0010). `/health` reports which screen is actually in front of the caller.
