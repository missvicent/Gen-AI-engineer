# Workshop · Deployed stack  (ends Phase 8)

**Effort.** ~8 h of focused build time · +2 h for the integration tier · ~15 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

The assistant works, it is hardened, and it speaks MCP. It is also a black box that
costs an amount nobody has measured and takes a length of time nobody has bounded.
This layer closes that: **see it, then make it cheaper without making it worse.**

Do the work in six folders, in this order. Each one has its own `before/`, its own
`make test`, and a README that names the first failure. The rest of this file is
the map of the finished system.

| Folder | You edit | First command |
|--------|----------|---------------|
| [`08-trace/before`](08-trace/before/) | `observe.py`, `cache.py`, `usage.py` | `uv run pytest -q tests/test_observe.py` |
| [`08-corpus/before`](08-corpus/before/) | `adapters.py` | `uv run pytest -q tests/test_corpus.py::test_reingesting_the_same_source_updates_instead_of_accumulating` |
| [`08-reliability/before`](08-reliability/before/) | `deadline.py`, `resilience.py`, `idempotency.py`, `outbox.py`, `approvals.py` | `uv run pytest -q tests/test_resilience.py::test_a_bug_is_not_retried_three_times` |
| [`08-service/before`](08-service/before/) | `service.py`, `api.py`, `core.py`, `composers.py`, `connectors.py`, `output_gate.py`, `audit_log.py` | `uv run pytest -q tests/test_service.py::test_health_reports_the_offline_tier` |
| [`08-auth/before`](08-auth/before/) | `auth.py`, `oauth.py` | `uv run pytest -q tests/test_auth.py::test_without_a_key_source_the_zero_key_demo_path_stays_open` |
| [`08-evidence/before`](08-evidence/before/) | `report.py` | `uv run pytest -q tests/test_report.py::test_build_portfolio_is_still_the_page` |

`release.py` and `evidence.py` are already finished. The defect lab stays in
`after/`, because that is the tree the Dockerfile builds.

Two modules, in this order, because the second one is only safe once the first exists.

## Architecture

```
assistant.request              service.name on a Resource · request.id · http.route
├── auth.verify                auth.mode · auth.accepted · why it was refused
└── assistant.pipeline         llm.model_name · llm.prompt_template.version (DERIVED)
    ├── guardrail.screen       guardrail.blocked · reason
    ├── memory.recall          enduser.id · how many memories came back
    ├── rag.search             documents.count vs documents.kept ← the security signal
    │                          retrieval.sources: which documents actually answered
    ├── agent.run ─────────┬── tool.read_emails    traced_registry: no tool edited,
    │   steps · paused     └── tool.send_telegram  none can forget
    ├── llm.compose            token_count.total · cost.usd (the same meter CI gates)
    └── guardrail.output       request.abandoned, when the caller stopped listening

one budget per request ──► deadline + "caller left", read at every seam
      504 out of time · 499 nobody there · each call's timeout capped by what's left

mutations ──► Idempotency-Key ──► the ORIGINAL answer replayed, key released on failure
irreversible ──► outbox: pending BEFORE the call ──► sent | failed ──► GET /outbox

spans ──► time_by_tool · slowest_tool · gated_tool_calls  (safety report)
              │
              ▼
goal ──► cached_run(goal, run, cache) ──► AnswerCache
              look · run · OFFER              refuses: paused runs,
                                              side-effecting runs, step-cap
                                              runs, empty answers
```

## The seam

`before/src/assistant/observe.py` — `recorder` (the Resource), `request_scope`,
`stage`, `traced_tool` / `traced_registry`, `traced_run`, and the readers
(`tool_calls`, `time_by_tool`, `slowest_tool`, `failed_spans`, `children_of`,
`descendants_of`, `one_trace`, `gated_tool_calls`). The attribute names are given —
use them rather than inventing your own, or your trace is legible only to you.
`streaming_root` / `within` / `child_of` / `mark` are given too, and worth reading
before you use them: they exist because "the current span" is a ContextVar, and a
generator that yields between spans cannot rely on one.

`before/src/assistant/provenance.py` — the derived stamps (`prompt_version`,
`corpus_version`, `dataset_version`) shared by the spans and the CI report.

`before/src/assistant/usage.py` — one token-and-cost meter, used by the compose span
and by the cost gate.

`before/src/assistant/core.py` — `_root_attributes`, `_usage_attributes`, `pipeline`,
`_composed`, and a span per stage inside `_gather` and `ingest`.

`before/src/assistant/cache.py` — `answer_key`, `is_cacheable`, `AnswerCache.get` /
`offer`, and `cached_run`.

`before/src/assistant/rag.py` — `Chunk.id` and `chunk_document`: the two decisions
that make a corpus operable. `before/src/assistant/adapters.py` — `InMemoryRag.add`
and `.delete`, `QdrantStore.add`, the measured embedding dimension, and the two MCP
policies (discovery may retry, invocation may not).

`before/src/assistant/deadline.py` — `Budget`, the ContextVar, and `capped`: one
answer to "is this still worth doing?" for both the clock and the caller.
`before/src/assistant/resilience.py` — `TRANSIENT`, `is_transient`, `ONCE`, and the
three things that stop a retry loop. `before/src/assistant/connectors.py` — the
raise-inward/apologise-outward split that makes a retry policy real.

`before/src/assistant/idempotency.py` — `recall` / `release` / `store` / `run`: a
replay returns the original answer, and a failure stays retryable.
`before/src/assistant/outbox.py` — `reserve` / `settle` / `pending` and
`recorded_registry`: the intent is committed before the effect.
`before/src/assistant/api.py` — the request budget middleware, the 504/499 split,
the `once` helper on every mutating route, and `GET /outbox`.

## Why the tools are wrapped, not edited

Tracing is a cross-cutting concern, and cross-cutting concerns rot when they live
inside every implementation: somebody adds a tool, forgets the decorator, and six
weeks later there is a hole in the trace nobody can explain. `traced_registry` makes
instrumentation a property of the **seam**, so a new tool is traced whether its author
thought about it or not.

## Why caching an agent is not caching RAG

A RAG answer is a pure function of a question and a corpus. An agent run sends
messages and books meetings. If you cache "text my boss the summary", the second
request returns instantly — and no message is sent. Nothing errors. Nobody is paged.
The user finds out from their boss.

That is why `offer` exists instead of `put`: the caller proposes an answer and the
policy decides, reading the **trace** to see which gated tools actually fired. The
observability module isn't just a dashboard feed; it is the input to a safety
decision.

## Deliverables

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really this, and a place to stop that is not quitting. **Full** is the version you
would show someone. **Stretch** is for when the full pass came easily.

This is the longest list in the course, and reading it end to end is the fastest
way to talk yourself out of starting. So the **minimum** is three things, and they
are the three at the bottom of the capstone list below:

1. the stack comes up with one command and zero API keys;
2. CI runs tests, a smoke eval and the red-team suite as required checks;
3. it is deployed somewhere real, with secrets, a health check and a rollback path.

Ship those and you have a deployed, gated system — the thing the phase is named
after. Everything else on this page is the **full** pass: the observability,
reliability and operability work that turns a running service into one you can
debug at 3 a.m. Take it in any order; each item is independently green-able.

- [ ] One request is **one trace**: a root span with a child per stage — auth, screen,
      memory, retrieval, the agent run, compose, the output gate — and `one_trace`
      asserts it, because a stage that opens its span outside the request's context
      still records, just somewhere nobody will look
- [ ] The service has an **identity**: `service.name` on a `Resource`, not just a
      tracer name. Without it the SDK ships `unknown_service` and the filter that
      should narrow to one service narrows to nothing
- [ ] The root says **which system answered** — model, and a prompt version derived by
      hashing `grounded_prompt`'s own source rather than typed next to it — and the
      same derivation stamps the CI report, so a trace and the report agree by
      construction
- [ ] `rag.search` records **hits and kept**, not just one of them: the gap is how many
      documents the screen threw away, and one number cannot show a gap — plus
      `retrieval.sources`, which turns "retrieval was slow" into "the slow ones all hit
      the same document"
- [ ] Re-ingesting a document **updates it**. Chunk ids derive from
      `(tenant, source, ordinal)`, so the loader can be re-run — after a crash, after a
      config change, on a cron — without the corpus growing a copy each time. A test
      ingests the same source three times and asserts one chunk survives
- [ ] An **edit keeps the id and changes the version**. Content is deliberately not in
      the id: hash the text in and every edit shelves the old paragraph beside the new
      one, and retrieval cites whichever ranked higher
- [ ] Chunk **offsets are checkable**: `body[start:end] == chunk.text`, asserted. A
      citation with a character range that does not hold is a lie with a number in it
- [ ] `DELETE /corpus/{source}` **removes every chunk of one source, tenant-scoped**,
      and leaves an audit row. A corpus you cannot delete from will eventually hold
      something you are not allowed to keep; a delete that crosses tenants is a
      denial-of-service with a REST interface
- [ ] `GET /evidence/{chunk_id}` **resolves a citation back to its text**, and 404s once
      the source is forgotten. That round trip is the difference between a citation and
      a decoration
- [ ] A **redacted** chunk keeps its source, version and offsets — otherwise every
      document containing an email address becomes uncitable, which is a quiet incentive
      to screen less
- [ ] `ASSISTANT_EMBED_MODEL` swaps the hash vector for **real embeddings**, and the
      collection's dimension is **measured from the injected embedder**, not declared. A
      hardcoded `64` is a 400 from Qdrant on the first write after the deploy
- [ ] The **collection is named after the embedder and its width**. Qdrant checks the
      dimension of an incoming vector and nothing else, so two 768-wide models write into
      each other's index, search cleanly and return noise — a corruption with no error
      in it anywhere
- [ ] Search runs **both arms**: a dense prefetch and a sparse one, fused by Qdrant's own
      RRF, with the tenant filter on **each** — a filter applied to one of two prefetches
      is not a filter
- [ ] `ASSISTANT_MIN_SCORE` puts a floor under the **dense** score, not the fused one.
      Fused scores are reciprocal ranks: the top hit of a search that found nothing
      relevant still scores like a top hit
- [ ] The floor is **set, and it is measured**. `0.58` in `docker-compose.yml`, from a
      synonym-only true positive at 0.66 and the best off-topic neighbour at 0.51. A vector
      store with no floor reports `relevance` in `degraded` and `tier.threshold: "none"`,
      and `require_real_tiers` refuses to publish release numbers measured without one —
      recall scored against a store that cannot say "nothing here" is not a recall number
- [ ] **Admission is per arm.** The cosine floor judges dense candidates; a sparse
      candidate is admitted when a distinctive query term — one carrying a digit, or
      written in caps — appears in it verbatim. Applying the cosine floor to both arms
      looks tidier and silently disables hybrid retrieval: `ZX-99417` scored 0.535 dense
      against a floor of 0.58, on the one document that contained it exactly. The chunk
      carries the arm that admitted it in `scored_by`, because 1.96 read as a cosine is
      worse than no number
- [ ] The sparse rule stays **narrow on purpose**. Admitting on any shared word would
      restore the failure the floor was added to fix, from the other direction: "work"
      appears verbatim across an office corpus, and a system that admits on it never
      abstains again
- [ ] Admission is the **only** relevance test on the document path. Nothing downstream
      re-checks a retrieved chunk for words shared with the question. A word-overlap filter
      standing behind a semantic retriever throws away exactly the hits it was bought to
      find, and it is invisible from the API: "how quickly do i get money back for a work
      trip" retrieves the travel-expenses page at 0.66, returns it in `contexts`, and
      answers "I don't know"
- [ ] Each citation carries `score` and `scored_by`, so an answer shows the number that
      admitted its evidence. `scored_by` matters more than `score` — `cosine` and `rrf` are
      different units and a client comparing across them is reading noise
- [ ] A failure is **classified before it is retried**: `TypeError` and a 4xx surface on
      the first attempt, a dead socket gets another go. `retry on Exception` buys three
      times the latency before the same 500, with the traceback pointing at the third
      identical retry instead of at the cause
- [ ] `send_telegram` retries **exactly zero times** (`ONCE`), and so does every MCP tool
      call. A timeout tells you nothing about whether the server acted, and nothing in
      the MCP protocol says whether calling a discovered tool twice charges a card twice
- [ ] Each connector **raises inward and apologises outward** — a connector that catches
      its own `OSError` and returns `{"error": ...}` is a connector nothing can retry,
      because from the outside a returned error dict is a successful call. That is how a
      retry policy ends up decorative
- [ ] One **budget per request** (`REQUEST_DEADLINE_SECONDS`), read at every seam, with
      `deadline.capped` shrinking each call's timeout to what is left. Without it every
      layer's timeout composes by *addition* and the total is a number nobody has
      computed
- [ ] A caller who **left** stops the work: the disconnect is watched at the HTTP edge
      and checked per streamed frame, because a long generation cannot be stopped
      anywhere else. Under load this is the failure that compounds
- [ ] **504 for a deadline, 499 for a disconnect** — a 504 is an alert, a 499 is
      somebody closing a tab, and conflating them fills a dashboard with incidents that
      are really just people leaving
- [ ] **Every** mutating route is idempotent, not just `/approve`, and a replay returns
      the **original answer**. `{"replayed": true}` avoids the double effect and still
      breaks the client, which asked a question and got a receipt
- [ ] A **failed** operation releases its key, so the retry the client is about to send
      actually runs. Recording the key up front turns one transient failure into a
      permanent one, silently
- [ ] Keys are namespaced by **subject and operation**: they are client-chosen, "retry-1"
      is what everyone picks, and one flat namespace lets a retried ingest swallow an
      approval
- [ ] Irreversible intent is **durable before the effect** — a `pending` outbox row
      committed before the call, settled after, and `GET /outbox` lists what never
      settled. A crash mid-send should leave a question, not silence
- [ ] `llm.compose` carries **tokens and `cost.usd`** from the same meter `report.py`
      totals for the cost gate — two meters would be two answers to "what did last week
      cost"
- [ ] One **request id** on the span, in the response body and in the `x-request-id`
      header, honouring an inbound one, so a user quoting an id from an error message
      resolves to a trace without joining on timestamps
- [ ] A refused request is **visible**: a 401 leaves an `auth.verify` span with a
      reason and no pipeline beneath it; a blocked question leaves a screen span and no
      `llm.compose` at all — the cheapest possible proof the block happened before the
      spend
- [ ] Streaming produces the **same tree** as the batch path, and stays streaming — a
      test fails if the metering buffers the generator to make the token count tidier
- [ ] Every tool is traced by **wrapping the registry** — a test proves a newly-added
      tool is instrumented without its author doing anything
- [ ] A failing tool is marked **ERROR on its span and still raises** — the trace
      records the failure, it does not absorb it
- [ ] The root `agent.run` span carries **step count and whether it paused**, so a
      run that stopped for a human is distinguishable from one that hit the cap
- [ ] The same root span tells the **approval story**: which grants the run started
      with (`agent.approved_tools`), which tool a paused run waits on
      (`agent.pending_tool`), and a single `agent.outcome` — `completed`,
      `paused_for_approval` or `policy_violation` (the last one is span-status
      ERROR, because it is the trace a reviewer must never scroll past)
- [ ] `gated_tool_calls` reports **which irreversible tools actually fired**, and a
      test proves a contained run reports none
- [ ] `time_by_tool` / `slowest_tool` answer **where the wall clock went**, read off
      the spans rather than from a timer you sprinkled around
- [ ] The cache **refuses** a paused run, a side-effecting run, a step-cap run and an
      empty answer — one test per rule, because each is a different way to break trust
- [ ] A repeated read-only question **does not rerun the agent**; a side-effecting one
      reruns **every time**, and a test asserts both
- [ ] The whole layer runs offline: `InMemorySpanExporter` in tests, no collector, no
      vendor account

## The operate drill: diagnose from the trace, not the code

Production failures do not announce which tool broke. `tests/test_diagnosis.py`
seeds the two classic incidents — one tool quietly degraded, one intermittently
failing — and diagnoses them **from the spans alone**:

1. **Symptom** — the `agent.run` root span got slow (`duration_ms`).
2. **Localise** — `time_by_tool` / `slowest_tool` name the culprit; no code read.
3. **Confirm** — `failed_spans` separates *broken* (retry/fix upstream) from
   *slow* (capacity/caching) on the same trace, because the two need different
   fixes.

The drill also shows why the CI gate budgets **P99, not the mean**: one degraded
call in ten barely moves the average and owns the tail. Rehearse it offline, then
run the same three readers against the collector copy of the spans in the deployed
stack — that is the "diagnose" step of `RUNBOOK.md`, practised before the pager
goes off.

## The capstone: one running service

Everything above still treats the assistant as a library. The capstone makes it a
service, organized so each module owns exactly one concern — `service.py` is the
**composition root** (the one file where adapters are chosen), `api.py` is the
HTTP surface, `core.py` is the request pipeline, `composers.py` turns evidence
into prose, `screening.py` guards the trust boundaries, and `fallbacks.py` keeps
a dead adapter from becoming a dead service. Together they wire every layer you
built:

```
/ingest  ─► screen ────────────► RAG  (refused rows are audited and counted back
           (before it is stored)       to the caller; PII never reaches disk)

request ─► screen ────────────► agent.run (traced) ─► tools (read-only output
           (input)                │                    re-screened; irreversible
                                  │                    ones gated by /approve)
                                  ├─► RAG   (in-memory BM25  | Qdrant + embeddings)
                                  │          chunks: text + source + version + span
                                  ├─► memory (in-process     | SQLite)
                                  └─► spans  (in-memory      | + OTLP)
           guardrails.output_ok ◄─ answer
```

One `screen` object serves all four arrows, injected by the composition root, so
`ASSISTANT_GUARD_MODEL` hardens every untrusted channel at once instead of the
one someone remembered.

Offline and deterministic by default — the fast tests drive the real FastAPI app
with a TestClient and no network. Each real adapter turns on with one env var
(`settings.py`): `QDRANT_URL`, `ASSISTANT_EMBED_MODEL` (real embeddings instead of
the deterministic hash vector — it matches on shared vocabulary, so
"reimbursement" will not find a page about "refunds"), `ASSISTANT_MIN_SCORE` (the
dense relevance floor that lets retrieval return nothing instead of the nearest
thing it has — the deployed stack sets `0.58`, the sparse arm is admitted by exact
identifier match instead, and a vector store without a floor is reported as
degraded), `ASSISTANT_RERANK_MODEL` (an optional cross-encoder over the fused
candidates), `OLLAMA_HOST`,
`MCP_SERVER`, `ASSISTANT_DB`,
`OTEL_EXPORTER_OTLP_ENDPOINT`, plus the optional hardening/connector vars —
`ASSISTANT_JWT_SECRET` **or** `ASSISTANT_JWKS_URL` (Bearer JWT on every mutating
endpoint, `auth.py`; the first verifies HS256 against a shared secret, the second
verifies RS256 against an issuer's published keys), `ASSISTANT_JWT_ISSUER` and
`ASSISTANT_JWT_LEEWAY` to tighten it, `ASSISTANT_GUARD_MODEL` (a local model as a
second opinion on every untrusted string — it may add a block, never clear one,
and fails open to the deterministic verdict; `guard.py`), and
`TELEGRAM_BOT_TOKEN` / `NEWS_FEED_URL`
(real connector bodies, `connectors.py`; the approval gate and output re-screen
apply unchanged). `adapters.py` holds the real implementations,
`sqlite_memory.py` is the memory store that survives a restart, and
`mcp_server.py` is a small real MCP server the assistant discovers tools from.

`auth.py` is the resource server: it checks whatever token arrives. `oauth.py` is
the other half — authorization code + PKCE, discovery, exact redirect matching —
because "we support OAuth" is easy to claim and easy to get subtly wrong. The two
are exercised together by `phase8-deploy/01-compose`'s OAuth overlay, which puts
a real Keycloak in front of the assistant and swaps the shared secret for JWKS
without the service code noticing.

The HTTP surface is `/health`, `/ingest`, `/ask`, `/ask/stream` (the same answer
as server-sent events, chunk by chunk), and `/approve`. Grounded answers carry
structured `citations` (`[{id, source, snippet}]`) that the model-tier prompt
labels with the same `[c#]` ids.

The `Dockerfile` builds one image that runs as both the assistant API and the MCP
server; `phase8-deploy/01-compose` deploys it next to a pinned Qdrant, answering
from the Ollama on your own machine through `host.docker.internal`, and
`src/verify-e2e.sh` proves the composed stack end to end, under the secure
profile so every request is authenticated: boot on healthchecks, tier report, an
unauthenticated request refused, grounded answer, approval containment bound to
the approving subject, an MCP call over the wire, two callers whose memories never
cross, spans left behind — and, via the `docker-compose.observability.yml` overlay,
the same spans arriving at a real otel-collector **outside the process**.

### Capstone deliverables

- [ ] `service.py` + `core.py` compose **every** layer — guardrails on input,
      hardened tool output, gated tools behind `/approve`, RAG contexts, memory
      writes, spans — and the fast tests prove each seam offline
- [ ] `/ask/stream` streams the same gated pipeline as SSE, grounded answers carry
      structured citations, and **nothing the gate releases can belong to a token it
      later blocks**. Two bounds, tighter one wins: the longest span any output pattern
      can match, and the start of the trailing character run no pattern's charset can
      cross. Only the first is intuitive and only the first was there — a 1000-character
      email address matches on its last 64 characters, which leaves 600 characters of
      address that are part of no match and were released. The suite derives both bounds
      from the compiled patterns and states the property over candidate *runs*, because
      the version stated over match *spans* passed 300 out of 300 cases against the
      leaking gate (`tests/test_stream.py`, ADR-0006)
- [ ] The JWT gate rejects missing, expired,
      **never-expiring**, subject-less, wrong-audience, wrong-issuer and
      wrong-scope tokens — in both the shared-secret and JWKS lanes, and against
      an algorithm-confusion attempt (all proven offline in `tests/test_auth.py`)
- [ ] The token-minting half is real too: PKCE with `S256`, a `state` compared in
      constant time, and a redirect URI that must match exactly
      (`tests/test_oauth.py`; run it against Keycloak with the OAuth overlay)
- [ ] The screen reads two surfaces before it decides — the input **expanded**
      (base64, percent-encoding, HTML entities) and **squashed** (NFKC, invisible
      characters, leet folding, separator removal) — with controls proving
      benign prose still gets through (`tests/test_guardrails.py`), and the
      optional guard model can only ever add a block (`tests/test_guard.py`)
- [ ] Security depth holds offline: a poisoned document is refused **at ingest**
      so it is never stored, poisoned retrieved documents are still dropped
      before composition, tenants cannot cross-read documents or memories, every
      approval and tool run lands in the persistent audit log — every row bound
      to the same request id, trace id, subject, approval id, canonical args
      hash and outcome the spans carry, so "what did this trace do" is a query
      and not a regex over prose — and a replayed
      approval never fires twice (`tests/test_security.py`,
      `tests/test_reliability.py`; threat model in `THREAT-MODEL.md`, incident
      response in `RUNBOOK.md`)
- [ ] Memory survives a **process restart** (`sqlite_memory.py`; the tests reopen
      the store cold)
- [ ] A recalled memory **answers the question** rather than riding along beside a
      refusal — attributed, uncited, and reported as `grounding: "memory"`. Assert on
      the answer: the version of this test that read `answer["memories"]` passed
      against an assistant that replied "I don't know" with the fact in its own payload
- [ ] An **irrelevant** memory grants nothing. Recall is greedy, so without a relevance
      filter knowing one fact about a caller would answer every question the corpus
      could not
- [ ] A discovered tool is **gated by default**, and the server's `readOnlyHint` cannot
      change that. The old default read an absent `requires_approval` key as permission,
      so a server ungated every tool it published by staying quiet about them
- [ ] `ASSISTANT_MCP_READONLY_ALLOWLIST` is the only thing that opens the gate, hints can
      only tighten it, and `tier.mcp_ungated` on `/health` says how many tools the
      operator has actually reviewed
- [ ] MCP tools arrive by **discovery through the real SDK** (`adapters.mcp_tools`
      against `mcp_server.py`, in-memory in the integration lane, over HTTP in the
      composed stack)
- [ ] `docker compose up --build` in `phase8-deploy/01-compose/after` reaches
      healthy, the `docker-compose.secure.yml` overlay turns the gate on, and
      `./verify-e2e.sh` passes all fifteen checks — including the operate tier:
      authenticated requests only, a citation that resolves back to its text and a
      source that can be deleted, per-subject memory isolation proven over HTTP,
      spans observed at a real collector outside the process, degraded-but-honest
      answers with Qdrant stopped, and state that survives a restart (grounded
      answers, audit rows, memories)
- [ ] The answers in that run came from the **model**, not the fallback composer —
      check 4 reads `degraded.brain` off `/health` and fails if the composer fell
      back. Worth doing on your own service: "grounded in the corpus" and "written
      by the model" are different claims, and the offline composer satisfies the
      first. This suite went green on all fifteen checks while the model was timing
      out on every single request (`phase8-deploy/VERIFIED.md`)
- [ ] `make report` writes `PORTFOLIO.md` — eval scores per slice, live red-team
      containment, latency percentiles read off the spans, the cost story and the
      ADR list, every number measured by `src/assistant/report.py` and the
      generator itself tested (a breach cannot pass silently) — and the page
      opens with a provenance block naming every instrument, because the number
      most likely to be misquoted is the one from the cheapest tier
- [ ] `make release-evidence` runs the same harness against the DEPLOYED stack:
      Qdrant with the semantic embedder, hybrid retrieval, reranking on, a RAGAS
      0.4 judge on a pinned model, and all 58 rows of the Phase 6 red-team
      dataset with its eleven benign controls. It refuses to run on a fallen-back
      component, and it reports bypasses and false positives side by side — so
      "contained everything" cannot be bought by refusing everything
- [ ] The SAME run writes `evals/report.json`, version-stamped with values
      derived from what ran (the `prompt` stamp is a hash of the prompt builder's
      source, so editing the prompt and keeping the label is not possible), and
      `make gate` puts the four merge gates from `phase8-deploy/02-ci` in front
      of it. The root workflow's `evidence` job does the same thing against the
      **image** built from the current commit — a gate reading a committed report
      only ever checks that somebody remembered to edit a file
- [ ] The release lane is code, not a wiki page
      (`phase8-deploy/03-deploy-observe/src/release.py` + `deploy/`): only commit
      SHAs are published and a dirty tree is refused, the manifest names its
      secrets and a pasted key fails the deploy, four smoke probes run after it
      — including `/health`'s `version` against the SHA just shipped, which is
      the only one that catches a half-finished rollout still serving a healthy
      old machine — and the rollback returns `halt` rather than pretending, when
      the only thing behind you is a moving tag
- [ ] Backup and restore are scripted and *proven*: SQLite's online backup rather
      than `cp` (the test races a writer against it), row counts verified in the
      same script that takes the copy, retention by name, and the writer stopped
      before a restore
- [ ] The design is documented like a system, not a homework: diagrams in
      `ARCHITECTURE.md`, decisions with alternatives and costs in `adr/`, threats
      in `THREAT-MODEL.md`, incidents, deploy failures and backup/restore in
      `RUNBOOK.md`

## Stretch goals

- Ship the spans to a real backend. The observability overlay already proves they
  leave the process; swap its debug exporter for an `otlp` exporter at a local
  Phoenix or a Langfuse project and look at the actual tree. Change **nothing** in
  `observe.py` — if you have to, your instrumentation is not portable yet.
- Find the single most expensive request you have ever made and explain, from the
  trace alone, why it cost that. `cost.usd` is already on `llm.compose`; the skill is
  reading a tree back.
- Add a **semantic** layer to the cache (`phase8-deploy/04-cost-latency`) and sweep the
  threshold on your own traffic. Report the wrong-reuse count, not just the hit rate.
- Gate the deploy on the tail: fail CI when P99 across your golden set exceeds the
  budget, using the guard from `03-deploy-observe`.

Implement `observe.py`, `provenance.py`, `usage.py`, the span seams in `core.py`,
and `cache.py`. Reference and tests: `after/src/assistant/observe.py`,
`after/src/assistant/cache.py`, `after/tests/test_observe.py` (the loop),
`after/tests/test_tracing.py` (the whole request), `after/tests/test_cache.py`.
Decision record: [`adr/0011`](adr/0011-one-trace-per-request-and-stamps-that-derive.md).

**Measure before you optimize, and report the pair.** A cache that halves your bill and
loses two points on the Phase-3 gate is not a win, it is an undeclared quality cut. The
order — see it, then cache it, then route it — exists so that when the score moves you
know exactly which rung to blame.
