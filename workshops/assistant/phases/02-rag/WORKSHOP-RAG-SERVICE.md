# Workshop · Ship a real RAG service  (ends Phase 2)

**Effort.** ~2 h of focused build time · +60 min for the integration tier · ~3.5 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

Build the assistant's retrieval core: hybrid search (keyword + vector, fused),
returning grounded chunks with an abstain path.

## Do this

```bash
cd src/workshops/assistant/phases/02-rag/before
make setup
make test
```

Open `src/assistant/rag.py`. Edit only that file. The three TODOs are `Chunk.id`,
`chunk_document`, and `RagStore._rank`.

`make test` in this folder runs only this workshop. The first failure is
`test_semantic_search`, inside `_rank`: both rankings exist, and nothing fuses
them yet. Write the RRF fusion first, then ids, then offsets. No model server
and no Docker.

Chunk id, edit, and offset checks are `tests/test_rag_chunks.py` in the same
folder. `tests/test_retrieval.py` builds the whole service, including
`guard.py`, and it is not in this folder. A failure in `guard.py` means you
are in the wrong place.

When `make test` is green, or you are stuck, diff only `rag.py` against
`../after/src/assistant/rag.py`.

## What you are not doing here

Qdrant, `docker compose`, and `POST /ask` are Workshop 8. The eval gate is
Workshop 3.

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really this, and a place to stop that is not quitting. **Full** is the version you
would show someone. **Stretch** is for when the full pass came easily.

## Minimum
- [ ] `RagStore.search(query, k)` returns relevant chunks
- [ ] Unanswerable questions get an abstention, not an invented answer

## Full
- [ ] Hybrid: exact IDs (e.g. INV-88231) are found, not just semantic matches
- [ ] Runs offline; the store sits behind an interface you could swap for Qdrant

## Stretch
- [ ] `docker compose up` with Qdrant, answering from your machine's Ollama, zero API keys — the production shape
      Workshop 8 will require anyway
- [ ] Contextual chunks (`phase2-retrieval/04-contextual-chunks`) on the slice where
      recall is weakest, with the before/after number written down

Implement `rag.py`. Tests: `tests/test_rag.py` for the walking skeleton, then
`tests/test_rag_chunks.py`. Grounding, abstention and citations that need the
whole service live in `tests/test_retrieval.py`, which is a later workshop.
Next phase: [`../03-evals/WORKSHOP-EVAL-SUITE.md`](../03-evals/WORKSHOP-EVAL-SUITE.md)
proves this one actually works.
