# Workshop · Ship a real RAG service  (ends Phase 2)

**Effort.** ~2 h of focused build time · +60 min for the integration tier · ~3.5 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

Build the assistant's retrieval core: hybrid search (keyword + vector, fused),
returning grounded chunks with an abstain path.

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
`tests/test_retrieval.py` — which is where grounding, abstention and citations are
actually proved, and where the Full pass is judged.
Next layer: `WORKSHOP-EVAL-SUITE.md` proves this one actually works.
