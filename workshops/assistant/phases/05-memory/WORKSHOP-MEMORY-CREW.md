# Workshop · Memory + research crew  (ends Phase 5)

**Effort.** ~3.5 h of focused build time · +30 min for the integration tier · ~5.5 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

The assistant currently starts every session as a stranger and does every job on the
frontier model. This layer adds the two things that make it feel like software: **memory
with an expiry date** and **delegation with a receipt**.

Both plug into the eval layer from Phase 3 — recall becomes golden-set rows, and the cost
comparison sits *next to* the quality score rather than replacing it.

## Architecture

```
turn ──► classify ──► write(kind, text, source=…, ttl=…) ──► THIS SUBJECT's store
                                                              │  four kinds
user correction ──► correct(id, …) ──► old row DELETED         │  provenance + TTL
                                                              ▼
task ──► context_for(memory, task, guardrails, budget) ──► Context
              pins first · rank · park what doesn't fit    tokens/budget receipt

TenantMemory(make) ──► store_for(subject) ──► one store per person
              recall never crosses a subject · all() deliberately does

jobs ──► delegate(jobs, route=tiered) ──► CrewRun
              supervisor plans once (free tier)     cost per agent
              workers own their kind                quality vs min_tier
```

## The seam

`before/src/assistant/memory.py` — `write`/`recall`/`forget`, the `classify` router that
must return `None` for most turns, and the budgeted `assemble`.

`before/src/assistant/tenancy.py` — `store_for`, which decides whose memory a
recall is allowed to see. Read the module docstring before you implement it: the
bug it exists to fix passed every test it had, because labelling a row with the
caller's name is not the same as scoping the query to them.

`before/src/assistant/crew.py` — `delegate`, `quality`, `delegated`. Model calls are
simulated from a price table, so the whole layer stays in the fast tier.

## Deliverables

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really this, and a place to stop that is not quitting. **Full** is the version you
would show someone. **Stretch** is for when the full pass came easily.

### Minimum
- [ ] Every memory carries a **source and a TTL**, and `forget` makes a fact
      unrecallable — proven by a test, not by inspection
- [ ] The assistant recalls a fact from a **previous session** and cites where it
      learned it

### Full
- [ ] One caller's memory **never** appears in another caller's recall — with the
      partition in the database, so a restart cannot merge them
- [ ] A recalled memory colours an answer without ever becoming a **citation**:
      personalisation is not grounding
- [ ] A **corrected** fact replaces the old one: the stale row is deleted, not outranked
- [ ] `classify` returns `None` for a turn that is not worth remembering (most of them)
- [ ] Context assembly respects a **hard token budget**, keeps the guardrails pinned
      under pressure, and reports tokens used vs. parked
- [ ] A supervisor delegates to **tiered workers**, and a test asserts the delegation
      actually happened
- [ ] `compare()` reports **cost and quality together**, single-tier vs. tiered
- [ ] One failing worker leaves the run `partial` with reasons, not dead

## Stretch goals

- Add recall rows to the golden set — "given this history, does the right fact come
  back?" — and gate on them like any other slice.
- Add decay: episodic rows lose weight with age unless re-confirmed. Write down the
  half-life you chose and why.
- Swap `AssistantMemory` for a Mem0 or LangMem adapter behind the same method
  signatures (`phase5-memory/04-memory-frameworks`), run the identical tests, and record
  the verdict in your repo.

Implement `memory.py`, `tenancy.py` and `crew.py`. Reference and tests:
`after/src/assistant/memory.py`, `after/src/assistant/tenancy.py`,
`after/src/assistant/crew.py`, `after/tests/test_memory.py`,
`after/tests/test_tenancy.py`, `after/tests/test_crew.py`. Design notes:
[`adr/0008-memory-is-partitioned-by-subject.md`](adr/0008-memory-is-partitioned-by-subject.md).

**Prove the savings on your own suite.** A tiered crew that is cheaper and two points
worse on your Phase-3 gate is not a win, it is an undeclared quality cut. Report cost
and score together, or don't report the cost at all.
