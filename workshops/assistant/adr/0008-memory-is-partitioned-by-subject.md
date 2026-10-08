# ADR-0008 — Memory is partitioned by subject, not labelled with one

**Status:** accepted (amended — see "Amendment: a partition nobody reads from")

## Context

Both memory backends take a `user` and scope every read and write to it:
`SqliteMemory` filters on a `user` column, `AssistantMemory` keeps its own dict.
The composition root passed neither. It built one store with the default
`user="me"` and distinguished callers by writing `source=f"user:{subject}"` into
each row.

`recall` does not read `source`. It matched on text alone, across everyone. The
retrieval path had been fixed for exactly this — `rag.search` takes a `tenant`
and Qdrant filters server-side — and memory, the store that holds the *personal*
data, had not.

Nothing about the defect looks wrong from outside. Every caller gets sensible
answers, the database honestly records who said what, and the tests pass. It
fails only when two people's words overlap, which is to say in production and
not in the suite.

## Decision

One store per subject, built on first sight by a factory the composition root
supplies:

```python
TenantMemory(lambda subject: SqliteMemory(db, user=subject))
```

Isolation then comes from scoping the backends already implement, rather than
from a filter every call site has to remember to apply. Neither backend changed
to gain it. The partition lands where the backend keeps it — a column in the
shared SQLite file, a private dict in process — and the caller does not care
which.

`recall` requires a subject and never crosses one. `all()` deliberately does
cross, because it is the operator's view: "what has this service remembered" is
a question that needs the whole answer. The asymmetry is the only sharp edge and
it is documented at both ends.

## Amendment: a partition nobody reads from

The isolation above was correct and the recall it protected went nowhere.
`offline_compose` and both model composers abstained whenever contexts and tool
state were empty; `memories` was passed in, spotlighted into the prompt, attached
to the response — and never allowed to answer. Ask the assistant a question about
something you told it and it replied "I don't know" with the fact sitting in the
`memories` field of the very same payload.

The tests agreed with the bug because they asserted on `answer["memories"]`. That
is the shape of the mistake worth naming: a test that reads the metadata proves
the plumbing, and the plumbing was never what was broken.

**Memory is a third class of evidence**, alongside retrieved documents and tool
output. When it is the only evidence and it bears on the question, it answers.
Three rules make that safe:

1. **Relevance is required.** `relevant_memories` applies a content-word filter.
   Recall is greedy — it returns what it has about the caller — so without this,
   knowing someone's timezone would answer every question the corpus could not,
   confidently and wrongly. A document question with an unrelated memory in scope
   still abstains.

   A **lexical** filter is the right instrument here and only here, because memory
   recall is itself lexical (`memory.overlap`): the filter speaks the same language
   as the store it is filtering. The same filter was briefly applied to retrieved
   documents on the argument that both are "evidence that might not answer", and
   that was wrong for the mirror-image reason — retrieval had been made semantic, so
   a word-overlap test downstream discarded exactly the hits the embedder existed to
   find. Documents are now filtered by score, at the store, where the scores are.
   See ADR-0012.
2. **It is attributed, never asserted.** The answer opens "You told me earlier",
   and the model tier uses `memory_prompt` rather than `grounded_prompt` —
   because that prompt demands `[c#]` ids, and a model asked to cite with nothing
   to cite invents one.
3. **It never earns a citation.** Enforced by a test that was already there
   (memories present, `citations == []`) and only became load-bearing once
   memory could answer at all.

`grounding: "documents" | "tools" | "memory" | "none"` rides on the response so
the class is inspectable from outside. A caller can check a document answer
against its citations; a memory answer has none by design, and saying which is
which is the difference between personalisation and a silent downgrade.

## Alternatives considered

Add a `user=` argument to every `recall` call site (one forgotten call site is
the same leak, and there is no test that catches the one you forgot). Filter on
`source` after recall (post-hoc trimming — the same anti-pattern the Qdrant
tenant filter exists to avoid, and it ranks a stranger's memory into the top-k
before discarding it). One database file per subject (real isolation, and it
turns a backup into a directory walk and a schema migration into a fleet
operation). Encrypt per subject (defends a different threat — a stolen file, not
a crossed query).

## Consequences

A store is created per subject seen, so a service with many users holds many
small connections to one file; SQLite handles this, but it is a real resource
and a reason `TenantMemory` caches. Recall cannot be tested without deciding
whose recall it is, which is the point. Anonymous callers share one partition —
correct, since they share one identity — and that is an argument for turning the
JWT gate on, not against the partition.
