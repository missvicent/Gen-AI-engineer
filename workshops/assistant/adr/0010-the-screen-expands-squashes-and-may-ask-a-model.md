# ADR-0010 — The screen expands, squashes, and may ask a model

**Status:** accepted

## Context

`guardrails.screen` was a base64 expansion followed by seven regexes over the
result. The red-team suite passed, which said less than it looked like: the
suite was written from the same list of patterns as the filter, so it was
measuring whether the code matched its own author's imagination.

Three gaps showed up as soon as the suite was written adversarially instead.

Base64 was the only encoding handled, and it is the least likely one on a web
path. `%69%67%6e%6f%72%65` arrives from any URL, `&#105;gnore` from any HTML
page, and both walked straight through a filter that decoded neither.

Every pattern assumed the attacker would write the words with ordinary spacing.
`1gn0re`, `i g n o r e`, `ig<U+200B>nore` and `ｉｇｎｏｒｅ` are the same
instruction to the model reading them and four different strings to `re`. This
is not an exotic attack; it is what a spam filter has dealt with since 1998.

And screening was tied to the moment of retrieval. A poisoned document was
caught on the way to the composer, which meant it had already been *stored* —
sitting in the index, coming back on every matching search, one detector
regression away from being evidence. Worse for PII: redacting at retrieval keeps
the raw SSN on disk forever.

## Decision

The scan runs over two derived surfaces, and matching either one is a refusal.

**Expansion** appends every plaintext hidden in the input: base64 runs that
decode to printable text, `unquote_plus` when there is a `%`, `html.unescape`
when there is an `&`. Appending rather than substituting means a wrong decoding
costs a false positive somebody sees, not lost evidence.

**Squashing** removes everything an attacker can hide behind: NFKC, then every
`Cf`-category character (zero-width space, soft hyphen — characters whose only
role in a payload is to split the word a filter is looking for), then leet
folding, then every non-alphanumeric. `SQUASHED_INJECTION` is a separate pattern
list because a pattern written with `\s+` in it cannot match a string that has
no spaces left.

Screening also moved to **ingest**, and stayed at retrieval. `Assistant.ingest`
refuses a poisoned document before `rag.add` ever sees it, redacts PII before it
is written down, audits each rejection, and reports the count to the caller.
Retrieval-time screening remains because documents can arrive by paths that do
not go through the API, and because a detector improved tomorrow must still
apply to what was written yesterday.

Finally, `ASSISTANT_GUARD_MODEL` adds a **model in the loop** as a second
opinion, off by default, wired in one direction only: the deterministic screen
runs first and short-circuits on a refusal, so the model can add a block and can
never clear one. It fails open to the deterministic verdict when the model is
down, slow or unparseable. Because the screen is injected into `Assistant` from
the composition root rather than imported at each call site, turning it on
covers all four untrusted channels — question, retrieved documents, tool output,
ingested documents — instead of just the one someone remembered.

## Alternatives considered

Substitute decodings instead of appending (destroys the text a human later has
to read, and a base64 blob in a legitimate document is not an attack).
Normalise the *stored* text to its squashed form (unreadable, and it discards
the evidence needed to explain a refusal). Let the guard model overturn a
deterministic block (the text under review is the adversary's input, so this
hands them an appeal court to argue in). Fail closed when the guard model is
unreachable (an Ollama restart would take the service down, and the layers that
actually contain a landed injection — HITL, least privilege, tenant scoping —
do not depend on this file). Make the guard model default-on (a round trip per
untrusted string, on a course whose fast tier must run offline). Screen only at
ingest and drop retrieval screening (assumes one entry point, forever).

## Consequences

Aggressive squashing invents adjacencies: `...design. Ignore...` squashes to
`designignore`, so a pattern like `ignore` alone would fire on innocent prose.
The squashed patterns are therefore anchored on multi-word phrases, and
`test_squashing_does_not_invent_matches` exists to keep the next pattern honest.

Expansion costs a decode pass per untrusted string and can multiply the text it
scans; the length cap is applied to the expanded form for that reason.

The guard model, when on, adds one model call per untrusted string — four per
request in the worst case — and its verdicts are not reproducible across model
versions, which is why the deterministic screen remains the thing the tests
assert on and the guard lane is asserted only for direction (it may block more,
never less). `/health` reports which of the two is in front of the caller,
because "is the guard on?" is not answerable from the outside otherwise.
