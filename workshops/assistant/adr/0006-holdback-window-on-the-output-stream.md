# ADR-0006 — A holdback window on the output stream

**Status:** accepted (amended twice — the second amendment is the one that
actually closed the leak; read to the end before copying anything here)

## Context

`/ask` screens its answer before returning it. `/ask/stream` originally yielded
every chunk as the composer produced it and screened the joined text at the end.
Those are not the same guarantee. In the streaming path the gate fired *after*
delivery: the PII was already rendered in the caller's browser and written to
every proxy log on the way. The `done` event's redaction notice asked the client
to un-see it.

Two obvious fixes, both bad on their own. Buffer the whole answer and streaming
is gone — the feature exists to reduce time-to-first-token. Keep emitting and the
gate is decorative.

## Decision

Screen the accumulated answer on every chunk, and release only the prefix that
sits at least `HOLDBACK_CHARS` behind the head of the buffer.

The bound is what makes it sound. If no pattern `guardrails.output_ok` can match
spans more than `HOLDBACK_CHARS` characters, then a match that *starts* inside a
released prefix must already be complete — and therefore already caught — before
that prefix is released. Nothing unscreened crosses the boundary, and the client
still receives a long answer progressively.

`ASSISTANT_STREAM_MODE` selects `safe-buffered` (default) or `raw`. `raw` is the
original emit-then-judge behavior, kept because a test that pins the leak
side-by-side with the fix teaches more than a deleted branch, and documented as
local-only.

## Amendment: the bound is derived, not declared

The first version of this decision set `HOLDBACK_CHARS = 256` and asserted in a
comment that every pattern was "far shorter". That assertion was false. The
email pattern was:

```python
re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
```

`+` has no maximum. An address with a 325-character local part began *before* the
window, so by the time the closing `.com` made the match recognisable, 69
characters of it had already been released as safe. The soundness argument above
was correct; its premise was not, and nothing in the system checked the premise.
The one test that looked like it did built a 194-character email by hand and
called it maximal.

Two changes close it:

1. **Every output pattern declares its maximum span** (`guardrails.Bounded`).
   The email pattern is now bounded to RFC-shaped limits — 64 characters of
   local part, 63 per domain label, at most four labels, so 384 characters. An
   over-long local part is still detected, because the last 64 characters of it
   still match; only the *span* is bounded, not the detection.
2. **The window is computed from those declarations**:
   `HOLDBACK_CHARS = max(p.max_span for p in guardrails.PII)`. Adding a pattern
   that can match further widens the window automatically. A pattern that cannot
   state a bound cannot be added.

`test_stream.py` proves each declaration against the compiled regex using the
`re` parser's own width calculation, so a pattern that outgrows its declared
bound fails the suite. The proof lives in the test rather than in the gate: it
reaches into a private module, which is acceptable where a future Python breaks
it loudly and unacceptable on the request path.

The general lesson is worth more than the fix. A safety property written in a
comment is enforced by nobody. This one is now enforced by arithmetic.

## Amendment 2: a span bound is not enough — the unit is the token

The amendment above is sound and it did not close the leak. It bounded the span a
pattern can *match* and left unexamined the question of what must not be
*released*, which are not the same set of characters.

Bounding the email pattern to 64 characters of local part means an over-long
address still matches — on its **last** 64 characters. The hundreds of characters
in front of that are part of no match at all. Nothing held them, because the whole
release rule was phrased in terms of matches. Measured on the shipped gate: a
500-character local part released 108 characters of itself before blocking, and a
1000-character one released 608. The second audit found it by trying a longer
address than the regression test used, which is exactly how the first one was
found.

The mistake was the same both times, and naming it is the point of this amendment:
the gate was reasoning about **matches** when the thing that must not escape is a
**candidate**. Before the `@` arrives, a run of characters is either a word or the
front half of an address and the gate cannot tell which.

So the release rule now takes the tighter of two bounds:

1. **the span bound** — a character older than `HOLDBACK_CHARS` cannot be extended
   into a new match, because no pattern can match that far; and
2. **the token bound** — a character before the start of the trailing
   *unterminated* candidate run cannot belong to a match that is still forming,
   because no pattern can match across a `DELIMITER` (`[^\w.+@-]`, the complement
   of every class the patterns use).

Either alone would suffice today. Both are kept so that a future edit has to break
two independent invariants to leak, and both are proved mechanically rather than
asserted: `test_stream.py` derives each pattern's true maximum span from its
compiled form, and walks its parse tree to prove it can consume no delimiter. A
pattern with a space in it fails the suite instead of quietly widening the hole.

**The test was wrong in the same way the code was.** The first version of the new
property test asserted that released text never overlaps a span `guardrails.PII`
can match — and it passed against the OLD, leaking rule, on 300 generated cases
out of 300. Of course it did: being outside every match span is precisely how the
leaked bytes got out. The property has to be stated over the candidate RUN
containing the match, and once it was, the old rule failed 55 of those cases with
a worst leak of 1125 characters. A property test asserting the wrong property is
more dangerous than no test, because it is evidence.

## Alternatives considered

Full buffering (correct, and throws away the feature — though note that the token
bound *degrades* to full buffering for a single unbroken token, which is the right
behaviour precisely when that is what safety requires). A stateful incremental DLP
scanner that tracks per-pattern partial-match state across chunks (strictly more
general; the token bound is the cheap special case that works because every pattern
here is delimiter-free, and the general machine would be the answer if one were
not). A rolling redactor that rewrites PII in flight
instead of blocking (diverges from `/ask`, which blocks — two gates with two
verdicts is worse than one gate). Screening each chunk in isolation (catches
nothing: a pattern straddling a chunk boundary is invisible to both halves).
Trusting the model not to emit PII (not a control).

## Consequences

An answer shorter than `HOLDBACK_CHARS` arrives in a single frame, and every
answer reaches the client that many characters behind generation. The window
widened from 256 to 384 when the bound became real, which is the honest price of
the guarantee the original number was only claiming. Both are accepted and
measurable, and `tier.stream` on `/health` reports which mode is live so an
operator can tell from outside the process.

The token bound adds no cost to ordinary prose, which is full of delimiters — it
holds the word currently being written. It costs everything on adversarial input: a
single unbroken 4000-character token is buffered in full until something terminates
it. That is the intended shape. The characters a streaming safety gate cannot
release are exactly the characters it cannot yet classify.

Three attempts on one property is the real consequence worth recording. Each fix
was verified against the case that had just failed, and each was beaten by the next
case nobody had thought of. What ended it was not a better number but a different
question — asking what the gate is *holding* rather than what it can *match* — and
a property test over generated input that had to itself be corrected before it could
see the bug.
