"""The outbound trust boundary — nothing leaves before the gate has seen it.

`screening.py` guards what comes *in*. This guards what goes *out*, and the
streaming case is where that gets interesting.

The naive streaming shape yields each chunk as it arrives and screens the joined
answer at the end. That gate cannot do its job: by the time it fires, the PII is
already rendered in the caller's browser and sitting in their proxy logs. A
`done` event announcing "actually that was redacted" is a notification, not a
control.

Buffering the whole answer fixes it and throws away streaming. The middle is to
release only the prefix of the answer that no still-forming match could reach
back into. Getting that condition right took three attempts, and the two failures
are worth more than the fix.

**Attempt one** kept back a round 256 characters and a comment calling it "far
more than enough". The email pattern had an unbounded `+`, so it could match a
span of any length; a long local part pushed the start of the address out of the
window and onto the wire.

**Attempt two** bounded every pattern and *derived* the window from those bounds
— `HOLDBACK_CHARS` is the longest span any `guardrails.PII` entry can match. That
fixed the stated premise and left the real one unstated. Bounding the email regex
to 64 characters of local part means an over-long address still matches, but only
on its **last** 64 characters; the hundreds of characters in front of that are
part of no match at all, so nothing held them. A 1000-character local part
released 608 characters of itself and then blocked the terminal event, which is
the same leak wearing the previous fix as a costume.

The mistake both times was reasoning about **matches** when the thing that must
not escape is a **candidate**. Before the `@` arrives, the gate cannot know
whether the run of characters it is holding is a word or the front half of an
address. So the release rule now rests on two bounds and takes the tighter:

1. **the span bound** — a character older than `HOLDBACK_CHARS` cannot be
   extended into a new match, because no pattern can match that far; and
2. **the token bound** — a character before the start of the trailing
   *unterminated* candidate run cannot be part of a match that is still forming,
   because no pattern can match across a `DELIMITER`.

Either one alone would be sufficient today. Keeping both means a future edit has
to break two independent invariants to leak, and both are proved mechanically
rather than asserted in prose: `test_stream.py` derives each pattern's true
maximum span from its compiled form, and walks its parse tree to prove it can
consume no delimiter. A pattern that cannot state a bound, or that could match
across a space, fails the suite instead of quietly widening the hole.

The cost is honest and bounded. Ordinary prose is full of delimiters, so the
token bound is nearly free — the gate holds the word being typed. Adversarial
input is where it bites: a single unbroken 4000-character token is held in full
until something terminates it, which is to say the gate degrades to full
buffering exactly when full buffering is what safety requires. `ASSISTANT_STREAM_MODE=raw`
remains available for local work where none of this is worth paying for —
documented as local-only, because with it the gate becomes a notification again.

The other terminal event here is `truncated`. A model that dies halfway through
leaves a grammatical fragment that reads like a whole answer; presenting it as
one is its own kind of dishonesty, so the gate distinguishes "this is the answer"
from "this is as far as the answer got".
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from assistant import deadline, guardrails
from assistant.composers import StreamTruncated

SAFE_BUFFERED = "safe-buffered"
RAW = "raw"

#: The span bound, derived from the patterns rather than guessed ahead of them:
#: the longest span any `guardrails.PII` entry can match. Release a character
#: sooner than this and a match could still form behind it.
HOLDBACK_CHARS = max(p.max_span for p in guardrails.PII)

#: One character that no `guardrails.PII` pattern can consume — the complement of
#: every character class they use. A match therefore lies entirely inside one run
#: of non-delimiter characters, which is what makes the run boundary a safe place
#: to cut: everything before the trailing run is either already a complete match
#: (and blocked) or can never become one.
#:
#: This is a claim about the patterns, so it is checked against the patterns.
#: `test_stream.py` walks each compiled regex's parse tree and proves it can
#: consume nothing this expression matches. Add a pattern with a space in it and
#: that test fails; the alternative is a comment nobody re-reads.
DELIMITER = re.compile(r"[^\w.+@-]")

REDACTION = "[redacted: output failed the safety gate]"


def gated_chunks(
    chunks: Iterator[str],
    mode: str = SAFE_BUFFERED,
    check: Callable[[], None] | None = None,
) -> Iterator[tuple[str, str]]:
    """TODO 1: screen a stream on its way out.

    Yield `("chunk", text)` for every span cleared to leave the process, then
    exactly one terminal event:

      * `("done", answer)` — the whole answer arrived and passed
        `guardrails.output_ok`;
      * `("blocked", REDACTION)` — it did not pass;
      * `("truncated", partial)` — the source raised `StreamTruncated`, meaning
        it stopped mid-answer.

    `mode == RAW` is the naive shape and is written for you in `_raw` below —
    read it first, then build the default so the difference is concrete.

    For SAFE_BUFFERED, keep three pieces of state: the accumulated `buffer`, how
    many characters you have `released`, and where the trailing run of candidate
    characters starts (`run_starts_at`, which is just past the most recent
    `DELIMITER` match). On every incoming chunk:

      * append it to the buffer and re-screen the WHOLE buffer — a pattern can
        straddle a chunk boundary, so screening chunks individually catches
        nothing;
      * on failure, yield ("blocked", REDACTION) and stop. What you already
        released is clean by construction; the client still discards it;
      * advance `run_starts_at` by scanning the text you just appended for
        delimiters (`DELIMITER.finditer(buffer, scanned_to)` — scan only the new
        part, or an adversarial 4000-character token costs you a quadratic);
      * cut at `min(len(buffer) - HOLDBACK_CHARS, run_starts_at)` and, if that is
        ahead of `released`, yield `buffer[released:cut]` and move `released` up.

    Both halves of that `min` are load-bearing, and the module docstring above
    explains why at length. The short version, because it is the part that is
    genuinely hard: the span bound alone is not enough. Reason about it with a
    500-character local part and you will find that the address still gets
    detected — the regex matches its last 64 characters — while the first hundred
    characters of it are part of no match at all, and so nothing holds them back.
    Two shipped versions of this file had that bug.

    When the source is exhausted, screen once more, flush whatever is still held
    back, and yield ("done", buffer).

    Wrap the loop in `try: ... except StreamTruncated:` and remember that it
    happened. The tail still gets screened and flushed — text that already left
    the model is text you are responsible for — but the terminal event becomes
    `("truncated", buffer)` instead of `("done", buffer)`. Ending a died-early
    stream with `done` is how a fragment gets served as a finished answer.

    Iterate with `_while_wanted(chunks, check)` rather than over `chunks`, and
    catch `deadline.Expired` alongside `StreamTruncated`. `_while_wanted` is
    written for you; what it does is ask, before pulling each chunk, whether the
    caller is still there. That has to happen HERE and not only in the loop
    consuming this generator, and the reason is the holding behaviour you are
    about to build: on an unbroken token this yields nothing at all, so a
    cancellation check that runs between released frames runs zero times. An
    abandoned stream then drains the whole model, re-screening a growing buffer
    per chunk — measured at 9.2 seconds for 800 chunks nobody would ever read.
    An expiry ends the stream as `truncated`, for the same reason a dead model
    does: a buffer that was never sent is not an answer.

    The tests to hold yourself to are already written, and two of them are worth
    reading before you start: `test_no_local_part_is_long_enough_to_outrun_the_gate`
    sweeps the address length past the window at five different chunk sizes, and
    `test_nothing_the_gate_releases_belongs_to_a_token_it_blocks` states the
    property over generated inputs. Do not hardcode a number here — the bounds come
    from `HOLDBACK_CHARS` and `DELIMITER`, both derived from `guardrails.PII`.
    """
    raise NotImplementedError


def _while_wanted(
    chunks: Iterator[str], check: Callable[[], None] | None
) -> Iterator[str]:
    """The source, asking before each pull whether anyone still wants it.

    Written as `next()` in a loop rather than `for`, because the difference is the
    entire point: `for` asks the source for a chunk and only then hands control
    back, so a check placed around the body runs after the work it was meant to
    prevent.
    """
    source = iter(chunks)
    while True:
        if check is not None:
            check()
        try:
            yield next(source)
        except StopIteration:
            return


def _raw(
    chunks: Iterator[str], check: Callable[[], None] | None = None
) -> Iterator[tuple[str, str]]:
    """Emit first, judge later. Local-only: the verdict arrives after the content
    it was supposed to withhold, which is exactly the failure the default mode
    exists to prevent. Kept because seeing the two side by side is the lesson."""
    parts: list[str] = []
    truncated = False
    try:
        for chunk in _while_wanted(chunks, check):
            parts.append(chunk)
            yield "chunk", chunk
    except (StreamTruncated, deadline.Expired):
        truncated = True
    answer = "".join(parts)
    if not guardrails.output_ok(answer):
        yield "blocked", REDACTION
        return
    yield ("truncated" if truncated else "done"), answer
