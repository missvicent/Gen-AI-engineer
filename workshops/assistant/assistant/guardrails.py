"""Workshop 6 layer — harden the assistant. Containment, not wishful filtering.

L1 decode+squash+scan+redact on every untrusted string (incl. fetched emails and
news), spotlight untrusted content, and an output gate. Combined with the agent's
HITL + least-privilege tools, a landed injection still can't fire a gated tool.

The scan has two surfaces, because attackers work in two ways.

**Expansion** answers "what else does this text say?" — base64, percent-encoding
and HTML entities all let a payload travel as something that does not look like
a payload. Append each decoding rather than substituting it, so decoding can only
ever ADD evidence; a document that legitimately contains a base64 blob must not
be rewritten out from under the person who later reads it.

**Squashing** answers "what does this text say if you stop respecting the
separators?" `1gn0re`, `i g n o r e`, `Ign<zero-width-space>ore` and `IGNORE` are
one word to a human and four different strings to a regex.

Neither half makes the filter complete, and the bar is not "no injection ever
lands" — that bar cannot be met. The bar is that a landed injection cannot fire
an irreversible tool or leak PII, which is HITL's job, not this file's. What this
file buys is that the cheap obfuscations stop working.

Reference: ../../after/src/assistant/guardrails.py.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

#: (ok, cleaned_or_reason) — the shape every screen in this codebase speaks, so
#: a hardened one can be substituted wherever a plain one is expected.
Screen = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class Bounded:
    """A pattern together with the longest run of text it can match.

    The streaming gate in `output_gate.py` releases a prefix only once no
    still-forming match could reach back into it. That decision needs a number,
    and a number that is wrong in the unsafe direction leaks. Keeping the span
    beside the pattern means the two are edited together; `test_stream.py`
    checks each declaration against the compiled regex, so a pattern that
    outgrows its bound fails the suite instead of quietly shrinking the window.
    """

    pattern: re.Pattern[str]
    max_span: int

    def search(self, text: str) -> re.Match[str] | None:
        return self.pattern.search(text)

    def sub(self, repl: str, text: str) -> str:
        return self.pattern.sub(repl, text)


#: Every span here is bounded on purpose: `output_gate.HOLDBACK_CHARS` is derived
#: from these numbers, so an unbounded quantifier in this list would turn the
#: window into a guess and the gate's guarantee into a hope. See ADR-0006.
PII = [
    Bounded(re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 11),
    # RFC 5321 caps the local part at 64 characters and each domain label at 63;
    # four labels is more than a real address needs. Bounding the repetition
    # costs no detection: an over-long local part still matches on its last 64
    # characters, which is enough for the gate to refuse the answer.
    Bounded(re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,4}"), 384),
]

#: Matched against the EXPANDED text, where ordinary spacing still exists.
INJECTION = [
    re.compile(
        r"ignore\s+(all\s+|the\s+)?(previous|prior|above|earlier).{0,20}instructions?",
        re.I,
    ),
    re.compile(r"you are now|new system prompt|disregard", re.I),
    re.compile(r"forward .* to .*@", re.I),
    re.compile(r"reveal.{0,20}(system )?prompt", re.I),
    # multilingual override + exfiltration verbs — see phase6 red-team v3
    re.compile(
        r"(ignora|ignorez|ignoriere|ignore)\b.{0,40}"
        r"(instrucciones|instructions|anweisungen|instru\u00e7\u00f5es)",
        re.I,
    ),
    re.compile(r"(send|forward|email|upload|export)\b.{0,60}\bto\b.{0,40}@", re.I),
    re.compile(r"(upload|post|send)\b.{0,60}https?://", re.I),
]

#: Matched against the SQUASHED text — the same tells, written as if every space,
#: punctuation mark and invisible character had been removed, because there they
#: have been.
SQUASHED_INJECTION = [
    re.compile(r"ignor[ae](all|the)?(previous|prior|above|earlier)\w{0,12}instruc"),
    re.compile(r"youarenow|newsystemprompt|disregard"),
    re.compile(r"reveal\w{0,12}prompt"),
    re.compile(r"(ignora|ignorez|ignoriere)\w{0,24}(instruc|anweisung)"),
]

#: Digits and symbols that stand in for letters. Folded on the squashed surface
#: only — the reader must still see exactly the text that was sent.
LEET = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "|": "l",
})


def decode_and_normalize(text: str) -> str:
    """TODO 1: the text plus every plaintext hiding inside it.

    Base64 first: find long `[A-Za-z0-9+/]` runs, decode with `validate=True`,
    keep the ones that come back printable. Then the two encodings any payload
    meets on a web path — `urllib.parse.unquote_plus` when the text contains a
    `%`, `html.unescape` when it contains an `&`.

    Append, never substitute. A wrong decoding that ADDS text costs a false
    positive a human will see; one that REPLACES text loses the evidence."""
    raise NotImplementedError


def squash(text: str) -> str:
    """TODO 2: the text with every separator an attacker can hide behind removed.

    Four steps, in order: NFKC-normalise (fullwidth `\uff49\uff47\uff4e\uff4f\uff52\uff45`
    folds to `ignore`); drop every character whose `unicodedata.category` is
    `Cf` — the class of invisible formatting characters, zero-width space and
    soft hyphen among them, whose only role in a payload is to split the word a
    filter is looking for; lowercase and apply LEET; then delete everything that
    is not `[a-z0-9]`.

    `squash("Ign\\u200bore") == squash("1gn0re") == squash("i g n o r e")`."""
    raise NotImplementedError


def looks_like_injection(text: str) -> bool:
    """TODO 3: True if EITHER surface carries a known pattern.

    INJECTION against the expanded text, SQUASHED_INJECTION against the squash
    of that same expanded text — squashing the expansion, not the original, so a
    base64 payload gets both treatments."""
    raise NotImplementedError


def spotlight(untrusted: str) -> str:
    """TODO 4: mark untrusted content as DATA, never instructions."""
    raise NotImplementedError


def screen(text: str) -> tuple[bool, str]:
    """TODO 5: L1 — (ok, cleaned_or_reason).

    Cap the EXPANDED length (a small input that decodes to a megabyte is still a
    megabyte), refuse anything `looks_like_injection` flags with the reason
    `"injection"`, then redact PII from the ORIGINAL text and return it."""
    raise NotImplementedError


def output_ok(answer: str) -> bool:
    """TODO 6: L3 output gate — never leak PII in a response."""
    raise NotImplementedError
