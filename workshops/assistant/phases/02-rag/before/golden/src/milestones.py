"""TODO: fifty rows in three sittings — and what each sitting may claim.

"Write fifty golden rows" is the instruction most people abandon. Not because it
is hard, but because it is shapeless: no feedback until the end, and a task with
no feedback for four hours is a task you do once and never again.

So it comes in three: **10, 25, 50**. Each is a real stopping point, each gets
scored, and each carries an honest statement of what a score at that `n` can
support. The milestones are the motivation. The interval is the content.

Before you start, the number this module exists to argue with: **8 of 10 is not
0.80.** It is 0.80 with a 95% interval of roughly 0.49–0.94, which is to say a
genuinely mediocre system and a genuinely good one both produce 8/10 often enough
that you cannot tell them apart. Quote 0.80 from ten rows and you have said
something false in a way that sounds rigorous. Ten rows is still enormously
better than zero — it finds your obvious bugs on the first afternoon. It is just
not a measurement yet.

TODO 1 — `wilson(passed, n)`. Use the **Wilson score interval**, not the textbook
  `p ± z·sqrt(p(1-p)/n)`. The reason is concrete rather than academic: at 10/10
  the normal approximation returns a zero-width interval, confidently reporting
  from ten questions that the system never fails, and near the edges it produces
  bounds below 0 or above 1. Wilson stays inside [0, 1] and pulls the estimate
  toward 0.5 in proportion to how little data you have.

      centre = (p + z²/2n) / (1 + z²/n)
      spread = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)

  Raise `MilestoneError` on `n <= 0` or `passed > n`. A number there would be
  worse than an exception, because it would be quoted.

TODO 2 — `Interval.indistinguishable_from`. Two intervals that overlap cannot be
  told apart, whatever their point estimates say. This is the check that stops
  0.80 → 0.85 from being read as an improvement and shipped.

TODO 3 — `verdict(n)`. What a score at this sample size licenses: below 10, not a
  measurement; 10–24, a smoke test that finds breakage but must not be quoted;
  25–49, enough for per-slice shape but too noisy to gate a merge; 50+, gateable.
  Read `n` and **only** `n`. A rule that could see the score would be the loophole
  that lets a good number buy conclusions the sample size does not support.

TODO 4 — `reached`, `progress`, `render`. `render` puts score, interval and
  verdict on one line, and they stay together: an interval with no verdict leaves
  the reader to decide what it licenses, and readers are generous about their own
  work.

Reference: ../../after/src/milestones.py.
"""

from __future__ import annotations

from dataclasses import dataclass

MILESTONES = (10, 25, 50)

#: 95%. Kept as a constant because an interval is meaningless without saying so.
Z = 1.959963984540054


class MilestoneError(Exception):
    """A refusal to score. Raised where a number would be worse than nothing."""


@dataclass(frozen=True)
class Interval:
    passed: int
    n: int

    @property
    def score(self) -> float:
        return self.passed / self.n

    @property
    def bounds(self) -> tuple[float, float]:
        return wilson(self.passed, self.n)

    @property
    def width(self) -> float:
        low, high = self.bounds
        return high - low

    def indistinguishable_from(self, other: Interval) -> bool:
        raise NotImplementedError("TODO 2: do the two intervals overlap?")


def wilson(passed: int, n: int, z: float = Z) -> tuple[float, float]:
    raise NotImplementedError("TODO 1")


def verdict(n: int) -> str:
    raise NotImplementedError("TODO 3")


def reached(n: int) -> int | None:
    raise NotImplementedError("TODO 4")


def progress(rows: int) -> str:
    raise NotImplementedError("TODO 4")


def render(interval: Interval, label: str = "overall") -> str:
    raise NotImplementedError("TODO 4")
