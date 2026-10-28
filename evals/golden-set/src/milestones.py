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

"""

from __future__ import annotations

import math
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
        low, high = self.bounds
        other_low, other_high = other.bounds
        return low <= other_high and other_low <= high


def wilson(passed: int, n: int, z: float = Z) -> tuple[float, float]:
    if n <= 0:
        raise MilestoneError("no rows scored — there is nothing to put an interval on")
    if not 0 <= passed <= n:
        raise MilestoneError(f"passed ({passed}) out of {n} is not a proportion")
    p = passed / n
    centre = (p + z**2 / 2 / n) / (1 + z**2 / n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / 4 / n**2) / (1 + z**2 / n)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def verdict(n: int) -> str:
    if n < MILESTONES[0]:
        return "not yet a measurement - finish the first ten"
    if n < MILESTONES[1]:
        return (
            "smoke test. Enough to find obvious breakage and to prove the harness "
            "runs end to end. Not enough to compare two systems, and not enough to "
            "quote"
        )
    if n < MILESTONES[2]:
        return (
            "shape. Per-slice differences start to mean something, so this is where "
            "you learn which slice is dragging the average. Still too noisy to gate a "
            "merge on"
        )
    return (
        "gateable. Wide enough to hold a per-slice signal and narrow enough that a "
        "real regression clears the interval. This is the floor, not the target"
    )


def reached(n: int) -> int | None:
    """The highest milestone reached so far.

    If `n` is 9, return `None`; if `n` is 24, return 10; if `n` is 49, return 25;
    from 50 up, return 50.
    """
    hit = [m for m in MILESTONES if n >= m]
    return hit[-1] if hit else None


def progress(rows: int) -> str:
    """One line for the top of the report: where you are and where you're going."""
    done = reached(rows)
    if done is None:
        return (
            f"{rows}/{MILESTONES[0]} rows - "
            f"{MILESTONES[0] - rows} to the first milestone"
        )
    if done == MILESTONES[-1]:
        return f"{rows} rows - all milestones reached"
    nxt = next(m for m in MILESTONES if m > rows)
    return f"{rows} rows - milestone {done} reached, {nxt - rows} to {nxt}"


def render(interval: Interval, label: str = "overall") -> str:
    """One line for the bottom of the report: what you scored."""
    low, high = interval.bounds
    return (
        f"{label}: {interval.score:.2f} "
        f"(95% CI {low:.2f}–{high:.2f}, n={interval.n}) "
        f"— {verdict(interval.n)}"
    )
