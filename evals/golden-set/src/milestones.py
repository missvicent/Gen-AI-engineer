"""Score a golden set at 10, 25 and 50 rows, and say what each score may claim.

Fifty golden rows are written in three sittings, **10, 25 and 50**, and each one
is scored. A score is never shown alone: it comes with a 95% interval and a
verdict on what a sample of that size can support.

The number this module argues with: **8 of 10 is not 0.80.** It is 0.80 with a
95% interval of about 0.49–0.94, so a mediocre system and a good one both
produce 8/10 often enough that you cannot tell them apart. Ten rows still find
obvious bugs; they are not a measurement yet.

- `wilson(passed, n)` gives the interval. It uses the Wilson score interval
  rather than the textbook `p ± z·sqrt(p(1-p)/n)`, which returns a zero-width
  interval at 10/10 and bounds outside [0, 1] near the edges. Wilson stays in
  [0, 1] and pulls small samples toward 0.5:

      centre = (p + z²/2n) / (1 + z²/n)
      spread = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)

  It raises `MilestoneError` when `n <= 0` or `passed` is outside 0..n, because
  a number there would be quoted.
- `Interval.indistinguishable_from` says whether two scores can be told apart.
  Overlapping intervals cannot, whatever their point estimates say, so
  0.80 → 0.85 is not shipped as an improvement.
- `verdict(n)` says what a sample size licenses: below 10, not a measurement;
  10–24, a smoke test that must not be quoted; 25–49, per-slice shape but too
  noisy to gate a merge; 50+, gateable. It reads only `n`, so a good score
  cannot buy conclusions the sample size does not support.
- `reached`, `progress` and `render` report where the sittings stand. `render`
  keeps score, interval and verdict on one line so they are never separated.
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
        """True when the two 95% intervals overlap, so neither score is better."""
        low, high = self.bounds
        other_low, other_high = other.bounds
        return low <= other_high and other_low <= high


def wilson(passed: int, n: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval for `passed` out of `n`, clamped to [0, 1]."""
    if n <= 0:
        raise MilestoneError("no rows scored — there is nothing to put an interval on")
    if not 0 <= passed <= n:
        raise MilestoneError(f"passed ({passed}) out of {n} is not a proportion")
    p = passed / n
    centre = (p + z**2 / 2 / n) / (1 + z**2 / n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / 4 / n**2) / (1 + z**2 / n)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def verdict(n: int) -> str:
    """What a score from `n` rows may be used for. Reads `n` only, never the score."""
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
