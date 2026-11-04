"""TODO: turn a results file into an exit code.

`make gate` must fail for four different reasons, because they are four different
bugs:

  1. an **absolute bar** breach — the system isn't good enough, full stop;
  2. a **regression** against the committed baseline beyond `TOLERANCE` — a bar alone
     lets you rot from 0.94 to 0.86 with every PR passing;
  3. a **per-slice** regression — overall 0.86 can hide the unanswerable slice
     collapsing from 1.00 to 0.40, or a slice disappearing entirely;
  4. **instrument drift** — a different judge model, temperature or RAGAS version
     means the numbers are not comparable. That is a re-baseline, and it belongs in
     a reviewed commit.

Everything here is pure logic over two JSON files: no model, no network, no excuse
not to run it on every push.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

BARS = {"faithfulness": 0.85, "context_recall": 0.80}

# From lesson 3.3's calibration report — the judge's disagreement rate with your own
# labels, averaged over the suite. Recompute it when you re-calibrate; don't guess.
TOLERANCE = 0.03

# A slice at zero is never noise, whatever the tolerance says.
COLLAPSE_FLOOR = 0.5


@dataclass(frozen=True)
class Run:
    overall: dict[str, float]
    by_slice: dict[str, dict[str, float]]
    instrument: dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> Run:
        """TODO 1: read a results/baseline JSON file into a Run."""
        raise NotImplementedError


def check_bars(run: Run, bars: dict[str, float] = BARS) -> list[str]:
    """TODO 2: one readable string per metric that sits below its bar."""
    raise NotImplementedError


def check_instrument(run: Run, baseline: Run) -> list[str]:
    """TODO 3: if any instrument key changed, refuse to compare.

    The message must contain "instrument changed" and "re-baseline" — a reviewer
    should be able to act on it without opening this file.
    """
    raise NotImplementedError


def check_regressions(
    run: Run, baseline: Run, tolerance: float = TOLERANCE
) -> list[str]:
    """TODO 4: overall AND per-slice regressions beyond the tolerance.

    Also catch a slice that disappeared, and mark a collapse (dropping under
    COLLAPSE_FLOOR from above it) so nobody mistakes it for drift.
    """
    raise NotImplementedError


def gate(run: Run, baseline: Run, tolerance: float = TOLERANCE) -> list[str]:
    """TODO 5: every reason to block the merge. Empty list means it may land."""
    raise NotImplementedError


def diff_table(run: Run, baseline: Run, tolerance: float = TOLERANCE) -> str:
    """TODO 6: base / now / delta / status per metric and per slice.

    A failing gate is a code-review artifact. Statuses: "REGRESSED",
    "ok (within tolerance)" for a dip inside the noise floor, "ok" otherwise.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """TODO 7: print the table, then the reasons. 0 = pass, 1 = fail, 2 = bad usage."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
