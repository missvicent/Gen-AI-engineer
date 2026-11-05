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

import json
import sys
from dataclasses import dataclass, fields
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
        valid_fields = {f.name for f in fields(cls)}
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


def check_bars(run: Run, bars: dict[str, float] = BARS) -> list[str]:
    return [
        f"{metric} {run.overall[metric]:.3f} is below the bar of {bar:.2f}"
        for metric, bar in bars.items()
        if run.overall[metric] < bar
    ]


def check_instrument(run: Run, baseline: Run) -> list[str]:
    problems = []
    for key in sorted(run.instrument.keys() | baseline.instrument.keys()):
        if run.instrument.get(key) != baseline.instrument.get(key):
            problems.append(
                f"instrument changed: {key} from {baseline.instrument.get(key)} "
                f"to {run.instrument.get(key)} - re-baseline in a reviewed commit"
            )
    return problems


def check_regressions(
    run: Run, baseline: Run, tolerance: float = TOLERANCE
) -> list[str]:
    """TODO 4: overall AND per-slice regressions beyond the tolerance.

    Also catch a slice that disappeared, and mark a collapse (dropping under
    COLLAPSE_FLOOR from above it) so nobody mistakes it for drift.

    The baseline decides what must exist, so only its slices and metrics are
    checked. Anything new in the run has nothing to regress from and passes.
    """
    problems = []
    for metric in sorted(baseline.overall):
        if metric not in run.overall:
            problems.append(f"metric {metric} is missing in the run")
            continue
        if run.overall.get(metric) < baseline.overall.get(metric) - tolerance:
            problems.append(
                f"{metric} regressed: {run.overall.get(metric):.3f} is below "
                f"the baseline of {baseline.overall.get(metric):.3f} "
                f"- {tolerance:.2f}"
            )

    for slice in baseline.by_slice.keys():
        if slice not in run.by_slice:
            problems.append(f"slice '{slice}' disappeared")
            continue
        for metric in sorted(baseline.by_slice[slice]):
            if metric not in run.by_slice[slice]:
                problems.append(f"slice '{slice}' {metric} is missing in the run")
                continue
            if (
                baseline.by_slice.get(slice).get(metric) >= COLLAPSE_FLOOR
                and run.by_slice.get(slice).get(metric) < COLLAPSE_FLOOR
            ):
                problems.append(
                    f"slice '{slice}' {metric} COLLAPSED: "
                    f"{run.by_slice.get(slice).get(metric):.3f} is below the collapse "
                    f"floor of {COLLAPSE_FLOOR:.2f}"
                )
            elif (
                run.by_slice.get(slice).get(metric)
                < baseline.by_slice.get(slice).get(metric) - tolerance
            ):
                problems.append(
                    f"slice '{slice}' {metric} regressed: "
                    f"{run.by_slice.get(slice).get(metric):.3f} is below the baseline "
                    f"of {baseline.by_slice.get(slice).get(metric):.3f} "
                    f"- {tolerance:.2f}"
                )
    return problems


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
