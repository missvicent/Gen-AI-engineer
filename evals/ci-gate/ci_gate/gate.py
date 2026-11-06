"""Turn an eval results file into an exit code, so a bad run blocks the merge.

`make gate` compares a new results file with the committed baseline and fails for
four different reasons, because they are four different bugs:

  1. an **absolute bar** breach: the system isn't good enough, whatever the
     baseline says (`check_bars`);
  2. a **regression** against the baseline beyond `TOLERANCE`: a bar alone would
     let scores rot from 0.94 to 0.86 with every PR passing (`check_regressions`);
  3. a **per-slice** regression: an overall 0.86 can hide the unanswerable slice
     collapsing from 1.00 to 0.40, or a slice disappearing entirely
     (`check_regressions`);
  4. **instrument drift**: a different judge model, temperature or RAGAS version
     means the numbers can't be compared. The gate then skips the comparison and
     asks for a re-baseline in a reviewed commit (`check_instrument`).

A report with no scores fails too, so an empty file can't pass by accident.
`gate` collects every reason, and `diff_table` shows each metric next to its
baseline for the reviewer.

Everything here is pure logic over two JSON files: no model and no network, so
it can run on every push.
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
        if metric in run.overall and run.overall[metric] < bar
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


def _delta(base: float, now: float) -> float:
    """Rounded to 3 decimals, so a drop of exactly the tolerance isn't float noise."""
    return round(now - base, 3)


def check_regressions(
    run: Run, baseline: Run, tolerance: float = TOLERANCE
) -> list[str]:
    problems = []
    for metric in sorted(baseline.overall):
        if metric not in run.overall:
            problems.append(f"metric {metric} is missing in the run")
            continue
        if _delta(baseline.overall[metric], run.overall[metric]) < -tolerance:
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
                _delta(baseline.by_slice[slice][metric], run.by_slice[slice][metric])
                < -tolerance
            ):
                problems.append(
                    f"slice '{slice}' {metric} regressed: "
                    f"{run.by_slice.get(slice).get(metric):.3f} is below the baseline "
                    f"of {baseline.by_slice.get(slice).get(metric):.3f} "
                    f"- {tolerance:.2f}"
                )
    return problems


def gate(run: Run, baseline: Run, tolerance: float = TOLERANCE) -> list[str]:
    if not run.overall or not baseline.overall:
        return ["zero rows: run or baseline has no overall metrics"]
    problems = []
    problems.extend(check_bars(run, BARS))
    changed = check_instrument(run, baseline)
    problems.extend(changed)
    if not changed:
        problems.extend(check_regressions(run, baseline, tolerance))
    return problems


def diff_table(run: Run, baseline: Run, tolerance: float = TOLERANCE) -> str:
    lines = [f"{'Metric':<34} {'Baseline':<10} {'Run':<10} {'Delta':<10} Status"]
    for metric in sorted(baseline.overall.keys() | run.overall.keys()):
        baseline_value = baseline.overall.get(metric)
        run_value = run.overall.get(metric)
        lines.append(_row(f"OVERALL {metric}", baseline_value, run_value, tolerance))

    for name in sorted(baseline.by_slice.keys() | run.by_slice.keys()):
        base_scores = baseline.by_slice.get(name, {})
        run_scores = run.by_slice.get(name, {})
        for metric in sorted(base_scores.keys() | run_scores.keys()):
            lines.append(
                _row(
                    f"SLICE {name} {metric}",
                    base_scores.get(metric),
                    run_scores.get(metric),
                    tolerance,
                )
            )
    return "\n".join(lines)


def _row(
    metric: str,
    baseline_value: float | None,
    run_value: float | None,
    tolerance: float,
) -> str:
    status = _status(baseline_value, run_value, tolerance)
    both = baseline_value is not None and run_value is not None
    base_txt = f"{baseline_value:<10.3f}" if baseline_value is not None else "-"
    now_txt = f"{run_value:<10.3f}" if run_value is not None else "-"
    delta_txt = f"{run_value - baseline_value:<+10.3f}" if both else "-"
    return f"{metric:<34} {base_txt:<10} {now_txt:<10} {delta_txt:<10} {status}"


def _status(base: float | None, now: float | None, tolerance: float) -> str:
    if now is None:
        return "MISSING"
    if base is None:
        return "NEW"
    delta = _delta(base, now)
    if delta < -tolerance:
        return "REGRESSED"
    if delta < 0:
        return "ok (within tolerance)"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2:
        print("Usage: python gate.py <run_file> <baseline_file>")
        return 2
    runs = []
    for path in argv:
        try:
            runs.append(Run.load(path))
        except (OSError, ValueError, TypeError) as error:
            print(f"could not read {path}: {error}")
            return 2
    run, baseline = runs
    problems = gate(run, baseline)
    print(diff_table(run, baseline))
    if problems:
        print("gate FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
