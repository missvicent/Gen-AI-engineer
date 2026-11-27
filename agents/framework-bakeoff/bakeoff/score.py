"""Six rows, a winner only where the measured gap is real.

Spec: tests/test_score.py (`make test`).
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

ROWS = ("durability", "recovery", "complexity", "observability", "latency", "cost")
UNDECIDED = "undecided"


def only_true(probes: dict, row: str) -> str:
    """A yes/no row: the winner is the ONE framework that said yes."""
    yes = [fw for fw, p in probes.items() if p[row]]
    return yes[0] if len(yes) == 1 else UNDECIDED


def fewest_lines(probes: dict) -> str:
    """complexity: the fewest glue lines win, if the next one has 20% more."""
    (best, low), (_, next_low) = sorted(
        ((fw, p["glue_lines"]) for fw, p in probes.items()), key=lambda x: x[1]
    )[:2]
    return best if (next_low - low) / low > 0.2 else UNDECIDED


def by_framework(runs: list[dict], field: str) -> dict[str, list[float]]:
    """{"langgraph": [1.0, 1.1, ...], ...}: one field of every run, per framework."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        out[r["framework"]].append(r[field])
    return out


def fastest(runs: list[dict]) -> str:
    """latency: the lowest median wins, if the gap beats its own spread."""
    times = by_framework(runs, "latency_s")
    (best, m1), (_, m2) = sorted(
        ((fw, median(t)) for fw, t in times.items()), key=lambda x: x[1]
    )[:2]
    spread = max(times[best]) - min(times[best])
    return best if m2 - m1 > spread else UNDECIDED


def cheapest(runs: list[dict]) -> str:
    """cost: the lowest average cost wins, if it is strictly lower."""
    costs = by_framework(runs, "cost_usd")
    (best, c1), (_, c2) = sorted(
        ((fw, mean(c)) for fw, c in costs.items()), key=lambda x: x[1]
    )[:2]
    return best if c1 < c2 else UNDECIDED


def matrix(runs: list[dict], probes: dict) -> dict[str, str]:
    return {
        "durability": only_true(probes, "durability"),
        "recovery": only_true(probes, "recovery"),
        "complexity": fewest_lines(probes),
        "observability": only_true(probes, "observability"),
        "latency": fastest(runs),
        "cost": cheapest(runs),
    }


def undecided(m: dict[str, str]) -> list[str]:
    return [row for row in ROWS if m[row] == UNDECIDED]
