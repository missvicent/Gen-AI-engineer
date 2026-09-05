"""The bench itself: run one task across candidates and rank them honestly.

The data shapes are given. The five functions that turn them into a defensible
ranking are yours.

Two design decisions are already baked into the types, and it is worth seeing why
before you start filling in bodies:

1. `Runner` is injected. Nothing in this module imports a vendor SDK, which is
   what lets the whole bench run offline against a fake.
2. `Row.cost_per_success` exists, and `Row.cost_usd` is not enough on its own.
   Ranking on raw spend rewards a model that fails cheaply. Think about what the
   right value is when `ok == 0`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    """One thing under test: a model, where it lives, and what it charges."""

    name: str  # the config key used to start it: "local", "gpt", "claude"
    model: str
    price_in: float  # $ per million input tokens ($0 for local)
    price_out: float  # $ per million output tokens


@dataclass(frozen=True)
class Usage:
    """A vendor-agnostic view of what one call cost, in tokens."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class Reply:
    """What a runner hands back. Text plus the billing truth."""

    text: str
    usage: Usage


class Runner(Protocol):
    """Sends one prompt to one candidate. Raises on any failure."""

    def __call__(self, candidate: Candidate, prompt: str) -> Reply: ...


Validator = Callable[[str], bool]
Clock = Callable[[], float]


def cost(candidate: Candidate, usage: Usage) -> float:
    """Dollars for one call, computed from `usage` — never from an estimate."""
    p_input, p_output = candidate.price_in, candidate.price_out
    cached = usage.cache_read_input_tokens
    fresh = usage.input_tokens - cached
    return (
        p_input * fresh
        + p_output * usage.output_tokens
        + p_input * cached * 0.1  # cached input ~10% price
    ) / 1_000_000


@dataclass(frozen=True)
class Result:
    """One candidate, one case."""

    candidate: str
    ok: bool  # the reply arrived AND satisfied the validator
    latency_ms: float
    cost_usd: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile — no interpolation, and 0.0 for an empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    k = max(0, min(n - 1, math.ceil(p / 100 * n) - 1))
    return ordered[k]


@dataclass
class Row:
    """One candidate's aggregate across every case."""

    candidate: str
    model: str
    cases: int
    ok: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    p50_ms: float
    max_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.ok / self.cases if self.cases else 0.0

    @property
    def cost_per_success(self) -> float:
        """The number that decides the ranking.

        if there are no successes, return infinity since it's infinetely expensive per answer
        """
        return self.cost_usd / self.ok if self.ok else float("inf")


@dataclass
class BenchRun:
    task: str
    rows: list[Row]

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.rows)


def run_case(
    candidate: Candidate,
    prompt: str,
    runner: Runner,
    validate: Validator,
    clock: Clock = time.perf_counter,
) -> Result:
    """One call, timed and billed. Never raises - a dead provider is a data point"""

    started = clock()
    try:
        reply = runner(candidate, prompt)
    except Exception as e:
        return Result(
            candidate.name,
            ok=False,
            latency_ms=(clock() - started) * 1000,
            cost_usd=0.0,
            error=f"{type(e).__name__}: {str(e)}",
        )
    elapsed = (clock() - started) * 1000
    ok = validate(reply.text)
    return Result(
        candidate=candidate.name,
        ok=ok,
        latency_ms=elapsed,
        cost_usd=cost(candidate, reply.usage),
        tokens_in=reply.usage.input_tokens,
        tokens_out=reply.usage.output_tokens,
        error=None if ok else "schema violation",
    )


def run_bench(
    candidates: Iterable[Candidate],
    cases: Sequence[str],
    runner: Runner,
    validate: Validator,
    task: str = "task",
    clock: Clock = time.perf_counter,
) -> BenchRun:
    """Every candidate against every case, aggregated into one row each."""
    rows = list[Row]()
    for candidate in candidates:
        results = [run_case(candidate, case, runner, validate, clock) for case in cases]
        latencies = [r.latency_ms for r in results]
        rows.append(
            Row(
                candidate=candidate.name,
                model=candidate.model,
                cases=len(cases),
                ok=sum(1 for r in results if r.ok),
                tokens_in=sum(r.tokens_in for r in results),
                tokens_out=sum(r.tokens_out for r in results),
                cost_usd=sum(r.cost_usd for r in results),
                p50_ms=percentile(latencies, 50),
                max_ms=max(latencies) if latencies else 0.0,
                errors=[r.error for r in results if r.error],
            )
        )

    return BenchRun(task=task, rows=rows)


def rank(run: BenchRun) -> list[Row]:
    """Cheapest *working* answer first, ties broken by latency."""
    return sorted(run.rows, key=lambda r: (r.cost_per_success, r.p50_ms))
