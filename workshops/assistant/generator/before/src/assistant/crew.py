"""Crew layer — the assistant delegates research instead of doing everything itself.

The receipt is the deliverable: cost per run, per agent, next to a quality number. A
tiered crew that is cheaper and worse is not a saving, it is an undeclared quality cut,
so `compare()` prints both and never one alone.

Model calls are simulated from a price table so this layer stays in the fast tier. The
routing, delegation, attribution and failure handling are what you would ship — swap
`_invoke` for the Phase-1 client later and nothing else here changes.

Reference: ../../../after/src/assistant/crew.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

#: USD per million tokens, (input, output). Illustrative tiers, not vendor quotes.
PRICE: dict[str, tuple[float, float]] = {
    "local": (0.0, 0.0),
    "cheap": (1.0, 5.0),
    "frontier": (5.0, 25.0),
}

TIERS: tuple[str, ...] = ("local", "cheap", "frontier")

#: Who owns what. The supervisor owns neither, which is the whole point.
SKILLS: dict[str, str] = {"research": "researcher", "draft": "writer"}

SUPERVISOR = "supervisor"


@dataclass(frozen=True)
class Job:
    """One delegated unit of work, with the cheapest tier that still does it justice."""

    id: str
    kind: str  # "research" | "draft"
    in_tokens: int
    out_tokens: int
    min_tier: str = "local"


@dataclass(frozen=True)
class Call:
    agent: str
    tier: str
    kind: str
    in_tokens: int
    out_tokens: int
    job_id: str | None = None

    @property
    def cost(self) -> float:
        price_in, price_out = PRICE[self.tier]
        return (self.in_tokens * price_in + self.out_tokens * price_out) / 1_000_000


@dataclass
class CrewRun:
    calls: list[Call] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return sum(call.cost for call in self.calls)

    @property
    def status(self) -> str:
        return "partial" if self.errors else "ok"

    def by_agent(self) -> dict[str, float]:
        """TODO: total cost per agent — this is what makes the receipt readable."""
        raise NotImplementedError("by_agent() is yours to implement")


Router = Callable[[Job], str]
Worker = Callable[[Job, str], Call]


def tiered(job: Job) -> str:
    """Cheapest tier that still gets it right. Triage is cheap work; synthesis is not."""
    return job.min_tier


def all_frontier(_job: Job) -> str:
    """The baseline every cost claim has to be measured against."""
    return "frontier"


def _invoke(job: Job, tier: str, agent: str) -> Call:
    return Call(agent, tier, job.kind, job.in_tokens, job.out_tokens, job.id)


def worker(job: Job, tier: str) -> Call:
    """TODO: one `_invoke`, attributed to the agent that OWNS this kind (`SKILLS`)."""
    raise NotImplementedError("worker() is yours to implement")


def delegate(
    jobs: Sequence[Job],
    workers: dict[str, Worker] | None = None,
    route: Router = tiered,
    plan_tier: str = "local",
) -> CrewRun:
    """Supervisor plans once, then hands every job to the worker that owns it.

    TODO: exactly ONE supervisor call, on `plan_tier`, whatever the job count.
    TODO: a kind nobody owns is an entry in `crew.errors`, not an exception.
    TODO: catch whatever a worker raises into `crew.errors` — one flaky worker must not
          take the rest of the run down with it.
    """
    raise NotImplementedError("delegate() is yours to implement")


def quality(crew: CrewRun, jobs: Sequence[Job]) -> float:
    """Share of jobs run on a tier good enough for them — stand-in for your eval score.

    TODO: compare each job's tier against its `min_tier` using `TIERS.index`.
          No jobs means 1.0, not a ZeroDivisionError.
    """
    raise NotImplementedError("quality() is yours to implement")


def delegated(crew: CrewRun) -> bool:
    """Did the supervisor hand the work over, or quietly keep it?

    TODO: False as soon as the supervisor makes a call for a kind in `SKILLS`. A
          supervisor doing the work itself passes every output check while wasting the
          entire design — this is the trajectory check that catches it.
    """
    raise NotImplementedError("delegated() is yours to implement")


def compare(jobs: Sequence[Job], routes: dict[str, Router]) -> str:
    """Cost and quality side by side. Reporting one without the other is how you lie."""
    baseline = delegate(jobs, route=all_frontier)
    rows = [f"{'route':<12} {'cost':>10} {'vs base':>9} {'quality':>8}  status"]
    for name, router in routes.items():
        crew = delegate(jobs, route=router)
        delta = 0.0 if baseline.cost == 0 else 1 - crew.cost / baseline.cost
        rows.append(
            f"{name:<12} ${crew.cost:>9.4f} {delta:>8.0%} "
            f"{quality(crew, jobs):>8.0%}  {crew.status}"
        )
    return "\n".join(rows)
