"""Crew layer — the assistant delegates research instead of doing everything itself.

The receipt is the deliverable: cost per run, per agent, next to a quality number. A
tiered crew that is cheaper and worse is not a saving, it is an undeclared quality cut,
so `compare()` prints both and never one alone.

Model calls are simulated from a price table so the layer stays in the fast tier. The
routing, delegation, attribution and failure handling are what you would ship — swap
`_invoke` for the Phase-1 client and nothing else here changes.
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
        out: dict[str, float] = {}
        for call in self.calls:
            out[call.agent] = out.get(call.agent, 0.0) + call.cost
        return out


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
    return _invoke(job, tier, agent=SKILLS[job.kind])


def delegate(
    jobs: Sequence[Job],
    workers: dict[str, Worker] | None = None,
    route: Router = tiered,
    plan_tier: str = "local",
) -> CrewRun:
    """Supervisor plans once, then hands every job to the worker that owns it."""
    registry = workers or dict.fromkeys(SKILLS.values(), worker)
    crew = CrewRun()
    plan = Job("plan", "plan", 200 + 20 * len(jobs), 100)
    crew.calls.append(_invoke(plan, plan_tier, agent=SUPERVISOR))
    for job in jobs:
        owner = SKILLS.get(job.kind)
        if owner is None or owner not in registry:
            crew.errors[job.id] = f"no worker owns kind={job.kind!r}"
            continue
        try:
            crew.calls.append(registry[owner](job, route(job)))
        except Exception as exc:  # a flaky worker degrades the run, it does not end it
            crew.errors[job.id] = f"{type(exc).__name__}: {exc}"
    return crew


def quality(crew: CrewRun, jobs: Sequence[Job]) -> float:
    """Share of jobs run on a tier good enough for them — the stand-in for your eval score."""
    if not jobs:
        return 1.0
    handled = {call.job_id: call.tier for call in crew.calls if call.job_id}
    good = 0
    for job in jobs:
        tier = handled.get(job.id)
        if tier is not None and TIERS.index(tier) >= TIERS.index(job.min_tier):
            good += 1
    return good / len(jobs)


def delegated(crew: CrewRun) -> bool:
    """Did the supervisor hand the work over, or quietly keep it?"""
    return not any(call.agent == SUPERVISOR and call.kind in SKILLS for call in crew.calls)


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
