"""The crew layer: cost with quality, delegation asserted, worker failure contained."""

from assistant.crew import (
    SKILLS,
    SUPERVISOR,
    Call,
    Job,
    all_frontier,
    compare,
    delegate,
    delegated,
    quality,
    tiered,
    worker,
)

JOBS = [
    Job("j1", "research", 500, 200, min_tier="local"),
    Job("j2", "research", 600, 250, min_tier="cheap"),
    Job("j3", "draft", 900, 500, min_tier="frontier"),
]


def test_tiering_is_cheaper_at_equal_quality() -> None:
    crew = delegate(JOBS)
    baseline = delegate(JOBS, route=all_frontier)
    assert crew.cost < baseline.cost
    assert quality(crew, JOBS) == quality(baseline, JOBS) == 1.0


def test_routing_everything_local_is_a_quality_cut_not_a_saving() -> None:
    crew = delegate(JOBS, route=lambda _job: "local")
    assert crew.cost == 0.0
    assert quality(crew, JOBS) < 1.0


def test_the_supervisor_actually_delegates() -> None:
    crew = delegate(JOBS)
    assert delegated(crew)
    assert {call.agent for call in crew.calls} == {SUPERVISOR, *SKILLS.values()}


def test_a_supervisor_that_keeps_the_work_is_caught() -> None:
    def hoarding(job: Job, tier: str) -> Call:
        return Call(SUPERVISOR, tier, job.kind, job.in_tokens, job.out_tokens, job.id)

    assert not delegated(delegate(JOBS, workers=dict.fromkeys(SKILLS.values(), hoarding)))


def test_one_failing_worker_leaves_the_run_partial_not_dead() -> None:
    def exploding(job: Job, tier: str) -> Call:
        raise TimeoutError("search backend took too long")

    crew = delegate(JOBS, workers={"researcher": exploding, "writer": worker})
    assert crew.status == "partial"
    assert set(crew.errors) == {"j1", "j2"}
    assert [call.job_id for call in crew.calls if call.agent == "writer"] == ["j3"]


def test_planning_happens_once_and_on_the_free_tier() -> None:
    crew = delegate(JOBS)
    plans = [call for call in crew.calls if call.agent == SUPERVISOR]
    assert len(plans) == 1
    assert plans[0].cost == 0.0


def test_cost_is_attributed_per_agent() -> None:
    split = delegate(JOBS).by_agent()
    assert set(split) == {SUPERVISOR, "researcher", "writer"}


def test_the_comparison_reports_both_numbers() -> None:
    table = compare(JOBS, {"tiered": tiered, "all-local": lambda _job: "local"})
    assert "quality" in table
    assert table.count("\n") == 2
