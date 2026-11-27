"""Scoring (bakeoff/score.py) — written before score.py. Fake numbers, no runs.

`runs`: one dict per live run, as bench.py writes them to results/bench.json:
    {"framework": "langgraph", "latency_s": 1.2, "tokens_in": 300,
     "tokens_out": 40, "cost_usd": 0.0, "used_the_tool": True, "answer": "..."}
`probes`: the offline answers per framework:
    {"langgraph": {"durability": True, "recovery": True,
                   "observability": True, "glue_lines": 40}, ...}
"""

from __future__ import annotations

FRAMEWORKS = ("langgraph", "pydantic_ai", "crewai")


def runs_for(latencies: dict[str, list[float]], cost: dict[str, float]) -> list[dict]:
    return [
        {
            "framework": fw,
            "latency_s": s,
            "tokens_in": 300,
            "tokens_out": 40,
            "cost_usd": cost[fw],
            "used_the_tool": True,
            "answer": "Note 2 says: call dentist",
        }
        for fw, times in latencies.items()
        for s in times
    ]


def probe(durability, recovery, observability, glue_lines) -> dict:
    return {
        "durability": durability,
        "recovery": recovery,
        "observability": observability,
        "glue_lines": glue_lines,
    }


FREE = dict.fromkeys(FRAMEWORKS, 0.0)

# Every row has one clear winner.
CLEAR_RUNS = runs_for(
    {
        # median 1.1, spread 1.2 - 1.0 = 0.2; next median 3.0, gap 1.9 > 0.2
        "langgraph": [1.0, 1.1, 1.2, 1.0, 1.1],
        "pydantic_ai": [3.0, 2.9, 3.1, 3.0, 3.0],
        "crewai": [6.0, 6.2, 5.9, 6.1, 6.0],
    },
    {"langgraph": 0.004, "pydantic_ai": 0.001, "crewai": 0.009},
)
CLEAR_PROBES = {
    "langgraph": probe(True, False, False, 60),
    "pydantic_ai": probe(False, True, False, 40),  # 40 vs 60: 50% fewer lines
    "crewai": probe(False, False, True, 90),
}
CLEAR_WINNERS = {
    "durability": "langgraph",
    "recovery": "pydantic_ai",
    "complexity": "pydantic_ai",
    "observability": "crewai",
    "latency": "langgraph",
    "cost": "pydantic_ai",
}

# Every gap is noise.
NOISY_RUNS = runs_for(
    {
        # fastest median 1.5, its spread 2.0 - 1.0 = 1.0; next median 1.8, gap 0.3
        "langgraph": [1.0, 2.0, 1.5, 1.2, 1.8],
        "pydantic_ai": [1.8, 1.7, 1.9, 1.8, 1.8],
        "crewai": [1.9, 2.0, 1.9, 2.1, 1.9],
    },
    FREE,  # local model: every run costs $0
)
NOISY_PROBES = {
    "langgraph": probe(True, True, True, 50),
    "pydantic_ai": probe(True, True, True, 55),  # 10% apart: under 20%
    "crewai": probe(True, True, True, 52),
}


def test_matrix_has_every_row():
    from bakeoff.score import ROWS, matrix

    m = matrix(CLEAR_RUNS, CLEAR_PROBES)
    assert set(m) == set(ROWS)
    assert set(m.values()) <= {*FRAMEWORKS, "undecided"}


def test_real_gaps_have_winners():
    from bakeoff.score import matrix, undecided

    m = matrix(CLEAR_RUNS, CLEAR_PROBES)
    assert m == CLEAR_WINNERS
    assert undecided(m) == []


def test_noise_is_undecided():
    from bakeoff.score import ROWS, matrix, undecided

    m = matrix(NOISY_RUNS, NOISY_PROBES)
    assert m == dict.fromkeys(ROWS, "undecided")
    assert undecided(m) == list(ROWS)


def test_latency_gap_inside_the_spread_is_undecided():
    from bakeoff.score import matrix

    assert matrix(NOISY_RUNS, CLEAR_PROBES)["latency"] == "undecided"


def test_all_free_cost_is_undecided():
    """The expected live result: qwen3.5:9b on Ollama costs $0 for everyone."""
    from bakeoff.score import matrix

    runs = runs_for({fw: [1.0, 1.1, 1.2] for fw in FRAMEWORKS}, FREE)
    assert matrix(runs, CLEAR_PROBES)["cost"] == "undecided"


def test_equal_booleans_are_undecided():
    from bakeoff.score import matrix

    m = matrix(CLEAR_RUNS, NOISY_PROBES)  # every framework answered yes
    assert m["durability"] == m["recovery"] == m["observability"] == "undecided"


def test_undecided_keeps_rows_order():
    from bakeoff.score import undecided

    m = {
        "cost": "undecided",
        "latency": "langgraph",
        "durability": "undecided",
        "recovery": "pydantic_ai",
        "complexity": "undecided",
        "observability": "crewai",
    }
    assert undecided(m) == ["durability", "complexity", "cost"]
