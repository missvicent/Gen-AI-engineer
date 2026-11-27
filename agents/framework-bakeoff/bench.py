"""Live bench: each framework answers TASK 5 times on qwen3.5:9b (Ollama).

Writes results/bench.json, one dict per run, in the shape score.py reads:
    {"framework", "latency_s", "tokens_in", "tokens_out", "cost_usd",
     "used_the_tool", "answer"}
A run that crashes is recorded with "error" instead of stopping the bench.

Run: `make live` (asks first). Always ends with `ollama stop qwen3.5:9b`.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import agent_tools.tools as notes
from bakeoff.contract import TASK, used_the_tool
from bakeoff.run_crewai import run_crewai
from bakeoff.run_langgraph import run_langgraph
from bakeoff.run_pydantic_ai import run_pydantic_ai
from dotenv import load_dotenv

MODEL = "qwen3.5:9b"
RUNS = 5
OUT = Path(__file__).parent / "results" / "bench.json"
RUNNERS = {
    "langgraph": run_langgraph,
    "pydantic_ai": run_pydantic_ai,
    "crewai": run_crewai,
}


def one_run(framework: str, run) -> dict:
    start = time.perf_counter()
    try:
        r = run(TASK, notes.read_note)  # model=None: the real qwen3.5:9b
    except Exception as e:  # a crash is a measurement too
        return {
            "framework": framework,
            "latency_s": round(time.perf_counter() - start, 3),
            "error": f"{type(e).__name__}: {e}",
        }
    return {
        "framework": framework,
        "latency_s": round(time.perf_counter() - start, 3),
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "cost_usd": 0.0,  # local model: free
        "used_the_tool": used_the_tool(r),
        "answer": r.answer,
    }


def main() -> None:
    load_dotenv()
    runs = []
    try:
        for framework, run in RUNNERS.items():
            for i in range(RUNS):
                row = one_run(framework, run)
                print(framework, i + 1, row.get("latency_s"), row.get("error", "ok"))
                runs.append(row)
    finally:
        OUT.write_text(json.dumps(runs, indent=2))
        subprocess.run(["ollama", "stop", MODEL], check=False)
    print(f"wrote {OUT} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
