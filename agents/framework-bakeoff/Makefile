include ../../_lesson.mk

PY = PYTHONPATH=.:../tools

.PHONY: test live
test:  ## offline: contract, LangGraph, Pydantic AI, recovery, scoring (CrewAI skipped)
	$(PY) uv run pytest -q tests -m "not integration"

# Local model: ask before running, `ollama stop qwen3.5:9b` after.
live:  ## CrewAI live test + bench.py (5 runs per framework) -> results/bench.json
	$(PY) uv run pytest -q tests -m integration
	$(PY) uv run python bench.py
