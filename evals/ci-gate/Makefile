include ../../_lesson.mk

.PHONY: test gate
test:             ## all tests, offline, fixtures only
	uv run pytest -q tests

gate:             ## fail the build on a regression (no model needed)
	uv run python -m ci_gate.gate evals/results.json evals/baseline.json

check::           ## adds the type check to the shared lint + format check
	uv run pyright ci_gate tests

# The judged tier keeps its own files, so the fast tier's committed course run and
# baseline stay untouched (a toy-pipeline run would fail the bars on every PR).
# Rows per slice. The baseline and every later run must use the same number.
ROWS ?= 10
.PHONY: eval judged-gate
eval:             ## the judged tier: real RAGAS judge (Ollama qwen2.5:7b-instruct)
	cd ../llm-judge && uv run --group integration python run_eval.py $(ROWS) ../ci-gate/evals/judged-results.json

judged-gate:      ## fresh judged results vs the committed judged baseline
	uv run python -m ci_gate.gate evals/judged-results.json evals/judged-baseline.json
