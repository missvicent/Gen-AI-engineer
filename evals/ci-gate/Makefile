include ../../_lesson.mk

.PHONY: test gate
test:             ## all tests, offline, fixtures only
	uv run pytest -q tests

gate:             ## fail the build on a regression (no model needed)
	uv run python -m ci_gate.gate evals/results.json evals/baseline.json
