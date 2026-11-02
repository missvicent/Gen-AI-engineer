include ../../_lesson.mk

.PHONY: test report gate
test:             ## the 12 calibration tests, offline, fixtures only
	uv run pytest -q tests

report:           ## print the calibration report for evals/labeled.jsonl
	uv run python -m judge_calibration.calibration

gate: check test report  ## everything this lesson must pass before a commit
