include ../../_lesson.mk

.PHONY: test report my-report gate rows label
test:             ## all tests, offline, fixtures only
	uv run pytest -q tests

report:           ## print the calibration report for evals/labeled.jsonl
	uv run python -m judge_calibration.calibration

my-report:        ## print the calibration report for evals/my_rows.jsonl
	uv run python -m judge_calibration.calibration evals/my_rows.jsonl

rows:             ## build evals/my_rows.jsonl with Ollama; N=3 limits the row count
	PYTHONPATH=../llm-judge uv run --group integration python -m judge_calibration.build_rows $(N)

label:            ## label evals/my_rows.jsonl yourself, one row at a time
	uv run python -m judge_calibration.label_rows

gate: check test report  ## everything this lesson must pass before a commit
