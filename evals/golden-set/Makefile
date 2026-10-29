include ../../_lesson.mk

.PHONY: test validate gate
test:             ## the 25 dataset + milestone tests
	uv run pytest -q tests

validate:         ## print the dataset report for evals/golden.jsonl
	uv run python -m golden_set.dataset

gate: check test validate  ## everything this lesson must pass before a commit
