include ../../_lesson.mk

.PHONY: test live
test:  ## fast tests, offline, scripted brain
	uv run pytest -q tests

live:  ## one real run against a frontier model (costs API credits)
	PYTHONPATH=../../foundations/universal-client uv run python run_live.py
