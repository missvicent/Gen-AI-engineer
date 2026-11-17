include ../../_lesson.mk

.PHONY: test live
test:  ## fast tests, offline, in-memory notes
	uv run pytest -q tests

# make live [GOAL="..."] [BLANK=draft_reply]   (local model: ask before running)
live:  ## one run on the local model; BLANK hides that tool's docstring
	PYTHONPATH=.:../react-from-scratch:../../foundations/universal-client uv run python run_live.py
