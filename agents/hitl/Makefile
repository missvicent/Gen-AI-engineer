include ../../_lesson.mk

PY = PYTHONPATH=.:../tools:../react-from-scratch:../../foundations/universal-client

.PHONY: test test-integration live resume
test:  ## fast tier: Runner, offline
	uv run pytest -q tests -m "not integration"

test-integration:  ## LangGraph tier: interrupt + checkpointer, offline
	uv run pytest -q tests -m integration

# make live [GOAL="..."]   (local model: ask before running, ollama stop after)
live:  ## run your agent graph to the pause; kill it at approve? [y/n]
	$(PY) uv run python run_live.py

# make resume THREAD=<id>
resume:  ## new process, same SQLite file: resume the paused thread
	@test -n "$(THREAD)" || { echo "usage: make resume THREAD=<id>"; exit 1; }
	THREAD=$(THREAD) $(PY) uv run python run_live.py
