include ../../_lesson.mk

.PHONY: test test-integration
test:  ## fast tier: Runner, offline
	uv run pytest -q tests -m "not integration"

test-integration:  ## LangGraph tier: interrupt + checkpointer, offline
	uv run pytest -q tests -m integration
