include ../../_lesson.mk

.PHONY: test test-integration
test:             ## the fast tier: whole harness, fake judges, no model
	uv run pytest -q -m "not integration" tests

test-integration: ## the judged tier: real RAGAS judge, needs Ollama running
	uv run --group integration pytest -q -m integration tests
