include ../../_lesson.mk

.PHONY: test
test:  ## fast tests, offline, fake SDK clients
	PYTHONPATH=. uv run pytest -q tests
