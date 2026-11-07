include ../../../../../_lesson.mk

.PHONY: test
test:  ## run tests with pytest
	uv run pytest

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/rag.py" src/assistant/rag.py
