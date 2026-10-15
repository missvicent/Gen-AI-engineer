include ../../../../../_lesson.mk

.PHONY: test
test:  ## run tests with pytest
	uv run pytest

.PHONY: recall
recall:  ## exact-slice recall@k: dense / hybrid / hybrid+rerank
	PYTHONPATH=src uv run --group release python scripts/recall.py

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@echo "This is the first layer. There is nothing to copy." && exit 1
