include ../../../../../_lesson.mk

.PHONY: nltk-data
nltk-data:  ## download NLTK tokenizer + stopwords once (skipped if present)
	@uv run python -c "import nltk; nltk.data.find('tokenizers/punkt_tab'); nltk.data.find('corpora/stopwords')" 2>/dev/null \
		|| uv run python -m nltk.downloader -q punkt_tab stopwords

.PHONY: test
test: nltk-data  ## run tests with pytest
	uv run pytest

.PHONY: recall
recall: nltk-data  ## exact-slice recall@k: dense / hybrid / hybrid+rerank
	PYTHONPATH=src uv run --group release python scripts/recall.py

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@echo "This is the first layer. There is nothing to copy." && exit 1
