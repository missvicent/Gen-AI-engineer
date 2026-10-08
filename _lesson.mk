.PHONY: setup check format lint

setup:
	uv sync

check:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .
