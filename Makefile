.PHONY: install test lint eval gate run corpus docker

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uv run ruff check .

eval:
	uv run securerag eval

gate:
	uv run securerag eval --gate --min-recall 0.95

run:
	uv run securerag serve --port 8000

corpus:
	uv run securerag gen --out corpus

docker:
	docker build -t securerag .
	docker run --rm -p 8000:8000 securerag
