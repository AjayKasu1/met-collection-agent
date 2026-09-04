UV ?= uv

.PHONY: setup dev lint typecheck test check check-repository ingest ingest-visitors verify-golden qdrant

setup:
	$(UV) sync --project api --locked
	$(UV) run --project api --locked pre-commit install

dev:
	$(UV) run --project api --locked uvicorn met_agent.main:create_app --factory --host 127.0.0.1 --port 8000 --reload --reload-dir api/src --no-access-log

lint:
	$(UV) run --project api --locked ruff check api
	$(UV) run --project api --locked ruff format --check api

typecheck:
	$(UV) run --directory api --locked mypy

test:
	$(UV) run --directory api --locked pytest --cov=met_agent --cov-report=term-missing

check-repository:
	$(UV) run --project api --locked python api/scripts/check_repository.py

check: lint typecheck test check-repository

# Additional arguments remain explicit, for example: make ingest ARGS="--limit 200".
ARGS ?=

qdrant:
	docker compose --env-file /dev/null up -d qdrant

ingest:
	$(UV) run --project api --locked python api/scripts/ingest_collection.py $(ARGS)

ingest-visitors:
	$(UV) run --project api --locked python api/scripts/ingest_visitor_info.py $(ARGS)

verify-golden:
	$(UV) run --project api --locked python api/scripts/verify_golden.py $(ARGS)
