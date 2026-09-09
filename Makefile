UV ?= uv

.PHONY: setup dev lint typecheck test check check-all check-repository web-install web-dev web-check ingest ingest-visitors prepare-images verify-golden publish-index seed qdrant demo demo-check compose-check build-containers

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

check-all: check web-check

web-install:
	pnpm --dir web install --frozen-lockfile

web-dev:
	pnpm --dir web dev

web-check:
	pnpm --dir web check

# Additional arguments remain explicit, for example: make ingest ARGS="--limit 200".
ARGS ?=

qdrant:
	docker compose --env-file /dev/null up -d qdrant
	curl --fail --silent --show-error --retry 20 --retry-delay 1 --retry-connrefused http://127.0.0.1:6333/readyz >/dev/null

demo: qdrant
	QDRANT_URL=http://127.0.0.1:6333 $(MAKE) seed
	docker compose up --build api web

compose-check:
	docker compose config --quiet

build-containers:
	docker compose build api web

ingest:
	$(UV) run --project api --locked python api/scripts/ingest_collection.py $(ARGS)

ingest-visitors:
	$(UV) run --project api --locked python api/scripts/ingest_visitor_info.py $(ARGS)

prepare-images:
	$(UV) run --project api --locked python api/scripts/prepare_image_vectors.py $(ARGS)

verify-golden:
	$(UV) run --project api --locked python api/scripts/verify_golden.py $(ARGS)

publish-index:
	$(UV) run --project api --locked python api/scripts/publish_index.py $(ARGS)

seed:
	$(UV) run --project api --locked python api/scripts/seed.py $(ARGS)

demo-check:
	$(UV) run --project api --locked python api/scripts/demo_check.py

.PHONY: retrieval-check
retrieval-check:
	$(UV) run --project api --locked python api/scripts/retrieval_check.py $(ARGS)

.PHONY: evals evals-quick
evals:
	$(UV) run --project api --locked python evals/run_evals.py $(ARGS)

evals-quick:
	$(UV) run --project api --locked python evals/run_evals.py --quick $(ARGS)
