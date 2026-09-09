FROM ghcr.io/astral-sh/uv:0.11.17 AS uv

FROM python:3.12.11-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY prompts ./prompts
RUN uv sync --locked --no-dev --no-editable

RUN /app/.venv/bin/python -m met_agent.retrieval.prepare_models --cache-dir /app/models

FROM python:3.12.11-slim-bookworm AS runtime

ENV DATA_DIR=/app/data \
    MODEL_CACHE_DIR=/app/models \
    MODELS_OFFLINE=true \
    STARTUP_WARMUP=true \
    HF_HOME=/app/models/hf \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /app/data /home/app \
    && chown -R app:app /app /home/app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/models /app/models

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=180s --retries=5 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=3)"]

CMD ["sh", "-c", "exec uvicorn met_agent.main:create_app --factory --host 0.0.0.0 --port \"${PORT:-8000}\" --no-access-log"]
