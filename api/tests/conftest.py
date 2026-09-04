"""Isolate tests from developer credentials and restore process-global logging state."""

import logging
from collections.abc import Iterator

import pytest
import structlog

from met_agent.config import Settings, load_settings


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove only known setting names; never open the developer's .env file."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Prevent application logging setup from escaping into another test."""
    names = (
        "",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
    )
    states = [
        (logger, logger.handlers[:], logger.level, logger.propagate)
        for logger in (logging.getLogger(name) for name in names)
    ]
    original = structlog.get_config()
    yield
    for logger, handlers, level, propagate in states:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
    structlog.configure(**original)
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def valid_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Supply synthetic values that are never sent to any provider."""
    values = {
        "GEMINI_API_KEY": "unit-test-secret-do-not-use",
        "LLM_MODEL": "test/main-model",
        "LLM_MODEL_LITE": "test/lite-model",
        "EMBEDDING_MODEL": "test/embedding-model",
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_DIMENSIONS": "768",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.fixture
def settings(valid_environment: dict[str, str]) -> Settings:
    """Create test configuration without any dotenv lookup."""
    return load_settings(env_file=None)
