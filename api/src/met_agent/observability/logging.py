"""Emit consistent JSON logs and redact structured credentials before rendering."""

import logging
import sys
from collections.abc import Mapping
from typing import TextIO

import structlog
from pydantic import SecretStr
from structlog.typing import EventDict, Processor, WrappedLogger

_PRIVATE_FIELDS = frozenset(
    {
        "authorization",
        "cookie",
        "set_cookie",
        "password",
        "secret",
        "token",
        "api_key",
        "credentials",
        "settings",
        "config",
        "body",
        "query_string",
    }
)


def _redact(value: object) -> object:
    if isinstance(value, SecretStr):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            normalized = name.lower().replace("-", "_")
            private = normalized in _PRIVATE_FIELDS or normalized.endswith(
                ("_api_key", "_token", "_secret_key", "_access_key", "_access_key_id", "_password")
            )
            result[name] = "[REDACTED]" if private else _redact(item)
        return result
    if isinstance(value, list | tuple):
        return [_redact(item) for item in value]
    return value


def redact_credentials(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Defense in depth for structured fields; never log raw request or settings text."""
    for key, value in list(event_dict.items()):
        redacted = _redact({key: value})
        if isinstance(redacted, dict):
            event_dict[key] = redacted[key]
    return event_dict


def configure_logging(level: str, stream: TextIO | None = None) -> None:
    """Route application and standard-library logs through one JSON renderer.

    Replacing handlers makes repeated application-factory calls idempotent. Uvicorn
    access logging is disabled by the dev command because it logs raw query strings.
    """
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_credentials,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # These libraries can include request URLs, payloads, or keys in debug logs.
    for name in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
