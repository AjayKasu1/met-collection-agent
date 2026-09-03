"""Verify JSON rendering, structured secret redaction, and idempotent logging setup."""

import json
import logging
from io import StringIO

import structlog
from pydantic import SecretStr

from met_agent.observability.logging import configure_logging


def test_nested_credentials_are_redacted_without_hiding_usage_metrics() -> None:
    output = StringIO()
    configure_logging("INFO", stream=output)
    structlog.get_logger().info(
        "provider_completed",
        gemini_api_key="unit-test-gemini-secret",
        nested={
            "Authorization": "Bearer unit-test-token",
            "values": [SecretStr("unit-test-wrapped-secret"), {"R2_ACCESS_KEY_ID": "unit-test-id"}],
        },
        settings={"anything": "unit-test-setting-secret"},
        token_count=23,
    )
    entry = json.loads(output.getvalue())
    assert entry["gemini_api_key"] == "[REDACTED]"
    assert entry["nested"]["Authorization"] == "[REDACTED]"
    assert entry["nested"]["values"] == ["[REDACTED]", {"R2_ACCESS_KEY_ID": "[REDACTED]"}]
    assert entry["settings"] == "[REDACTED]"
    assert entry["token_count"] == 23
    assert "unit-test-" not in output.getvalue()


def test_standard_logging_uses_the_same_json_format() -> None:
    output = StringIO()
    configure_logging("INFO", stream=output)
    logging.getLogger("test.runtime").warning("service unavailable")
    entry = json.loads(output.getvalue())
    assert entry["event"] == "service unavailable"
    assert entry["level"] == "warning"
    assert entry["logger"] == "test.runtime"


def test_configuration_is_idempotent_and_respects_log_level() -> None:
    output = StringIO()
    configure_logging("WARNING", stream=output)
    configure_logging("WARNING", stream=output)
    structlog.get_logger().info("suppressed")
    structlog.get_logger().warning("once")
    entries = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [entry["event"] for entry in entries] == ["once"]
