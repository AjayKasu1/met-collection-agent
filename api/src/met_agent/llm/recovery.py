"""Extract allowlisted diagnostics without retaining provider bodies or credentials."""

import math
import re

from pydantic import JsonValue


def is_groq_tool_protocol_failure(error: Exception, *, model: str) -> bool:
    """Recognize only Groq's typed tool-generation failure without retaining its body.

    Groq may return a failed generation in the same response body. That content is
    untrusted and can contain model output, so only the exact provider error code is
    inspected and nothing from the body is logged.
    """

    if (
        model
        not in {
            "groq/openai/gpt-oss-120b",
            "groq/openai/gpt-oss-20b",
        }
        or getattr(error, "llm_provider", None) != "groq"
    ):
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    provider_error = body.get("error")
    return isinstance(provider_error, dict) and provider_error.get("code") == "tool_use_failed"


def failure_details(error: Exception, *, model: str, route: str) -> dict[str, JsonValue]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    status = getattr(response, "status_code", None) or getattr(error, "status_code", None)
    details: dict[str, JsonValue] = {
        "model": model,
        "route": route,
        "error_type": type(error).__name__,
        "status": status if isinstance(status, int) else None,
    }
    for header, key in (
        ("retry-after", "retry_after"),
        ("x-ratelimit-remaining-tokens", "remaining_tokens"),
        ("x-ratelimit-remaining-requests", "remaining_requests"),
        ("x-ratelimit-reset-tokens", "token_reset_after"),
        ("x-ratelimit-reset-requests", "request_reset_after"),
    ):
        value = str(headers.get(header, ""))
        if re.fullmatch(r"[0-9.]+(?:ms|s|m|h|d)?(?:[0-9.]+(?:ms|s|m|h|d))*", value):
            details[key] = value[:80]
    return details


def retry_delay(details: dict[str, JsonValue]) -> float:
    """Return the longest trusted provider reset hint in seconds.

    Groq's Retry-After is a number of seconds, while token reset headers use
    compact durations such as ``7.66s`` or ``2m59.56s``.
    """

    return max(
        _duration_seconds(details.get("retry_after")),
        _duration_seconds(details.get("token_reset_after")),
    )


def _duration_seconds(raw: JsonValue | None) -> float:
    value = str(raw or "")
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except ValueError:
        pass
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h|d)", value)
    if not parts or "".join(number + unit for number, unit in parts) != value:
        return 0.0
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    seconds = sum(float(number) * multipliers[unit] for number, unit in parts)
    return seconds if math.isfinite(seconds) else 0.0
