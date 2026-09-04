"""Extract allowlisted diagnostics without retaining provider bodies or credentials."""

import math
import re

from pydantic import JsonValue


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
    try:
        value = float(str(details.get("retry_after", 0)))
        return max(0.0, value) if math.isfinite(value) else 0.0
    except ValueError:
        return 0.0
