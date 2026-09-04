"""Build the ingestion embedding route with explicit provider configuration and bounded retries."""

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from met_agent.config import ConfigurationError, Settings

if TYPE_CHECKING:
    from litellm.router import Router


def _gateway_base(url: str) -> str:
    """Accept gateway roots or native Google bases; reject Workers AI inference URLs."""
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    suffix = parts[3:]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "gateway.ai.cloudflare.com"
        or parsed.query
        or parsed.fragment
        or len(parts) < 3
        or parts[0] != "v1"
        or not all(parts)
        or suffix
        not in (
            [],
            ["google-ai-studio"],
            ["google-ai-studio", "v1"],
            ["google-ai-studio", "v1beta"],
        )
    ):
        raise ConfigurationError(
            "CF_AI_GATEWAY_URL must be https://gateway.ai.cloudflare.com/v1/ACCOUNT/GATEWAY "
            "with an optional /google-ai-studio/v1beta suffix"
        )
    base = url.rstrip("/")
    if not suffix:
        base += "/google-ai-studio"
    if len(suffix) < 2:
        base += "/v1beta"
    return base


def embedding_route(settings: Settings) -> dict[str, Any]:
    """Add LiteLLM's provider prefix without changing the configured Google model identifier."""
    model = "gemini/" + settings.embedding_model.removeprefix("gemini/").removeprefix("models/")
    parameters: dict[str, Any] = {
        "model": model,
        "api_key": settings.gemini_api_key.get_secret_value(),
    }
    if settings.use_ai_gateway:
        if settings.cf_ai_gateway_url is None or settings.cf_ai_gateway_token is None:
            raise ValueError("AI Gateway configuration is incomplete")
        parameters["api_base"] = _gateway_base(str(settings.cf_ai_gateway_url))
        parameters["extra_headers"] = {
            "cf-aig-authorization": "Bearer " + settings.cf_ai_gateway_token.get_secret_value(),
        }
    return {"model_name": "embedding", "litellm_params": parameters}


def create_embedding_router(settings: Settings) -> "Router":
    """Retry 429, timeouts, and transient provider failures without changing vector spaces."""
    from litellm.router import Router
    from litellm.types.router import RetryPolicy

    # Provider exceptions can embed headers in free-form logs. The caller emits
    # sanitized error classes and ingestion progress through application logging.
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        logger = logging.getLogger(name)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False

    return Router(
        model_list=[embedding_route(settings)],
        timeout=settings.llm_timeout_seconds,
        num_retries=settings.llm_max_retries,
        retry_after=2,
        retry_policy=RetryPolicy(
            BadRequestErrorRetries=0,
            AuthenticationErrorRetries=0,
            TimeoutErrorRetries=settings.llm_max_retries,
            RateLimitErrorRetries=settings.llm_max_retries,
            InternalServerErrorRetries=settings.llm_max_retries,
        ),
        fallbacks=[],
        max_fallbacks=0,
        disable_cooldowns=True,
    )
