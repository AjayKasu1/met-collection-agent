"""Build the ingestion embedding route with explicit provider configuration and bounded retries."""

import logging
from typing import TYPE_CHECKING, Any

from met_agent.config import Settings

if TYPE_CHECKING:
    from litellm.router import Router


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
        base = str(settings.cf_ai_gateway_url).rstrip("/")
        if "/google-ai-studio" not in base:
            base += "/google-ai-studio"
        if base.endswith("/google-ai-studio"):
            base += "/v1beta"
        parameters["api_base"] = base
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
