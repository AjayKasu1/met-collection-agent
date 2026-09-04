"""Resolve explicit model credentials and provider-native gateway paths for chat."""

from typing import Any

from met_agent.config import Settings, model_provider
from met_agent.llm.router import _gateway_base


def google_model(name: str) -> str:
    return "gemini/" + name.removeprefix("google-ai-studio/").removeprefix("gemini/").removeprefix(
        "models/"
    )


def configured_models(settings: Settings) -> dict[str, str]:
    models = {"lite": settings.llm_model_lite, "main": settings.llm_model}
    for alias, model in (
        ("fallback", settings.llm_model_fallback),
        ("fallback_2", settings.llm_model_fallback_2),
    ):
        if model is not None and settings.provider_key(model_provider(model)) is not None:
            models[alias] = model
    return {
        alias: google_model(model) if model_provider(model) == "gemini" else model
        for alias, model in models.items()
    }


def chat_routes(settings: Settings) -> list[dict[str, Any]]:
    routes = []
    for alias, model in configured_models(settings).items():
        provider = model_provider(model)
        params: dict[str, Any] = {
            "model": model,
            "api_key": settings.require_api_key(provider).get_secret_value(),
        }
        if settings.use_ai_gateway:
            if settings.cf_ai_gateway_url is None or settings.cf_ai_gateway_token is None:
                raise ValueError("AI Gateway configuration is incomplete")
            google_base = _gateway_base(str(settings.cf_ai_gateway_url))
            root = google_base.rsplit("/google-ai-studio/", 1)[0]
            params.update(
                api_base=google_base if provider == "gemini" else f"{root}/{provider}",
                extra_headers={
                    "cf-aig-authorization": "Bearer "
                    + settings.cf_ai_gateway_token.get_secret_value()
                },
            )
        routes.append({"model_name": alias, "litellm_params": params})
    return routes
