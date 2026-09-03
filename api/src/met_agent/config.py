"""Load and validate process configuration at the application's sole environment boundary.

Only this module reads environment variables or dotenv files. Secret values stay
wrapped in SecretStr; callers must explicitly unwrap them at a provider boundary.
"""

import json
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Credential = Annotated[SecretStr, Field(min_length=1)]
CorsOrigins = Annotated[tuple[str, ...], NoDecode]


class OptionalServices(BaseModel):
    """Public configuration flags, deliberately separate from private settings."""

    ai_gateway: bool
    fallback_llm: bool
    langfuse: bool
    qdrant: bool
    hugging_face: bool
    r2: bool


class Settings(BaseSettings):
    """Typed settings shared by API, ingestion, evaluation, and deployment tooling.

    Model identifiers have no guessed defaults. Optional integrations activate only
    when their configuration is complete, except an explicitly enabled AI Gateway,
    which requires its URL and token. Dotenv loading is reserved for load_settings.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    git_sha: Annotated[str, Field(pattern=r"^(?:unknown|[0-9a-f]{7,40})$")] = "unknown"
    cors_origins: CorsOrigins = ("http://localhost:3000",)

    gemini_api_key: Credential = Field(repr=False)
    llm_model: NonEmptyString
    llm_model_lite: NonEmptyString
    embedding_model: NonEmptyString
    llm_model_fallback: NonEmptyString | None = None
    groq_api_key: Credential | None = Field(default=None, repr=False)
    llm_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 60
    llm_max_retries: Annotated[int, Field(ge=0, le=5)] = 2

    use_ai_gateway: bool = False
    cf_ai_gateway_url: HttpUrl | None = None
    cf_ai_gateway_token: Credential | None = Field(default=None, repr=False)
    cf_account_id: NonEmptyString | None = None
    cloudflare_api_token: Credential | None = Field(default=None, repr=False)

    qdrant_url: HttpUrl = HttpUrl("http://localhost:6333")
    qdrant_api_key: Credential | None = Field(default=None, repr=False)
    qdrant_collection: NonEmptyString = "met_objects"
    qdrant_visitor_collection: NonEmptyString = "met_visitor_info"
    met_api_base: HttpUrl = HttpUrl("https://collectionapi.metmuseum.org/public/collection/v1")
    ingest_max_objects: Annotated[int, Field(gt=0)] = 10_000
    embedding_batch_size: Annotated[int, Field(gt=0, le=100)] = 32
    data_dir: Path = Path("data")

    langfuse_public_key: Credential | None = Field(default=None, repr=False)
    langfuse_secret_key: Credential | None = Field(default=None, repr=False)
    langfuse_base_url: HttpUrl | None = None
    hf_dataset_repo: NonEmptyString | None = None
    hf_token: Credential | None = Field(default=None, repr=False)
    r2_endpoint_url: HttpUrl | None = None
    r2_access_key_id: Credential | None = Field(default=None, repr=False)
    r2_secret_access_key: Credential | None = Field(default=None, repr=False)
    r2_bucket: NonEmptyString | None = None
    gcp_region: NonEmptyString | None = None
    next_public_api_url: HttpUrl = HttpUrl("http://localhost:8000")

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        """Reject whitespace-only credentials and normalize copied configuration."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> tuple[str, ...]:
        """Accept JSON arrays of explicit HTTP origins, never wildcard credentials."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("CORS_ORIGINS must be a JSON array of origins") from None
        if not isinstance(value, list | tuple):
            raise ValueError("CORS_ORIGINS must be a JSON array of origins")
        origins: list[str] = []
        for origin in value:
            if not isinstance(origin, str):
                raise ValueError("Every CORS origin must be a string")
            try:
                url = urlsplit(origin)
                port = url.port
            except ValueError:
                raise ValueError("Invalid CORS origin") from None
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.path not in {"", "/"}
                or url.query
                or url.fragment
                or port == 0
                or "*" in origin
                or any(character.isspace() for character in origin)
            ):
                raise ValueError("CORS origins must contain only an HTTP scheme, host, and port")
            normalized = origin.rstrip("/")
            if normalized not in origins:
                origins.append(normalized)
        return tuple(origins)

    @field_validator(
        "cf_ai_gateway_url",
        "qdrant_url",
        "met_api_base",
        "langfuse_base_url",
        "r2_endpoint_url",
        "next_public_api_url",
    )
    @classmethod
    def reject_url_credentials(cls, value: HttpUrl | None) -> HttpUrl | None:
        """Keep credentials in secret fields instead of URLs or query strings."""
        if value is not None and (
            value.username is not None
            or value.password is not None
            or value.query is not None
            or value.fragment is not None
        ):
            raise ValueError("Service URLs cannot include credentials, query strings, or fragments")
        return value

    @model_validator(mode="after")
    def validate_gateway(self) -> Self:
        """An explicitly enabled gateway must never silently bypass authentication."""
        if self.use_ai_gateway:
            missing = []
            if self.cf_ai_gateway_url is None:
                missing.append("CF_AI_GATEWAY_URL")
            if self.cf_ai_gateway_token is None:
                missing.append("CF_AI_GATEWAY_TOKEN")
            if missing:
                raise PydanticCustomError(
                    "missing_configuration",
                    "Required settings: {keys}",
                    {"keys": ", ".join(missing)},
                )
        return self

    @property
    def optional_services(self) -> OptionalServices:
        """Report configured integrations without claiming connectivity or readiness."""
        return OptionalServices(
            ai_gateway=self.use_ai_gateway,
            fallback_llm=bool(self.groq_api_key and self.llm_model_fallback),
            langfuse=bool(
                self.langfuse_public_key and self.langfuse_secret_key and self.langfuse_base_url
            ),
            qdrant=True,
            hugging_face=bool(self.hf_dataset_repo),
            r2=bool(
                self.r2_endpoint_url
                and self.r2_access_key_id
                and self.r2_secret_access_key
                and self.r2_bucket
            ),
        )


class ConfigurationError(RuntimeError):
    """A safe startup error that names settings but never includes their values."""


def load_settings(env_file: Path | None = Path(".env")) -> Settings:
    """Read process env and an explicit dotenv path without searching parent folders.

    Run the service from the repository root to load its .env, or inject process
    environment variables in containers. Pass None to disable dotenv loading.
    """
    try:
        return Settings(_env_file=env_file)
    except ValidationError as error:
        problems = []
        for item in error.errors(include_input=False, include_context=False):
            if item["type"] == "missing_configuration":
                problems.append(item["msg"])
            else:
                name = ".".join(str(part).upper() for part in item["loc"])
                reason = "missing" if item["type"] == "missing" else "invalid"
                problems.append(f"{name}: {reason}")
        raise ConfigurationError("Invalid configuration: " + "; ".join(problems)) from None
    except (SettingsError, OSError):
        raise ConfigurationError(
            "Unable to load configuration; check .env syntax and access"
        ) from None
