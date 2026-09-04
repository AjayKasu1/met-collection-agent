"""Exercise settings validation, optional integrations, and secret-safe startup failures."""

from pathlib import Path

import pytest

from met_agent.config import ConfigurationError, Settings, load_settings


def test_missing_required_keys_are_reported_together() -> None:
    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)
    for key in ("GEMINI_API_KEY", "LLM_MODEL", "LLM_MODEL_LITE", "EMBEDDING_MODEL"):
        assert f"{key}: missing" in str(error.value)


def test_defaults_support_direct_provider_and_local_services(settings: Settings) -> None:
    assert settings.use_ai_gateway is False
    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.cors_origins == ("http://localhost:3000",)
    assert settings.optional_services.model_dump() == {
        "ai_gateway": False,
        "fallback_llm": False,
        "langfuse": False,
        "qdrant": True,
        "hugging_face": False,
        "r2": False,
    }


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_blank_required_values_fail(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", value)
    monkeypatch.setenv("LLM_MODEL", value)
    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)
    assert "GEMINI_API_KEY" in str(error.value)
    assert "LLM_MODEL" in str(error.value)


def test_secrets_are_hidden_from_repr_and_json(settings: Settings) -> None:
    secret = settings.gemini_api_key.get_secret_value()
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_MAX_RETRIES", "-1"),
        ("INGEST_MAX_OBJECTS", "0"),
        ("EMBEDDING_BATCH_SIZE", "101"),
        ("USE_AI_GATEWAY", "not-a-bool"),
        ("GIT_SHA", "not-a-git-revision"),
        ("LOG_LEVEL", "unit-test-secret-do-not-use"),
        ("QDRANT_URL", "https://user:unit-test-secret-do-not-use@localhost"),
        ("MET_API_BASE", "https://localhost?key=unit-test-secret-do-not-use"),
    ],
)
def test_invalid_values_fail_without_echoing_input(
    valid_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=f"{name}: invalid") as error:
        load_settings(env_file=None)
    assert "unit-test-secret-do-not-use" not in str(error.value)
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "value",
    [
        '"http://localhost:3000"',
        '["*"]',
        '["https://*.example.org"]',
        '["https://example.org/path"]',
        '["https://user:password@example.org"]',
        '["https://example.org?key=secret"]',
        '["https://example.org#part"]',
        '["https://example.org:bad"]',
        '["https://example.org:0"]',
        '["https://example .org"]',
        "[123]",
    ],
)
def test_cors_rejects_non_origins(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CORS_ORIGINS", value)
    with pytest.raises(ConfigurationError, match="CORS_ORIGINS: invalid"):
        load_settings(env_file=None)


def test_cors_normalizes_and_deduplicates(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CORS_ORIGINS", '["http://localhost:3000/", "http://localhost:3000", "https://museum.org"]'
    )
    assert load_settings(env_file=None).cors_origins == (
        "http://localhost:3000",
        "https://museum.org",
    )


def test_empty_cors_array_disables_cross_origin_access(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "[]")
    assert load_settings(env_file=None).cors_origins == ()


def test_enabled_gateway_requires_both_configuration_values(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USE_AI_GATEWAY", "true")
    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)
    assert "CF_AI_GATEWAY_URL" in str(error.value)
    assert "CF_AI_GATEWAY_TOKEN" in str(error.value)
    monkeypatch.setenv("CF_AI_GATEWAY_URL", "https://gateway.ai.cloudflare.com/v1/account/gateway")
    with pytest.raises(ConfigurationError, match="CF_AI_GATEWAY_TOKEN"):
        load_settings(env_file=None)
    monkeypatch.setenv("CF_AI_GATEWAY_TOKEN", "unit-test-gateway-token")
    assert load_settings(env_file=None).optional_services.ai_gateway is True


def test_incomplete_optional_services_are_disabled(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "unit-test-tracing-secret")
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-fallback-secret")
    monkeypatch.setenv("R2_BUCKET", "collection")
    services = load_settings(env_file=None).optional_services
    assert not services.langfuse
    assert not services.fallback_llm
    assert not services.r2


def test_complete_optional_services_are_enabled(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {
        "LANGFUSE_PUBLIC_KEY": "unit-test-public-key",
        "LANGFUSE_SECRET_KEY": "unit-test-secret-key",
        "LANGFUSE_BASE_URL": "https://localhost:3001",
        "GROQ_API_KEY": "unit-test-groq-key",
        "LLM_MODEL_FALLBACK": "test/fallback-model",
        "HF_DATASET_REPO": "test/collection",
        "R2_ENDPOINT_URL": "https://localhost:9000",
        "R2_BUCKET": "collection",
        "R2_ACCESS_KEY_ID": "unit-test-access-key",
        "R2_SECRET_ACCESS_KEY": "unit-test-storage-secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    services = load_settings(env_file=None).optional_services
    assert services.langfuse and services.fallback_llm and services.hugging_face and services.r2


def test_explicit_dotenv_path_and_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic_file = tmp_path / "synthetic.env"
    synthetic_file.write_text(
        "GEMINI_API_KEY=unit-test-dotenv-key\nLLM_MODEL=test/main-model\n"
        "LLM_MODEL_LITE=test/lite-model\nEMBEDDING_MODEL=test/embedding-model\n"
        "INGEST_MAX_OBJECTS=25\nGROQ_API_KEY=\nEXTERNAL_SETTING=ignored\n"
    )
    monkeypatch.setenv("INGEST_MAX_OBJECTS", "50")
    result = load_settings(env_file=synthetic_file)
    assert result.ingest_max_objects == 50
    assert result.groq_api_key is None
    assert result.gemini_api_key.get_secret_value() == "unit-test-dotenv-key"


def test_dotenv_does_not_search_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=unit-test-parent-secret\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY: missing"):
        load_settings()


def test_example_covers_every_typed_setting() -> None:
    example = Path(__file__).resolve().parents[2] / ".env.example"
    keys = {
        line.split("=", maxsplit=1)[0]
        for line in example.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert keys == {name.upper() for name in Settings.model_fields}


def test_lowercase_log_level_and_comma_separated_origins(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "info")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://museum.example")
    settings = load_settings(env_file=None)
    assert settings.log_level == "INFO"
    assert settings.cors_origins == ("http://localhost:3000", "https://museum.example")


def test_malformed_cors_json_is_reported_without_input(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["private-input"')
    with pytest.raises(ConfigurationError, match="CORS_ORIGINS") as error:
        load_settings(env_file=None)
    assert "private-input" not in str(error.value)
