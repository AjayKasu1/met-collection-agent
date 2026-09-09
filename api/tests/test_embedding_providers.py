"""Keep local inference isolated from Google routing and preserve the optional Gemini path."""

from types import SimpleNamespace

import pytest

from met_agent.config import Settings
from met_agent.retrieval import providers
from met_agent.retrieval.schema import EmbeddingIdentity


def test_factory_routes_only_to_the_selected_provider(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_: Settings) -> None:
        pytest.fail("Local inference cannot construct a Gemini router")

    calls: list[tuple[str, int, int]] = []

    def local(
        model: str, dimensions: int, *args: object, threads: int, local_files_only: bool
    ) -> SimpleNamespace:
        assert local_files_only is False
        calls.append((model, dimensions, threads))
        return SimpleNamespace(
            identity=EmbeddingIdentity(provider="local", model=model, dimensions=dimensions)
        )

    monkeypatch.setattr(providers, "create_embedding_router", forbidden)
    monkeypatch.setattr(providers, "LocalEmbedder", local)
    configured = settings.model_copy(
        update={
            "embedding_provider": "local",
            "embedding_model": "intfloat/multilingual-e5-large",
            "embedding_dimensions": 1024,
            "embedding_threads": 2,
        }
    )
    assert providers.create_embedder(configured).identity.provider == "local"
    assert calls == [("intfloat/multilingual-e5-large", 1024, 2)]
    monkeypatch.setattr(providers, "create_embedding_router", lambda _: SimpleNamespace())
    assert providers.create_embedder(settings).identity == EmbeddingIdentity(
        provider="gemini", model=settings.embedding_model, dimensions=768
    )
