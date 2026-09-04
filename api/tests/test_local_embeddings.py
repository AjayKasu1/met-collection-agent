"""Verify local prefixes, pinned assets, output checks, and provider isolation offline."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from met_agent.retrieval import local_embeddings as local
from met_agent.retrieval.embeddings import EmbeddingError


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(inputs=[], downloads=[], vectors=None, failure=False, tokens=4)

    class Model:
        @staticmethod
        def list_supported_models() -> list[dict[str, object]]:
            return [{"model": local.LOCAL_MODEL, "dim": 1024}]

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["specific_model_path"] == "/synthetic/pinned"
            assert kwargs["providers"] == ["CPUExecutionProvider"]

        def embed(self, texts: list[str], **kwargs: Any) -> list[SimpleNamespace]:
            if state.failure:
                raise RuntimeError("private-input")
            state.inputs.extend(texts)
            vectors = state.vectors if state.vectors is not None else [[2.0] * 1024 for _ in texts]
            return [SimpleNamespace(tolist=lambda vector=vector: vector) for vector in vectors]

    def download(repo: str, **kwargs: Any) -> str:
        assert repo == local.LOCAL_WEIGHTS_REPO
        assert kwargs["revision"] == local.LOCAL_WEIGHTS_REVISION and kwargs["token"] is False
        state.downloads.append(kwargs)
        return "/synthetic/pinned"

    monkeypatch.setattr(local, "TextEmbedding", Model)
    state.download = download
    monkeypatch.setattr(local, "snapshot_download", download)
    monkeypatch.setattr(
        local,
        "Tokenizer",
        SimpleNamespace(
            from_file=lambda _: SimpleNamespace(
                no_truncation=lambda: None,
                no_padding=lambda: None,
                encode_batch=lambda texts: [SimpleNamespace(ids=[1] * state.tokens) for _ in texts],
            )
        ),
    )
    return state


def test_local_prefixes_and_normalization_for_multilingual_queries(
    backend: SimpleNamespace, tmp_path: Path
) -> None:
    model = local.LocalEmbedder(local.LOCAL_MODEL, 1024, tmp_path)
    assert model.identity.provider == "local"
    assert model.embed([]) == []
    queries = ["Où est ce tableau ?", "¿Qué obra es similar?", "这是什么作品?"]
    assert all(
        sum(value * value for value in vector) == pytest.approx(1)
        for vector in model.embed(queries, purpose="query")
    )
    model.embed(["Vessel"])
    assert backend.inputs == ["query: " + query for query in queries] + ["passage: Vessel"]
    assert backend.downloads[0]["local_files_only"] is True
    backend.tokens = 513
    with pytest.raises(EmbeddingError, match="512-token"):
        model.embed(["Long passage"])


@pytest.mark.parametrize(
    "vectors", [[], [[0.0] * 1024], [[1.0]], [[float("nan")] * 1024], [[float("inf")] * 1024]]
)
def test_local_invalid_vectors_cannot_reach_an_index(
    backend: SimpleNamespace, tmp_path: Path, vectors: list[list[float]]
) -> None:
    model = local.LocalEmbedder(local.LOCAL_MODEL, 1024, tmp_path)
    backend.vectors = vectors
    with pytest.raises(EmbeddingError, match="validation"):
        model.embed(["Vessel"])


def test_local_model_validation_download_fallback_and_safe_errors(
    backend: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(EmbeddingError, match="require"):
        local.LocalEmbedder("wrong-model", 1024, tmp_path)
    with pytest.raises(EmbeddingError, match="require"):
        local.LocalEmbedder(local.LOCAL_MODEL, 768, tmp_path)
    original = backend.download

    def uncached(repo: str, **kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return str(original(repo, **kwargs))

    monkeypatch.setattr(local, "snapshot_download", uncached)
    model = local.LocalEmbedder(local.LOCAL_MODEL, 1024, tmp_path)
    backend.failure = True
    with pytest.raises(EmbeddingError, match="inference failed") as caught:
        model.embed(["Vessel"])
    assert "private-input" not in str(caught.value)
    monkeypatch.setattr(
        local,
        "snapshot_download",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("private-input")),
    )
    with pytest.raises(EmbeddingError, match="initialization failed"):
        local.LocalEmbedder(local.LOCAL_MODEL, 1024, tmp_path)
    monkeypatch.setattr(local, "TextEmbedding", SimpleNamespace(list_supported_models=lambda: []))
    with pytest.raises(EmbeddingError, match="registry"):
        local.LocalEmbedder(local.LOCAL_MODEL, 1024, tmp_path)
