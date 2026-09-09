"""Keep initialization outside customer traffic and reject unusable deployment caches."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import LocalEntryNotFoundError
from qdrant_client import QdrantClient

from met_agent import main
from met_agent.config import Settings
from met_agent.retrieval import local_embeddings, prepare_models, rerank, service
from met_agent.retrieval.embeddings import EmbeddingError
from met_agent.runtime import Runtime


@pytest.mark.parametrize("provider", ["local", "gemini"])
def test_concurrent_warmup_is_once_and_does_not_call_remote_embeddings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> None:
    dense, sparse, rank = MagicMock(), MagicMock(), MagicMock()
    load_dense = MagicMock(return_value=dense)
    load_sparse = MagicMock(return_value=sparse)
    load_rank = MagicMock(return_value=rank)
    monkeypatch.setattr(service, "create_embedder", load_dense)
    monkeypatch.setattr(service, "BM25Embedder", load_sparse)
    monkeypatch.setattr(service, "LocalReranker", load_rank)
    config = settings.model_copy(
        update={
            "embedding_provider": provider,
            "model_cache_dir": tmp_path,
            "models_offline": True,
        }
    )
    with closing(QdrantClient(location=":memory:")) as client:
        search = service.SearchService(client, config)
        assert not search.warmed
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: search.warmup(), range(4)))
        assert search.warmed
        load_dense.assert_called_once()
        load_sparse.assert_called_once_with(tmp_path, local_files_only=True)
        load_rank.assert_called_once_with(tmp_path, threads=4, local_files_only=True)
        assert dense.embed.call_count == (1 if provider == "local" else 0)
        sparse.embed.assert_called_once()
        rank.score.assert_called_once()


def test_failed_inference_does_not_set_readiness(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    rank = MagicMock()
    rank.score.side_effect = EmbeddingError("failed inference")
    monkeypatch.setattr(service, "create_embedder", lambda _: MagicMock())
    monkeypatch.setattr(service, "BM25Embedder", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(service, "LocalReranker", lambda *a, **kw: rank)
    with closing(QdrantClient(location=":memory:")) as client:
        search = service.SearchService(client, settings)
        with pytest.raises(EmbeddingError):
            search.warmup()
        assert not search.warmed


@pytest.mark.parametrize("fail", [False, True])
def test_lifespan_gates_requests_on_warmup_and_closes_failed_runtime(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    runtime = MagicMock()
    if fail:
        runtime.warmup.side_effect = ValueError("private-provider-details")
    monkeypatch.setattr(main, "Runtime", lambda _: runtime)
    app = main.create_app(settings.model_copy(update={"startup_warmup": True}))
    if fail:
        with (
            pytest.raises(RuntimeError, match=r"^Retrieval startup warmup failed$"),
            TestClient(app),
        ):
            pytest.fail("A failed warmup must not accept traffic")
    else:
        with TestClient(app) as client:
            runtime.warmup.assert_called_once()
            assert client.get("/health").status_code == 200
    runtime.close.assert_called_once()


def test_runtime_readiness_includes_model_warmup(settings: Settings) -> None:
    runtime = MagicMock()
    runtime.settings = settings.model_copy(update={"startup_warmup": True})
    runtime.search.warmed = False
    runtime.events.ready.return_value = True
    runtime.readiness.side_effect = lambda: Runtime.readiness(runtime)
    assert not Runtime.readiness(runtime)["retrieval_models"]
    runtime.search.warmup.side_effect = lambda: setattr(runtime.search, "warmed", True)
    Runtime.warmup(runtime)
    assert Runtime.readiness(runtime)["retrieval_models"]
    assert runtime.search.store.call_count == 2
    runtime.events.ready.return_value = False
    with pytest.raises(RuntimeError, match="Required dependencies"):
        Runtime.warmup(runtime)


@pytest.mark.parametrize("kind", ["dense", "reranker"])
def test_missing_offline_weights_fail_without_network_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    download = MagicMock(side_effect=LocalEntryNotFoundError("uncached-private-path"))
    module = local_embeddings if kind == "dense" else rerank
    monkeypatch.setattr(module, "snapshot_download", download)
    with pytest.raises(EmbeddingError) as caught:
        if kind == "dense":
            local_embeddings.LocalEmbedder(
                local_embeddings.LOCAL_MODEL, 1024, tmp_path, local_files_only=True
            )
        else:
            rerank.LocalReranker(tmp_path, local_files_only=True)
    download.assert_called_once()
    assert download.call_args.kwargs["local_files_only"] is True
    expected = local_embeddings.LOCAL_FILES if kind == "dense" else rerank.MODEL_FILES
    assert download.call_args.kwargs["allow_patterns"] == list(expected)
    assert "uncached-private-path" not in str(caught.value)


def test_build_cache_fetches_only_pinned_weights_and_checks_sparse_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(prepare_models, "snapshot_download", lambda *a, **kw: calls.append(kw))
    sparse = MagicMock()
    monkeypatch.setattr(prepare_models, "BM25Embedder", sparse)
    prepare_models.prepare(tmp_path)
    assert [call["revision"] for call in calls] == [
        local_embeddings.LOCAL_WEIGHTS_REVISION,
        rerank.REVISION,
    ]
    assert all(call["token"] is False for call in calls)
    assert sparse.call_args.kwargs["local_files_only"] is True
