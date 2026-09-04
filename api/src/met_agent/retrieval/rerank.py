"""Load a pinned Apache-licensed ONNX cross-encoder for local candidate reranking."""

import math
from collections.abc import Sequence
from pathlib import Path
from threading import Lock

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from met_agent.retrieval.embeddings import EmbeddingError

MODEL = "Xenova/ms-marco-MiniLM-L-2-v2"
REVISION = "b84c4fa7efd7b4801931e75773c940f002a494f5"

_REGISTRY_LOCK = Lock()


class LocalReranker:
    """Use English query rewrites for the English records; keep inference off provider APIs."""

    def __init__(self, cache: Path, *, threads: int = 4) -> None:
        with _REGISTRY_LOCK:
            if MODEL not in {entry["model"] for entry in TextCrossEncoder.list_supported_models()}:
                TextCrossEncoder.add_custom_model(
                    model=MODEL,
                    sources=ModelSource(hf=MODEL),
                    license="apache-2.0",
                    size_in_gb=0.06,
                    description="Pinned two-layer MS MARCO MiniLM ONNX cross-encoder",
                )
        try:
            try:
                path = snapshot_download(
                    MODEL,
                    revision=REVISION,
                    token=False,
                    cache_dir=str(cache / "hf"),
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                path = snapshot_download(
                    MODEL,
                    revision=REVISION,
                    token=False,
                    cache_dir=str(cache / "hf"),
                    allow_patterns=["*.json", "onnx/model.onnx"],
                    max_workers=2,
                )
            self.model = TextCrossEncoder(
                model_name=MODEL,
                specific_model_path=path,
                threads=threads,
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            raise EmbeddingError(
                f"Local reranker initialization failed ({type(error).__name__})"
            ) from None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        try:
            scores = [
                float(score) for score in self.model.rerank(query, list(documents), batch_size=2)
            ]
        except Exception as error:
            raise EmbeddingError(f"Local reranking failed ({type(error).__name__})") from None
        if len(scores) != len(documents) or not all(math.isfinite(value) for value in scores):
            raise EmbeddingError("Reranker returned invalid scores")
        return scores
