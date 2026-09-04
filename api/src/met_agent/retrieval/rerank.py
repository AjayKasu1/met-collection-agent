"""Load a pinned Apache-licensed ONNX cross-encoder for local candidate reranking."""

import math
from collections.abc import Sequence
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from met_agent.retrieval.embeddings import EmbeddingError

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"


class LocalReranker:
    """Use English query rewrites for the English records; keep inference off provider APIs."""

    def __init__(self, cache: Path, *, threads: int = 4) -> None:
        if MODEL not in {entry["model"] for entry in TextCrossEncoder.list_supported_models()}:
            raise EmbeddingError("Configured reranker is absent from the installed registry")
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
                float(score) for score in self.model.rerank(query, list(documents), batch_size=8)
            ]
        except Exception as error:
            raise EmbeddingError(f"Local reranking failed ({type(error).__name__})") from None
        if len(scores) != len(documents) or not all(math.isfinite(value) for value in scores):
            raise EmbeddingError("Reranker returned invalid scores")
        return scores
