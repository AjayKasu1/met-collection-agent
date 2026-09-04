"""Run the registered multilingual E5 model from pinned ONNX weights with explicit prefixes."""

import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastembed import TextEmbedding
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

from met_agent.retrieval.embeddings import EmbeddingError
from met_agent.retrieval.schema import EmbeddingIdentity

LOCAL_MODEL = "intfloat/multilingual-e5-large"
LOCAL_DIMENSIONS = 1024
LOCAL_TOKEN_LIMIT = 512
LOCAL_WEIGHTS_REPO = "qdrant/multilingual-e5-large-onnx"
LOCAL_WEIGHTS_REVISION = "66076b8dc6e367337e3e90e6fb309fb0f3addaf6"
LOCAL_FILES = (
    "model.onnx",
    "model.onnx_data",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


class LocalEmbedder:
    """Preserve E5's query/passage contract and reject configuration or output drift."""

    def __init__(self, model: str, dimensions: int, cache_dir: Path, *, threads: int = 4) -> None:
        if model != LOCAL_MODEL or dimensions != LOCAL_DIMENSIONS:
            raise EmbeddingError(
                "Local embeddings require intfloat/multilingual-e5-large at 1024 dimensions"
            )
        registered = {
            entry["model"]: entry["dim"] for entry in TextEmbedding.list_supported_models()
        }
        if registered.get(model) != dimensions:
            raise EmbeddingError("The configured local model is absent from FastEmbed's registry")
        self.dimensions = dimensions
        self.identity = EmbeddingIdentity(provider="local", model=model, dimensions=dimensions)
        try:
            try:
                weights = snapshot_download(
                    LOCAL_WEIGHTS_REPO,
                    revision=LOCAL_WEIGHTS_REVISION,
                    token=False,
                    cache_dir=str(cache_dir / "hf"),
                    allow_patterns=list(LOCAL_FILES),
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                weights = snapshot_download(
                    LOCAL_WEIGHTS_REPO,
                    revision=LOCAL_WEIGHTS_REVISION,
                    token=False,
                    cache_dir=str(cache_dir / "hf"),
                    allow_patterns=list(LOCAL_FILES),
                    max_workers=2,
                )
            self.tokenizer = Tokenizer.from_file(str(Path(weights) / "tokenizer.json"))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            with warnings.catch_warnings():
                # FastEmbed warns about its pre-0.6 pooling change; this project pins 0.8.0.
                warnings.filterwarnings("ignore", message="The model .* now uses mean pooling")
                self.model = TextEmbedding(
                    model_name=model,
                    cache_dir=str(cache_dir),
                    threads=threads,
                    specific_model_path=weights,
                    providers=["CPUExecutionProvider"],
                )
        except Exception as error:
            raise EmbeddingError(
                f"Local embedding initialization failed ({type(error).__name__})"
            ) from None

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        prefix = "query: " if purpose == "query" else "passage: "
        inputs = [prefix + text for text in texts]
        if any(len(item.ids) > LOCAL_TOKEN_LIMIT for item in self.tokenizer.encode_batch(inputs)):
            raise EmbeddingError("Local embedding input exceeds E5's 512-token limit")
        try:
            vectors = [
                vector.tolist() for vector in self.model.embed(inputs, batch_size=len(texts))
            ]
        except Exception as error:
            raise EmbeddingError(
                f"Local embedding inference failed ({type(error).__name__})"
            ) from None
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions
            or not all(math.isfinite(value) for value in vector)
            or not math.isfinite(math.hypot(*vector))
            or math.hypot(*vector) == 0
            for vector in vectors
        ):
            raise EmbeddingError(
                "Local embedding response failed shape and finite-value validation"
            )
        normalized = []
        for vector in vectors:
            norm = math.hypot(*vector)
            normalized.append([value / norm for value in vector])
        return normalized
