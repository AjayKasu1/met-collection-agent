"""Count E5 passage tokens during preparation without loading the inference model."""

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

from met_agent.retrieval.embeddings import EmbeddingError
from met_agent.retrieval.local_embeddings import LOCAL_WEIGHTS_REPO, LOCAL_WEIGHTS_REVISION


def local_passage_counter(cache_dir: Path) -> Callable[[str], int]:
    """Use the pinned tokenizer, including the passage prefix and special tokens."""
    try:
        try:
            path = hf_hub_download(
                LOCAL_WEIGHTS_REPO,
                "tokenizer.json",
                revision=LOCAL_WEIGHTS_REVISION,
                token=False,
                cache_dir=str(cache_dir / "hf"),
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            path = hf_hub_download(
                LOCAL_WEIGHTS_REPO,
                "tokenizer.json",
                revision=LOCAL_WEIGHTS_REVISION,
                token=False,
                cache_dir=str(cache_dir / "hf"),
            )
        tokenizer = Tokenizer.from_file(path)
        tokenizer.no_truncation()
        tokenizer.no_padding()
    except Exception as error:
        raise EmbeddingError(
            f"Visitor tokenizer initialization failed ({type(error).__name__})"
        ) from None

    def count(text: str) -> int:
        return len(tokenizer.encode("passage: " + text).ids)

    return count
